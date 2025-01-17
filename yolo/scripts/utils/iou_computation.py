import numpy as np
from shapely.geometry import Polygon

def compute_polygon_iou(polygon1, polygon2):
    """
    Computes Intersection over Union (IoU) between two polygons using Shapely.
    
    Args:
        polygon1 (list): List of (x, y) points for the first polygon.
        polygon2 (list): List of (x, y) points for the second polygon.
        
    Returns:
        float: IoU value.
    """
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    return intersection / union if union > 0 else 0.0

def parse_annotations(txt_path):
    """
    Parses annotations from the custom .txt file.
    
    Args:
        txt_path (str): Path to the .txt file.
        
    Returns:
        list: List of tuples with (class_id, polygon_points).
    """
    annotations = []
    with open(txt_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            points = list(map(float, parts[1:]))
            polygon_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
            annotations.append((class_id, polygon_points))
    return annotations

def predict_and_evaluate(model_path, image_path, txt_path):
    """
    Runs YOLO inference and computes IoU with ground truth polygons.
    
    Args:
        model_path (str): Path to the YOLO model file.
        image_path (str): Path to the input image.
        txt_path (str): Path to the ground truth .txt file.
        
    Returns:
        list: List of IoUs for each prediction-ground truth pair.
    """
    from ultralytics import YOLO
    
    # Load YOLO model
    model = YOLO(model_path)

    # Run inference
    results = model(image_path)
    
    # Parse ground truth polygons
    ground_truths = parse_annotations(txt_path)

    # Get predicted polygons (convert boxes to polygons for comparison)
    pred_bboxes = results[0].boxes.xyxy.cpu().numpy()  # Predicted bounding boxes
    pred_polygons = [
        [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])] 
        for box in pred_bboxes
    ]

    # Compute IoU for each prediction-ground truth pair
    iou_scores = []
    for _, gt_polygon in ground_truths:
        for pred_polygon in pred_polygons:
            iou = compute_polygon_iou(gt_polygon, pred_polygon)
            iou_scores.append(iou)

    return iou_scores

if __name__ == "__main__":
    # Example usage
    model_path = "yolo/models/final/segmentation/segmentation_final.pt"  # Path to your YOLO model
    image_path = "datasets/pallet_segmentation_dataset/test/images/15_jpg.rf.2bd74bf3bf6d153af4a9ce16d2925696.jpg"  # Path to the input image
    ground_truth_path = "datasets/pallet_segmentation_dataset/test/labels/15_jpg.rf.2bd74bf3bf6d153af4a9ce16d2925696.txt"  # Path to the ground truth mask
    
    
    ious = predict_and_evaluate(model_path, image_path,ground_truth_path)
    print(f"IoU scores: {ious}")
    print(f"Mean IoU: {np.mean(ious):.4f}")



