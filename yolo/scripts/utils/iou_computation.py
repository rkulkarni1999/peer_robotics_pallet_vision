from ultralytics import YOLO
import numpy as np

def compute_raw_iou():
    """
    Compute raw IoU for each predicted mask and its ground truth.
    """
    # Load your trained segmentation model
    model = YOLO('yolo/models/final/segmentation/segmentation_final.pt')
    
    # Validate the model on your validation dataset
    results = model.val(data='datasets/pallet-segmentation-dataset.v1i.yolov11/data.yaml', save_json=True)
    
    # Initialize dictionary to store IoU values per class
    iou_per_class = {cls_id: [] for cls_id in range(len(results.names))}
    
    # Loop through predictions and ground truths
    for pred, gt in zip(results.pred, results.gt):
        for cls_id in range(len(results.names)):  # Iterate over classes
            # Get predicted and ground truth masks for the current class
            pred_mask = (pred.masks == cls_id).cpu().numpy()  # Predicted mask for the class
            gt_mask = (gt.masks == cls_id).cpu().numpy()      # Ground truth mask for the class
            
            # Compute Intersection and Union
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            # Calculate IoU
            iou = intersection / union if union > 0 else 0
            iou_per_class[cls_id].append(iou)
    
    # Aggregate results: compute mean IoU per class
    mean_iou_per_class = {cls_id: np.mean(ious) if ious else 0 for cls_id, ious in iou_per_class.items()}
    
    # Print results
    print("Raw IoU Results:")
    for cls_id, mean_value in mean_iou_per_class.items():
        print(f"Class {cls_id} ({results.names[cls_id]}): Mean IoU = {mean_value:.4f}")
    
    print("\nRaw IoUs for Each Class:")
    for cls_id, ious in iou_per_class.items():
        print(f"Class {cls_id} ({results.names[cls_id]}): IoUs = {ious}")

# Execute the function
compute_raw_iou()
