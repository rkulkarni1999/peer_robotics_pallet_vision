import os
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

# Define process_results function
def process_results(image, results):
    # Initialize a semantic mask for each class
    semantic_masks = {
        0: np.zeros(image.shape[:2], dtype=np.uint8),  # Ground
        1: np.zeros(image.shape[:2], dtype=np.uint8),  # Pallet
    }

    # Combine masks for the same class
    for mask, cls_id in zip(results[0].masks.data, results[0].boxes.cls):
        cls_id = int(cls_id)
        mask_np = mask.cpu().numpy().astype(np.uint8)  # Convert to binary mask (0 or 1)
        mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        semantic_masks[cls_id] = cv2.bitwise_or(semantic_masks[cls_id], mask_resized)

    # Overlay each class mask and label the entire region
    for cls_id, semantic_mask in semantic_masks.items():
        if np.any(semantic_mask):  # Check if the mask contains any pixels
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # Colors: Ground (green), Pallet (blue)
            label = "Ground" if cls_id == 0 else "Pallet"

            # Create a colored overlay for the mask
            overlay = np.zeros_like(image, dtype=np.uint8)
            overlay[semantic_mask == 1] = color

            # Blend the overlay with the original image
            image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

            # Calculate centroid for labeling
            moments = cv2.moments(semantic_mask.astype(np.uint8))
            if moments["m00"] > 0:  # Ensure the area is not zero
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])
                cv2.putText(
                    image,
                    label,
                    (centroid_x, centroid_y),  # (x, y) format for OpenCV
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (255, 255, 255),  # White text
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
    return image

def main(args):
    # Load the YOLO segmentation model
    model = YOLO(args.model_path)
    model = model.to("cuda:0")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Get a list of image files in the directory
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()  # Optional: sort files alphabetically

    # Perform segmentation on each image
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(args.image_dir, image_file)

        # Predict segmentation
        results = model.predict(source=image_path, conf=args.confidence, save=False)

        # Read the original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Process results to create semantic segmentation overlay
        semantic_result = process_results(original_image, results)

        # Save semantic overlay
        semantic_output_path = os.path.join(args.output_dir, os.path.splitext(image_file)[0] + "_semantic.png")
        semantic_result_bgr = cv2.cvtColor(semantic_result, cv2.COLOR_RGB2BGR)  # Convert back to BGR
        cv2.imwrite(semantic_output_path, semantic_result_bgr)

        print(f"Processed and saved: {semantic_output_path}")

    print("Semantic segmentation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation using YOLOv8")
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo/models/final/segmentation/segmentation_final.pt",
        help="Path to the YOLO segmentation model",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/warehouse_data",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="yolo/inferences/segmentation/yolo_semantic_segmentation/",
        help="Directory to save the segmentation results",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="Confidence threshold for YOLO predictions",
    )
    args = parser.parse_args()

    main(args)
