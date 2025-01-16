from ultralytics import YOLO
import os
import numpy as np
import cv2  

# Load a pretrained YOLO segmentation model
model = YOLO("yolo/models/pallet_segmentation.pt")  # Path to your YOLO segmentation model
model = model.to("cuda:0")  # Move model to GPU

# Directory containing images
image_dir = "data/warehouse_images"  # Path to the directory with input images
# image_dir = "datasets/wooden_pallet_dataset2/test/images"
# image_dir = "data/images/saved_images_from_rosbag"
output_dir_semantic = "yolo/inferences/yolo_semantic_segmentation/"  # Directory to save semantic overlay
output_dir_bbox = "yolo/inferences/yolo_bbox_overlay/"  # Directory to save bounding box overlay

# Ensure output directories exist
os.makedirs(output_dir_semantic, exist_ok=True)
os.makedirs(output_dir_bbox, exist_ok=True)

# Define class ID to label mapping
class_id_to_label = {0: "Ground", 1: "Pallet"}

# Define colors for each class (for masks and bounding boxes)
colors = [
    (0, 255, 0),  # Ground (green)
    (0, 0, 255),  # Pallet (blue)
]

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()  # Optional: sort files alphabetically

# Perform segmentation on each image
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    
    # Use model.predict to process the image for segmentation
    results = model.predict(source=image_path, conf=0.2, save=False)  # Do not save individual instance masks yet
    
    # Read the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # 1. Semantic Segmentation Overlay
    
    semantic_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

    for i, mask in enumerate(results[0].masks.data):  # `results[0].masks.data` contains masks as a list
        mask_np = mask.cpu().numpy()  # Convert mask to a NumPy array
        mask_np = (mask_np > 0.5).astype(np.uint8)  # Binarize mask
        mask_resized = cv2.resize(mask_np, (semantic_mask.shape[1], semantic_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        cls_id = int(results[0].boxes.cls[i])  # Get class ID
        semantic_mask[mask_resized == 1] = cls_id + 1  # Avoid overlap by incrementing by 1

    # Create color overlay for semantic mask
    color_mask = np.zeros_like(original_image)
    for cls_id in range(1, len(colors) + 1):
        color_mask[semantic_mask == cls_id] = colors[cls_id - 1]

    # Blend original image with semantic mask
    overlayed_image_semantic = cv2.addWeighted(original_image, 0.6, color_mask, 0.4, 0)

    # Save semantic overlay
    semantic_output_path = os.path.join(output_dir_semantic, os.path.splitext(image_file)[0] + "_semantic.png")
    overlayed_image_semantic_bgr = cv2.cvtColor(overlayed_image_semantic, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    cv2.imwrite(semantic_output_path, overlayed_image_semantic_bgr)

    # 2. Bounding Box Overlay
    overlayed_image_bbox = original_image.copy()
    for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        
        cls_id = int(cls_id)
        
        if cls_id == 1:
            x1, y1, x2, y2 = map(int, box)  
            label = class_id_to_label[cls_id]
            color = colors[cls_id]
            cv2.rectangle(overlayed_image_bbox, (x1, y1), (x2, y2), color, thickness=4)
            cv2.putText(overlayed_image_bbox, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=2, lineType=cv2.LINE_AA)

    # Save bounding box overlay
    bbox_output_path = os.path.join(output_dir_bbox, os.path.splitext(image_file)[0] + "_bbox.png")
    overlayed_image_bbox_bgr = cv2.cvtColor(overlayed_image_bbox, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    cv2.imwrite(bbox_output_path, overlayed_image_bbox_bgr)

    print(f"Processed and saved: {semantic_output_path} and {bbox_output_path}")

print("Semantic segmentation and bounding box overlays complete!")
