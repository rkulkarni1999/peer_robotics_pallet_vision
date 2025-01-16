from ultralytics import YOLO
import os
import shutil
import cv2

# Load your trained YOLO model
model = YOLO("yolo/models/best.pt")  # detection model
model = model.to("cuda:0")

# Input folder containing images
input_image_dir = "data/images/Pallets"

# Output directories for images and labels
output_images_dir = "data/pallet_dataset/images"
output_labels_dir = "data/pallet_dataset/labels"

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Get a list of image files in the input directory
image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()  # Optional: sort files alphabetically

# Process each image
for image_file in image_files:
    image_path = os.path.join(input_image_dir, image_file)
    
    # Perform inference
    results = model.predict(source=image_path, save=False, conf=0.4)  # No saving of image overlays
    
    # Get image dimensions
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Generate corresponding label file
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(output_labels_dir, label_file)
    
    with open(label_path, "w") as f:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in absolute pixel coordinates (x1, y1, x2, y2)
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            
            for box, class_id in zip(boxes, class_ids):
                # Convert to YOLO format (x_center, y_center, width, height)
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                # Write to label file
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
    
    # Copy the original image to the output images directory
    shutil.copy(image_path, os.path.join(output_images_dir, image_file))
    print(f"Processed {image_file}: labels saved to {label_path}")

print("Dataset creation complete!")
