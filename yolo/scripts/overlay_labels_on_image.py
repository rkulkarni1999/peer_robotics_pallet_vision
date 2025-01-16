import cv2
import os

# Paths to the image and labels file
image_path = "data/wooden_pallet_dataset/train/images/0002_jpg.rf.c0144f5423f02d46babf23e2f9caa35f.jpg"
labels_path = "data/wooden_pallet_dataset/train/labels/0002_jpg.rf.c0144f5423f02d46babf23e2f9caa35f.txt"
output_path = "yolo/inferences/yolo_detection/image_with_labels.jpg"

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Get image dimensions
height, width, _ = image.shape

# Load the labels file
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels file not found at {labels_path}")

with open(labels_path, "r") as f:
    lines = f.readlines()

# Parse the labels and draw bounding boxes
for line in lines:
    label_data = line.strip().split()
    class_id = int(label_data[0])  # Class ID
    x_center, y_center, box_width, box_height = map(float, label_data[1:])

    # Convert YOLO normalized coordinates to absolute pixel values
    x_min = int((x_center - box_width / 2) * width)
    y_min = int((y_center - box_height / 2) * height)
    x_max = int((x_center + box_width / 2) * width)
    y_max = int((y_center + box_height / 2) * height)

    # Draw the bounding box on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Add class ID text (adjusting placement and font size)
    font_scale = 0.6  # Adjust font scale for better visibility
    font_thickness = 2  # Thicker text for better readability
    text_size = cv2.getTextSize(f"Class {class_id}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x = x_min
    text_y = max(y_min - 10, text_size[1])  # Ensure text is above the box and within image bounds
    cv2.putText(
        image,
        f"Class {class_id}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),  # Green text color
        font_thickness,
        lineType=cv2.LINE_AA,
    )

# Save and display the image with bounding boxes
cv2.imwrite(output_path, image)
print(f"Image with labels saved at {output_path}")