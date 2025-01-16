from ultralytics import YOLO
import os

# Load a pretrained YOLO model
# model = YOLO("yolo/models/best_1_200.pt")  # detection model
model = YOLO("yolo/models/best_2_150.pt")  # detection model
# model = YOLO("yolo/models/best.pt")  # detection model
# model = YOLO("yolo/models/yolo11n.pt")  # detection model

model = model.to("cuda:0")

# Directory containing images
# image_dir = "data/images/Pallets_10"
image_dir = "data/warehouse_images"
# image_dir = "datasets/wooden_pallet_dataset2/test/images"
# image_dir = "data/images/saved_images_from_rosbag"
output_dir = "yolo/inferences/yolo_detection/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()  # Optional: sort files alphabetically
image_files = image_files  # Limit to the first 10 images

# Perform inference on each image
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    
    # Use model.predict to process the image
    model.predict(source=image_path, save=True, conf=0.4)

    print(f"Processed and saved: {os.path.join(output_dir, image_file)}")

print("Processing complete!")
