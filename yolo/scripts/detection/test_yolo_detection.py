from ultralytics import YOLO
import os

# Load a pretrained YOLO model
# model = YOLO("yolo/models/best_1_200.pt")  # detection model
# model = YOLO("yolo/models/best_2_150.pt")  # detection model
# model = YOLO("yolo/models/best.pt")  # detection model
# model = YOLO("yolo/models/yolo11n.pt")  # detection model

# model = YOLO("yolo/models/final/pallet_detector_1_150.pt").to("cuda:0")
model = YOLO("yolo/models/final/pallet_detector_1_200.pt").to("cuda:0")


# Directory containing images
# image_dir = "data/images/Pallets_10"
image_dir = "data/warehouse_data"

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()  
image_files = image_files  # Limit to the first 10 images

# Perform inference on each image
for idx, image_file in enumerate(image_files):
    
    image_path = os.path.join(image_dir, image_file)
    model.predict(source=image_path, save=True, conf=0.2)

print("Processing complete!")
