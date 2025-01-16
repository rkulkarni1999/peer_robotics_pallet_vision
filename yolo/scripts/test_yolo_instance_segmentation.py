from ultralytics import YOLO
import os

# Load a pretrained YOLO segmentation model
model = YOLO("yolo/models/pallet_segmentation.pt")  # Path to your YOLO segmentation model
model = model.to("cuda:0")  # Move model to GPU

# Directory containing images
image_dir = "data/warehouse_images"  # Path to the directory with input images
output_dir = "yolo/inferences/yolo_segmentation/"  # Directory to save segmentation results

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()  # Optional: sort files alphabetically

# Perform segmentation on each image
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    
    # Use model.predict to process the image for segmentation
    results = model.predict(source=image_path, save=True, conf=0.4)  # `save_mask=True` ensures masks are saved

    # Display results (optional)
    print(f"Processed and saved: {os.path.join(output_dir, image_file)}")

print("Segmentation complete!")
