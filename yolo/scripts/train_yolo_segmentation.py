from ultralytics import YOLO

# Load a YOLO segmentation model
model = YOLO("yolo/models/yolo11n-seg.pt")  # Use a segmentation model (e.g., YOLOv8 for segmentation)
model = model.to("cuda:0")  # Move the model to GPU for training

# Train the model on your pallet-ground segmentation dataset
model.train(
    data="datasets/pallet_ground_segmentation_dataset/data.yaml",  # Path to your dataset configuration file
    imgsz=640,  # Image size
    batch=16,  # Batch size
    epochs=250,  # Number of epochs
    device='0'  # Specify GPU device
)
