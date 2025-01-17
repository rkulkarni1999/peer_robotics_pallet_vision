import glob
import time
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/segment/segmentation_train/weights/best.pt").to("cuda:0")
# model = YOLO("yolo/models/optimized_models_detection/optimized_detection.onnx").to("cuda:0")

# Perform a warm-up inference
warmup_image = "data/warehouse_data/image_0001.jpg"  # Replace with a valid image path
print(f"Performing warm-up inference on {warmup_image}...")
_ = model(warmup_image)

# Path to the folder containing test images
image_folder = "data/warehouse_data/*.jpg"  # Update with your folder path
images = glob.glob(image_folder)

# Check if folder is empty
if not images:
    print("No images found in the folder.")
    exit()

# Measure inference time for all images
total_time = 0.0
for img in images:
    start_time = time.time()
    _ = model(img)  # Perform inference (result is discarded for timing)
    end_time = time.time()
    total_time += (end_time - start_time)

# Calculate average inference time
average_time = total_time / len(images)

# Print results
print(f"Processed {len(images)} images.")
print(f"Average Inference Time: {average_time:.3f} seconds per image")

# import glob
# import time
# import onnxruntime as ort
# import numpy as np
# from PIL import Image

# # Path to the ONNX model
# model_path = "yolo/models/optimized_models_detection/optimized_detection.onnx"

# # Load the ONNX model with ONNX Runtime on GPU
# print(f"Loading ONNX model from: {model_path}")
# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # Use CUDA with fallback to CPU
# session = ort.InferenceSession(model_path, providers=providers)

# # Check if CUDAExecutionProvider is available
# available_providers = session.get_providers()
# if "CUDAExecutionProvider" not in available_providers:
#     print("WARNING: CUDAExecutionProvider is not available. The model will run on CPU.")

# # Get input and output names
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name

# # Preprocess the input image
# def preprocess_image(image_path):
#     image = Image.open(image_path)

#     # Resize to the model's input size
#     image = image.resize((640, 640))  # Adjust size if necessary
#     image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
#     image = np.transpose(image, (2, 0, 1))  # Change shape to (C, H, W)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# # Path to the folder containing test images
# image_folder = "data/warehouse_data/*.jpg"  # Update with your folder path
# images = glob.glob(image_folder)

# # Check if the folder is empty
# if not images:
#     print("No images found in the folder.")
#     exit()

# # Perform a warm-up inference
# warmup_image = images[0]
# print(f"Performing warm-up inference on {warmup_image}...")
# warmup_data = preprocess_image(warmup_image)
# _ = session.run([output_name], {input_name: warmup_data})

# # Measure inference time for all images
# total_time = 0.0
# for img_path in images:
#     input_data = preprocess_image(img_path)
#     start_time = time.time()
#     _ = session.run([output_name], {input_name: input_data})  # Perform inference
#     end_time = time.time()
#     total_time += (end_time - start_time)

# # Calculate average inference time
# average_time = total_time / len(images)

# # Print results
# print(f"Processed {len(images)} images.")
# print(f"Average Inference Time: {average_time:.3f} seconds per image")
