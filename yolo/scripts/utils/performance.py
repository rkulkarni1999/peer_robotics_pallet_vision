import argparse
from ultralytics import YOLO

def segmentation_performance(model_path, data_yaml, img_size=640, batch_size=16, plots=False):
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=img_size, split='test', batch=batch_size, plots=plots, verbose=False)
    seg_metrics = results.seg
    metrics = {
        'mIoU': seg_metrics.maps.mean(),
        'mIoU_50': seg_metrics.map50.mean(),
        'precision': seg_metrics.p.mean(),
        'recall': seg_metrics.r.mean()
    }
    return metrics

def detection_performance(model_path, data_yaml, img_size=640, batch_size=16, plots=True):
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=img_size, split='test', batch=batch_size, plots=plots)
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model performance.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLO model weights file.")
    parser.add_argument("--data-yaml", type=str, required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for validation (default: 640).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for validation (default: 16).")
    parser.add_argument("--plots", action="store_true", help="Generate and save plots.")
    parser.add_argument("--task", type=str, choices=["segmentation", "detection"], required=True, help="Specify the task: 'segmentation' or 'detection'.")
    args = parser.parse_args()

    if args.task == "segmentation":
        metrics = segmentation_performance(
            model_path=args.model_path,
            data_yaml=args.data_yaml,
            img_size=args.img_size,
            batch_size=args.batch_size,
            plots=args.plots
        )
        print(f"Segmentation Metrics:\n"
              f"mIoU (0.50-0.95): {metrics['mIoU']:.4f}\n"
              f"mIoU (0.50): {metrics['mIoU_50']:.4f}\n"
              f"Mean Precision: {metrics['precision']:.4f}\n"
              f"Mean Recall: {metrics['recall']:.4f}")
    
    elif args.task == "detection":
        print("Detection Metrics:")
        detection_performance(
            model_path=args.model_path,
            data_yaml=args.data_yaml,
            img_size=args.img_size,
            batch_size=args.batch_size,
            plots=args.plots
        )


# Segmentation FULL PRECISION

# Speed: 0.2ms preprocess, 2.2ms inference, 0.0ms loss, 0.7ms postprocess per image
# Segmentation Metrics:
# mIoU (0.50-0.95): 0.7668
# mIoU (0.50): 0.9546
# Mean Precision: 0.9239
# Mean Recall: 0.9240


# Segmentation HALF PRECISION

# Speed: 0.2ms preprocess, 2.7ms inference, 0.0ms loss, 0.6ms postprocess per image
# Segmentation Metrics:
# mIoU (0.50-0.95): 0.7905
# mIoU (0.50): 0.9609
# Mean Precision: 0.9607
# Mean Recall: 0.9276


# Detection hal precision

# Speed: 0.2ms preprocess, 2.5ms inference, 0.0ms loss, 1.2ms postprocess per image
# mAP50: 0.7929262822118579
# mAP50-95: 0.5674033615455085
# Precision: [    0.81439]
# Recall: [    0.70399]

# Detection Full Precision

# Speed: 0.2ms preprocess, 2.6ms inference, 0.0ms loss, 1.1ms postprocess per image
# mAP50: 0.7638162085740027
# mAP50-95: 0.5448061679553133
# Precision: [    0.70497]
# Recall: [    0.73134]













# import glob
# import time
# from ultralytics import YOLO

# # Load the trained model
# model = YOLO("runs/segment/segmentation_train/weights/best.pt").to("cuda:0")
# # model = YOLO("yolo/models/optimized_models_detection/optimized_detection.onnx").to("cuda:0")

# # Perform a warm-up inference
# warmup_image = "data/warehouse_data/image_0001.jpg"  # Replace with a valid image path
# print(f"Performing warm-up inference on {warmup_image}...")
# _ = model(warmup_image)

# # Path to the folder containing test images
# image_folder = "data/warehouse_data/*.jpg"  # Update with your folder path
# images = glob.glob(image_folder)

# # Check if folder is empty
# if not images:
#     print("No images found in the folder.")
#     exit()

# # Measure inference time for all images
# total_time = 0.0
# for img in images:
#     start_time = time.time()
#     _ = model(img)  # Perform inference (result is discarded for timing)
#     end_time = time.time()
#     total_time += (end_time - start_time)

# # Calculate average inference time
# average_time = total_time / len(images)

# # Print results
# print(f"Processed {len(images)} images.")
# print(f"Average Inference Time: {average_time:.3f} seconds per image")


## inference_speed.py

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