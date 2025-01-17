from ultralytics import YOLO
import torch
import subprocess 
from torch.quantization import quantize_dynamic
import os

########################
# Train Model with FP16
########################
model = YOLO("yolo/models/yolo11n-seg.pt").to("cuda:0")
model.half()
model.train(
    data="datasets/pallet_ground_segmentation_dataset/data.yaml",  # Path to dataset YAML file
    imgsz=640,                # Input image size
    batch=8,                  # Batch size
    epochs=200,               # Number of epochs
    device='0',               # Use GPU device 0
    save_period=50            # Save model weights every 50 epochs
)

################################
# Quantize model after training
################################
quantized_model = quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8)
quantized_model_path = "yolo/models/optimized_models_segmentation/quantized_yolo_model.pt"
os.makedirs(quantized_model_path, exist_ok=True)
torch.save(quantized_model.state_dict(), quantized_model_path)
print(f"Quantized model saved to {quantized_model_path}")

#######################
# Export model to ONNX
#######################
onnx_model_path = "yolo/models/optimized_models_segmentation/model_fp16.onnx"
os.makedirs(onnx_model_path, exist_ok=True)
model.export(onnx_model_path, half=True)  # Export to ONNX with FP16 precision
print(f"ONNX model exported to {onnx_model_path}")

######################
# Convert to tensorRT
######################
tensorrt_model_path = "yolo/models/optimized_models_segmentation/model_fp16_int8.trt"
os.makedirs(tensorrt_model_path, exist_ok=True)
subprocess.run([
    "trtexec",
    f"--onnx={onnx_model_path}",  # Input ONNX model
    f"--saveEngine={tensorrt_model_path}",  # Output TensorRT engine
    "--fp16",  
    "--int8",  
    "--explicitBatch"  # Enable explicit batch mode for TensorRT
])

print(f"TensorRT model saved to {tensorrt_model_path}")
