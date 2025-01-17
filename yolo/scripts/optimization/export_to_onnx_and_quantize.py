import argparse
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

def main(args):
    # Load the YOLO model
    model = YOLO(args.model_path)
    
    # Export the model to ONNX format
    export_path = model.export(format=args.export_format)
    print(f"Model exported to {export_path}")
    
    # Perform ONNX model quantization
    quantize_dynamic(
        args.onnx_input_path,
        args.onnx_output_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Quantized model saved to {args.onnx_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to export YOLO model to ONNX and perform quantization")
    
    # Add arguments
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="runs/detect/detection_train_optimized/weights/best.pt",
        help="Path to the YOLO model file (.pt)"
    )
    parser.add_argument(
        "--export_format", 
        type=str, 
        default="onnx", 
        help="Format to export the model (default: onnx)"
    )
    parser.add_argument(
        "--onnx_input_path", 
        type=str, 
        default="runs/detect/detection_train_optimized/weights/best.onnx",
        help="Path to the exported ONNX model file"
    )
    parser.add_argument(
        "--onnx_output_path", 
        type=str, 
        default="yolo/models/optimized_models_detection/optimized_detection.onnx",
        help="Path to save the quantized ONNX model"
    )
    
    args = parser.parse_args()
    main(args)
