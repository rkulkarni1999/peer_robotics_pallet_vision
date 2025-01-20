import argparse
from ultralytics import YOLO

def main(args):

    model = YOLO(args.model_path)

    model.export(
        format=args.format,       # Export format (default: "engine")
        device=args.device,       # Device (default: None)
        int8=args.int8,           # Enable INT8 precision (default: True)
        half=args.half            # Enable FP16 precision (default: False)
    )
    print(f"Model exported successfully in {args.format} format.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT format")
    
    # Model path argument
    parser.add_argument(
        "--model-path",
        type=str,
        default="yolo/models/final/detection/detection_fp32.pt", 
        help="Path to the YOLO model file. Default: 'yolo/models/final/segmentation/segmentation_fp16.pt'."
    )
    
    # Export format argument
    parser.add_argument(
        "--format",
        type=str,
        default="engine",
        choices=["onnx", "engine"],
        help="Export format: 'onnx' or 'engine'. Default: 'engine'."
    )
    
    # Device argument
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for export (e.g., 'cuda', 'dla:0'). Default: None (uses the system default)."
    )
    
    # INT8 precision argument
    parser.add_argument(
        "--int8",
        action="store_true",
        default=True,  
        help="Enable INT8 precision for TensorRT export. Default: True."
    )
    
    # FP16 precision argument
    parser.add_argument(
        "--half",
        action="store_true",
        default=False, 
        help="Enable FP16 precision for TensorRT export. Default: False."
    )
    
    args = parser.parse_args()
    main(args)
