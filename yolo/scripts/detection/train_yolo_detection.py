import argparse
from ultralytics import YOLO
import torch

def main(args):
    # Load YOLO model and move it to GPU
    model = YOLO(args.model_path).to("cuda:0")
    if args.use_fp16:
        model.half()  

    # Train the model
    model.train(
        data=args.data_path,       
        imgsz=args.image_size,     
        batch=args.batch_size,     
        epochs=args.epochs,        
        device=args.device,        
        save_period=args.save_period  
    )

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    
    # Model path argument
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo/models/yolo11n.pt",
        help="Path to the YOLO model file."
    )
    
    # Dataset and training arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="./datasets/pallet_detection_dataset/data.yaml",
        help="Path to the dataset YAML file."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Input image size for training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use for training (e.g., '0' for GPU 0)."
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=25,
        help="Save model weights every specified number of epochs."
    )
    
    # Optional FP16 precision argument
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use FP16 precision for training."
    )
    
    args = parser.parse_args()
    main(args)
