import argparse
import os
from ultralytics import YOLO

def main(args):

    model = YOLO(args.model_path).to("cuda:0")

    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()  # Sort files for consistent order
    image_files = image_files[:args.limit]  # Limit the number of images processed

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(args.image_dir, image_file)
        model.predict(source=image_path, save=True, conf=args.confidence)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLO Inference Script")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo/models/final/detection/detection_final.pt",
        help="Path to the YOLO model file."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/warehouse_youtube",
        help="Path to the directory containing the images."
    )
    
    # Confidence threshold
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="Confidence threshold for YOLO detection."
    )
    
    # Limit the number of images to process
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images to process. Set to None for all images."
    )
    
    args = parser.parse_args()
    main(args)