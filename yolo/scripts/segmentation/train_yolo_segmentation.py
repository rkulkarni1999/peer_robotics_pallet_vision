from ultralytics import YOLO

model = YOLO("yolo/models/yolo11n-seg.pt").to("cuda:0")
  
model.train(data="./datasets/pallet_segmentation_dataset/data.yaml",
            imgsz=640,
            batch=8,
            epochs=200,
            device='0',
            save_period = 50,
            )

