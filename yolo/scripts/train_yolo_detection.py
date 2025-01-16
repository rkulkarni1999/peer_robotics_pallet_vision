from ultralytics import YOLO

model = YOLO("yolo/models/yolo11n.pt") # detection model
model = model.to("cuda:0")

# model.train(data="./datasets/wooden_pallet_dataset1/data.yaml", imgsz=640, batch=8, epochs=200, device='0')
# model.train(data="./datasets/wooden_pallet_dataset2/data.yaml", imgsz=640, batch=16, epochs=250, device='0')
model.train(data="./datasets/wooden_pallet_dataset3/data.yaml", imgsz=640, batch=16, epochs=150, device='0')

