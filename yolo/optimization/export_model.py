from ultralytics import YOLO

model = YOLO("yolo/models/pallet_segmentation.pt")
model.export(format="onnx", opset=12) 

