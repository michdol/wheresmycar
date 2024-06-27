from ultralytics import YOLO

# model = YOLO("yolov8n.yaml")
model = YOLO("runs/detect/train5/weights/best.pt")

results = model.train(data="config.yaml", epochs=20)
