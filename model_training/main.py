"""Script for training the model."""
from ultralytics import YOLO


def main():
    """main function
    """
    # Original model
    # model = YOLO("yolov8n.yaml")
    model = YOLO("runs/detect/train5/weights/best.pt")
    model.train(data="config.yaml", epochs=20)


if __name__ == "__main__":
    main()
