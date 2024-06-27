from os import path
import torch
from ultralytics import YOLO


def load_model(device: str):
    model = YOLO("/wheresmycar/model_training/runs/detect/train5/weights/best.pt")
    model.to(device)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    img_path = "/wheresmycar/model_training/data/images/val/images/"
    # img_path = "/wheresmycar/model_training/data/images/val/images/Cars417_jpg.rf.3ba955c4339fa1f5a957cca648f2d464.jpg"
    output = model.predict(img_path, conf=0.5)

    # print("Results", output)
    file_name = path.basename(img_path)
    # print(output)
    for r in output:
        # print(r)
        # print("boxes")
        # print(r.boxes.data.tolist())
        if len(r.boxes) > 0:
            f_name = path.basename(r.path)
            r.save_crop("./output/{}".format(f_name))


if __name__ == "__main__":
    main()
