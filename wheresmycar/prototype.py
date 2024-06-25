from os import path
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image



# Initialize inference transforms
def process_single_image(img_path, model, weights):
    name = path.basename(img_path)
    img = read_image(img_path)
    preprocess = weights.transforms()

    # Apply inference preprocessing transforms
    print("preprocess")
    batch = [preprocess(img)]

    # Use model and visualize prediction

    print("predict")
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]

    print('draw')
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=30)

    print("to img")
    im = to_pil_image(box.detach())
    im.save(f"../outputs/{name}")


if __name__ == "__main__":
    # Initialize model with best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    images = [
        "../assets/maseratti.jpg",
        "../assets/multiple.jpg",
    ]
    for img_path in images:
        print("Processing", img_path)
        process_single_image(img_path, model, weights)
