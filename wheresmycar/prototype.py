from os import path
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image



class ObjectDetector():
    def __init__(self):
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.9)
        self.model.eval()

    def get_predictions(self, paths):
        batch = []
        images = []
        for image_path in paths:
            file_name = path.basename(image_path)
            print("Processing", file_name)
            img = read_image(image_path)
            images.append((img, file_name))
            preprocessed_img = self.preprocess(img)
            batch.append(preprocessed_img)

        print("Predicting")
        return self.model(batch), images

    def draw_boxes(self, predictions, images):
        print("drawing boxes")
        for prediction, img_with_file_name in zip(predictions, images):
            img, file_name = img_with_file_name
            labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
            box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                    labels=labels,
                                    colors="red",
                                    width=4, font_size=30)

            im = to_pil_image(box.detach())
            output_path = f"../outputs/{file_name}"
            im.save(output_path)
            print("Saved", output_path)

    def detect(self, file_paths):
        predictions, images = self.get_predictions(file_paths)
        self.draw_boxes(predictions, images)


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
                            width=2, font_size=30)

    print("to img")
    im = to_pil_image(box.detach())
    im.save(f"../outputs/{name}")


if __name__ == "__main__":
    detector = ObjectDetector()

    file_paths = [
        "../assets/maseratti.jpg",
        "../assets/multiple.jpg",
    ]

    detector.detect(file_paths)


def old_test():
    # Initialize model with best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()

    images = [
        "../assets/maseratti.jpg",
        "../assets/multiple.jpg",
    ]
    for img_path in images:
        print("Processing", img_path)
        process_single_image(img_path, model, weights)
