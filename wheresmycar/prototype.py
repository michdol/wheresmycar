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
            print("Labels", labels)
            print(prediction)
            print()
            print(prediction["boxes"])
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



"""
Single prediction:
{'boxes': tensor([[161.3252,  89.2484, 195.5776, 116.5811],
        [ 98.7918, 112.9300, 137.8670, 146.8210],
        [ 23.1158, 111.9958,  59.3464, 142.6446],
        [176.1744,  31.9956, 196.6972,  45.3241],
        [207.9511,  40.2661, 242.6822,  77.2972],
        [207.2502, 140.7019, 259.8787, 180.7847],
        [171.9459,  44.5718, 195.0038,  60.4903],
        [ 38.5808,  90.3898,  70.8102, 115.3833],
        [190.1453,  28.9320, 208.9821,  40.7814],
        [133.1966,  58.8471, 157.2412,  79.7087],
        [ 41.5847,  68.5076,  67.8517,  93.0567],
        [ 87.5870,  53.7847, 107.4557,  69.5762],
        [ 43.9778,  58.1519,  62.9133,  69.5613]], grad_fn=<StackBackward0>), 'labels': tensor([3, 3, 3, 3, 8, 3, 3, 3, 3, 3, 3, 3, 3]), 'scores': tensor([0.9961, 0.9924, 0.9905, 0.9850, 0.9837, 0.9753, 0.9732, 0.9710, 0.9662,
        0.9565, 0.9365, 0.9251, 0.9007], grad_fn=<IndexBackward0>)}
"""