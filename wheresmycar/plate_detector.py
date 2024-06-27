"""plate_detector module"""
import os
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


class PlateDetector():
    """Class for Vehicle Number Plate Detection based on pretrained YOLOv8 model.
    """
    DEVICE_CPU = "cpu"
    DEVICE_CUDA = "cuda"

    def __init__(self, enable_cuda=True):
        self.device = self.get_device(enable_cuda)
        self._model = self.load_model(self.device)

    def load_model(self, device: str) -> YOLO:
        """
        Function to load pretrained model.
        params:
        - device <str>: device on which the model should run
        returns:
        - <ultralytics.YOLO>: pretrained YOLOv8 model
        """
        weights_path: str = self._get_weights_path()
        model = YOLO(weights_path)
        model.to(device)
        return model

    def _get_weights_path(self) -> str:
        dirname = os.path.dirname(__file__)
        return os.path.join(dirname, "weights/best.pt")

    @property
    def model(self) -> YOLO:
        """
        Access pretrained model
        returns:
        - <ultralytics.YOLO>: pretrained YOLOv8 model
        """
        return self._model

    def get_device(self, enable_cuda: bool) -> str:
        """
        Gets target device for inference.
        params:
        - enable_cude <bool>: if True, will return cuda if available
        returns:
        - <str> either cuda or cpu
        """
        if enable_cuda:
            return self.DEVICE_CUDA if torch.cuda.is_available() else self.DEVICE_CPU
        return self.DEVICE_CPU

    def detect(self, target_path: str, conf: float = 0.5, **kwargs) -> Results:
        """
        Get predictions on given input.
        params:
        - target_path <str>: path to directory with images or image's file path
        - conf <float>: minimum confidence threshold for detection
        returns:
        - <ultralytics.engine.results.Results>: inference results
        """
        return self._model.predict(target_path, conf=conf, **kwargs)
