# wheresmycar

First attempt at utilizing YOLOv8 model for vehicle number plate detection.

> [!NOTE]
> This project is for education and skills presentation purposes.


# Motivation

Gain practical knowledge with machine learning technologies in real-world example.

The main goal was to gain hands-on experience in machine learning project utilizing PyTorch library and several technologies to improve software engineering skills.

# About the project

Small Python package providing a class for object detection, utilizing YOLOv8 model which was trained to detect number plates on vehicles.

# Documentation

<a id="plate_detector"></a>

# plate\_detector

plate_detector module

<a id="plate_detector.PlateDetector"></a>

## PlateDetector Objects

```python
class PlateDetector()
```

Class for Vehicle Number Plate Detection based on pretrained YOLOv8 model.

<a id="plate_detector.PlateDetector.load_model"></a>

#### load\_model

```python
def load_model(device: str) -> YOLO
```

Function to load pretrained model.
params:
- device <str>: device on which the model should run
returns:
- <ultralytics.YOLO>: pretrained YOLOv8 model

<a id="plate_detector.PlateDetector.model"></a>

#### model

```python
@property
def model() -> YOLO
```

Access pretrained model
returns:
- <ultralytics.YOLO>: pretrained YOLOv8 model

<a id="plate_detector.PlateDetector.get_device"></a>

#### get\_device

```python
def get_device(enable_cuda: bool) -> str
```

Gets target device for inference.
params:
- enable_cude <bool>: if True, will return cuda if available
returns:
- <str> either cuda or cpu

<a id="plate_detector.PlateDetector.detect"></a>

#### detect

```python
def detect(target_path: str, conf: float = 0.5, **kwargs) -> Results
```

Get predictions on given input.
params:
- target_path <str>: path to directory with images or image's file path
- conf <float>: minimum confidence threshold for detection
returns:
- <ultralytics.engine.results.Results>: inference results

