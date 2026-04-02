# Computer Vision Usage Guide

This document explains how to use the `ComputerVision` module and work with the detection objects it returns.

## Overview

The `ComputerVision` module combines YOLOv8 object detection with the ZED SDK's 3D processing pipeline. It provides:
- **2D bounding boxes** (pixel coordinates)
- **3D world coordinates** (x, y, z positions)
- **Confidence scores** for each detection
- **Object classification** (person, cup, bottle, etc.)

## Basic Usage

```python
from bira_components.computer_vision import ComputerVision
from bira_components.camera import Camera

camera = Camera()
cv = ComputerVision(camera=camera)

# Capture frame from camera
camera.open()
with camera:
    if camera.grab():
        frame = camera.get_frame()
        
        # Perform detection
        sl_objects, detection_labels = cv.detect_objects(frame)
```

### Getting Label Names

The `ComputerVision` class provides two convenient methods to get human-readable label names from YOLO class IDs:

```python
# Get a single label name
label_name = cv.get_label_name(0)  # Returns 'person'
label_name = cv.get_label_name(2)  # Returns 'car'

# Get the complete label dictionary
label_dict = cv.label_dict  # Returns {0: 'person', 1: 'bicycle', 2: 'car', ...}
```

## Return Types

### `sl.Objects` Container
The first return value is a `pyzed.sl.Objects` instance, which is a **container** for all detections in a frame.

```python
sl_objects.object_list          # List[sl.ObjectData] - the actual detections
sl_objects.is_new()             # bool - whether this is fresh data
sl_objects.is_tracked()         # bool - whether object tracking is enabled
sl_objects.timestamp()          # Timestamp for frame synchronization
```

### `List[int]` Detection Labels
The second return value contains YOLO's raw class indices:

```python
detection_labels  # List[int] where each int is a class ID (0=person, 1=cup, etc.)
```

## Working with Individual Objects

Each object in `object_list` is a `pyzed.sl.ObjectData` instance with the following properties:

### Properties You'll Use Most

#### Position (World Coordinates)
```python
for obj in sl_objects.object_list:
    position = obj.position  # Returns [x, y, z] as numpy array
    # position is in the coordinate system defined by camera init parameters
    # Typically: x=right, y=down, z=forward
    
    x, y, z = position[0], position[1], position[2]
    print(f"Object at world position: ({x:.2f}, {y:.2f}, {z:.2f})")
```

#### 2D Bounding Box (Image Coordinates)
```python
for obj in sl_objects.object_list:
    bbox_2d = obj.bounding_box_2d  # Returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    # Points are in clockwise order starting from top-left:
    # A ------ B
    # | Object |
    # D ------ C
    
    top_left = bbox_2d[0]      # Point A
    top_right = bbox_2d[1]     # Point B
    bottom_right = bbox_2d[3]  # Point C
    bottom_left = bbox_2d[2]   # Point D
```

#### Confidence Score
```python
for obj in sl_objects.object_list:
    confidence = obj.confidence  # Returns float 0-100
    if confidence > 80:
        print(f"High confidence detection: {confidence}%")
```

#### Object Classification
```python
for obj in sl_objects.object_list:
    label = obj.label  # Returns sl.OBJECT_CLASS enum
    # Examples: PERSON, CUP, BOTTLE, BACKPACK, etc.
    
    # Convert to string for logging
    label_str = str(label)
    print(f"Detected: {label_str}")
```

#### Label Correspondence
The raw class IDs coming from YOLO are integers such as `0`, `1`, `2`, and their human-readable names come from the loaded model metadata in `model.names`.

```python
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")
print(model.names)
# {0: 'person', 1: 'bicycle', 2: 'car', ...}

label_id = 0
label_name = model.names[label_id]
print(label_name)  # person
```

In this project, `sl_objects.object_list` contains ZED `pyzed.sl.ObjectData` instances. For each object:
- `obj.label` is the ZED class enum, not the raw YOLO integer
- `obj.id` is the tracking ID and is often `-1` unless tracking is enabled
- `obj.position` is the approximate 3D position `[x, y, z]`

To check whether something was detected, test whether `sl_objects.object_list` is empty.

#### Object ID (Tracking)
```python
for obj in sl_objects.object_list:
    obj_id = obj.id  # Returns int
    # Returns -1 if object tracking is not enabled
    # If tracking is enabled, this stays consistent across frames for the same object
```

### Advanced Properties

| Property | Type | Description |
|----------|------|-------------|
| `position` | `[x, y, z]` | 3D centroid in world coordinates |
| `velocity` | `[vx, vy, vz]` | 3D velocity vector |
| `bounding_box` | 8 3D points | 3D bounding box (8 corner points) |
| `bounding_box_2d` | 4 2D points | 2D bounding box (4 corner points in pixels) |
| `dimensions` | `[width, height, length]` | 3D object dimensions |
| `tracking_state` | Enum | Current tracking state |
| `confidence` | float | Detection confidence 0-100 |
| `label` | Enum | Object class (PERSON, CUP, etc.) |
| `raw_label` | int | Plain integer label |

## Example: Using Objects in SLM_Manager

Here's how to use detection objects to inform language model decisions:

```python
def decide_object_action(sl_objects, detection_labels):
    """
    Determine if objects are detected and get their positions.
    """
    
    # Check if any objects detected
    if not sl_objects.object_list:
        return {
            "objects_found": False,
            "message": "No objects detected"
        }
    
    # Extract object info for decision making
    objects_info = []
    for obj in sl_objects.object_list:
        info = {
            "type": str(obj.label),
            "position": {
                "x": float(obj.position[0]),
                "y": float(obj.position[1]),
                "z": float(obj.position[2])
            },
            "confidence": float(obj.confidence),
            "distance": float(np.linalg.norm(obj.position))  # Distance from camera
        }
        objects_info.append(info)
    
    return {
        "objects_found": True,
        "count": len(objects_info),
        "objects": objects_info
    }

# Usage in SLM context
result = decide_object_action(sl_objects, detection_labels)

if result["objects_found"]:
    closest = min(result["objects"], key=lambda x: x["distance"])
    prompt = f"There is a {closest['type']} {closest['distance']:.2f}m away. What should I do?"
else:
    prompt = "No objects detected. What should I do?"
```

## Important Notes

### Object IDs and Tracking
- **By default**, `obj.id` returns `-1` because object tracking is disabled
- To enable cross-frame object tracking:
  1. Enable tracking in camera initialization: `ObjectDetectionParameters.enable_tracking = True`
  2. Then `obj.id` will return a consistent integer ID across frames
  3. The same physical object will have the same `id` in consecutive frames

### Coordinate Systems
- **2D coordinates** (bbox_2d): Pixel positions with (0,0) at top-left
- **3D coordinates** (position): World coordinates relative to camera, typically:
  - X = right
  - Y = down  
  - Z = forward (depth)

### Performance
- Detection runs on GPU (CUDA)
- Processing time depends on:
  - Input image size (416px by default)
  - Number of objects in frame
  - GPU availability

## Updating BiraContext

When storing detection results in `BiraContext`:

```python
from bira_orchestration.context import BiraContext
import pyzed.sl as sl

context = BiraContext()

# Store detected objects (already properly typed)
context.objects_detected = sl_objects.object_list  # type: list[sl.ObjectData]

# Store labels
context.detection_labels = detection_labels  # type: list[int]

# Now use them
for obj in context.objects_detected:  # IDE provides full autocomplete
    print(f"Object at {obj.position} with confidence {obj.confidence}")
```

## Debugging

If you get unexpected values:

```python
# Check actual detection data
print(f"Is new data: {sl_objects.is_new()}")
print(f"Is tracked: {sl_objects.is_tracked()}")
print(f"Number of objects: {len(sl_objects.object_list)}")

# Inspect first object (if any)
if sl_objects.object_list:
    obj = sl_objects.object_list[0]
    print(f"ID: {obj.id} (expect -1 if tracking disabled)")
    print(f"Confidence: {obj.confidence} (0-100)")
    print(f"Position: {obj.position}")
    print(f"Label: {obj.label}")
    print(f"BBox 2D shape: {obj.bounding_box_2d.shape} (expect [4, 2])")
```

## Related Files

- **Implementation**: [`src/bira_components/computer_vision.py`](src/bira_components/computer_vision.py)
- **Type hints**: [`src/bira_orchestration/context.py`](src/bira_orchestration/context.py)
- **Usage example**: [`src/bira_orchestration/states/vision_state.py`](src/bira_orchestration/states/vision_state.py)
- **ZED SDK docs**: https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1ObjectData.html

## Deprecation Note

The file `src/cv_viewer/labels.py` is **no longer needed**. All label resolution now goes through the `ComputerVision` instance using `get_label_name()` or the `label_dict` property. This ensures labels always match the loaded YOLO model, regardless of which model you use or if you train a custom one.
