# Hailo Model Inference Guide

This document will teach you how to use converted Hailo models for AI inference. Whether you want to detect objects in images or analyze video content, there are detailed tutorials here.

## What is Model Inference?

Simply put, model inference is letting an AI model "look at" images or videos and then tell you what it "sees". For example:
- 🚗 Finding cars and pedestrians in images
- 🐱 Recognizing cats and dogs in images
- 👤 Detecting human poses and movements
- 🎭 Segmenting different regions in images

## Preparation

### 1. Make sure you have the following

- ✅ **Hailo Toolbox installed**
- ✅ **Python environment ready**
- ✅ **Test images or videos** (or use camera)

### 2. Check installation

```python
# Verify installation in Python
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox import create_source
print("Hailo Toolbox is correctly installed!")
```

## Basic Inference Tutorial

### Step 1: Understand the Basic Structure

All inference follows the same pattern:

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

# 1. Create input source
source = create_source("your_input_source")

# 2. Load model
model = ModelsZoo.task_type.model_name()

# 3. Process each frame
for img in source:
    results = model.predict(img)
    for result in results:
        # 4. Process results
        print("Processing results...")
```

### Step 2: Choose Input Source

```python
# Image file
source = create_source("test_image.jpg")

# Video file
source = create_source("video.mp4")

# Camera (device ID usually 0)
source = create_source(0)

# Network camera
source = create_source("rtsp://username:password@192.168.1.100:554/stream")

# Image folder
source = create_source("./images/")

# Network video
source = create_source("https://example.com/video.mp4")
```

## Supported Task Types and Examples

### 1. Object Detection (Finding Objects)

**Example File**: `examples/Hailo_Object_Detection.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import DetectionVisualization
import cv2

if __name__ == "__main__":
    # Create input source
    source = create_source("test_video.mp4")  # or use 0 for camera
    
    # Load YOLOv8 detection model
    inference = ModelsZoo.detection.yolov8s()
    visualization = DetectionVisualization()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # Visualize results
            img = visualization.visualize(img, result)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)
            
            # Get detection results
            boxes = result.get_boxes()      # Bounding boxes
            scores = result.get_scores()    # Confidence scores
            class_ids = result.get_class_ids()  # Class IDs
            
            print(f"Detected {len(result)} objects")
            # Show first 5 detection results
            for i in range(min(5, len(result))):
                print(f"  Object{i}: bbox{boxes[i]}, score{scores[i]:.3f}, class{class_ids[i]}")
```

**What it can detect**: People, cars, animals, daily objects, etc. (80 types of objects)

**Available Models**:
- `ModelsZoo.detection.yolov8n()` - Fastest speed
- `ModelsZoo.detection.yolov8s()` - Balanced speed and accuracy
- `ModelsZoo.detection.yolov8m()` - Higher accuracy
- `ModelsZoo.detection.yolov8l()` - High accuracy
- `ModelsZoo.detection.yolov8x()` - Highest accuracy

### 2. Instance Segmentation (Precise Contours)

**Example File**: `examples/Hailo_Instance_Segmentation.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import SegmentationVisualization
import cv2

if __name__ == "__main__":
    source = create_source("test_video.mp4")
    
    # Load YOLOv8 segmentation model
    inference = ModelsZoo.segmentation.yolov8s_seg()
    visualization = SegmentationVisualization()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # Visualize segmentation results
            img = visualization.visualize(img, result)
            cv2.imshow("Segmentation", img)
            cv2.waitKey(1)
            
            # Get segmentation results
            if hasattr(result, "masks") and result.masks is not None:
                print(f"Segmentation mask shape: {result.masks.shape}")
            
            boxes = result.get_boxes_xyxy()  # Bounding boxes
            scores = result.get_scores()     # Confidence scores
            class_ids = result.get_class_ids()  # Class IDs
```

**What it can do**: Not only find objects but also draw precise contours

**Available Models**:
- `ModelsZoo.segmentation.yolov8n_seg()` - Fast segmentation
- `ModelsZoo.segmentation.yolov8s_seg()` - Standard segmentation
- `ModelsZoo.segmentation.yolov8m_seg()` - High accuracy segmentation

### 3. Pose Estimation (Human Keypoints)

**Example File**: `examples/Hailo_Pose_Estimation.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import KeypointVisualization
import cv2

if __name__ == "__main__":
    source = create_source("test_video.mp4")
    
    # Load YOLOv8 pose estimation model
    inference = ModelsZoo.pose_estimation.yolov8s_pose()
    visualization = KeypointVisualization()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # Visualize pose results
            img = visualization.visualize(img, result)
            cv2.imshow("Pose Estimation", img)
            cv2.waitKey(1)
            
            print(f"Detected {len(result)} persons")
            # Show pose information for first 3 persons
            for i, person in enumerate(result):
                if i >= 3:  # Only show first 3
                    break
                keypoints = person.get_keypoints()  # Keypoint coordinates
                score = person.get_score()          # Person confidence
                boxes = person.get_boxes()          # Bounding boxes
                
                print(f"  Person{i}: {len(keypoints)} keypoints, confidence{score[0]:.3f}")
```

**What it can do**: Detect 17 human keypoints, analyze human poses and movements

**Available Models**:
- `ModelsZoo.pose_estimation.yolov8s_pose()` - Standard pose estimation
- `ModelsZoo.pose_estimation.yolov8m_pose()` - High accuracy pose estimation

### 4. Image Classification (Identify Main Objects)

**Example File**: `examples/Hailo_Classification.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

if __name__ == "__main__":
    source = create_source("test_image.jpg")
    
    # Load classification model
    inference = ModelsZoo.classification.resnet18()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # Get classification results
            class_name = result.get_class_name()        # Most likely class
            confidence = result.get_score()             # Confidence score
            top5_names = result.get_top_5_class_names() # Top 5 classes
            top5_scores = result.get_top_5_scores()     # Top 5 scores
            
            print(f"Classification result: {class_name} (confidence: {confidence:.3f})")
            print(f"Top5 classes: {top5_names}")
            print(f"Top5 scores: {[f'{score:.3f}' for score in top5_scores]}")
```

**Available Models**:
- `ModelsZoo.classification.mobilenetv1()` - Lightweight classification
- `ModelsZoo.classification.resnet18()` - Classic classification model

### 5. Face Detection

**Example File**: `examples/Hailo_Face_Detection.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

def visualize_face_detection(img, boxes, scores, landmarks):
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        # Draw face box
        cv2.rectangle(img, (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    source = create_source("test_video.mp4")
    
    # Load face detection model
    inference = ModelsZoo.face_detection.scrfd_10g()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            print(f"Detected {len(result)} faces")
            
            boxes = result.get_boxes(pixel_coords=True)      # Face boxes
            scores = result.get_scores()                     # Confidence scores
            landmarks = result.get_landmarks(pixel_coords=True)  # Facial landmarks
            
            img = visualize_face_detection(img, boxes, scores, landmarks)
            cv2.imshow("Face Detection", img)
            cv2.waitKey(1)
```

**Available Models**:
- `ModelsZoo.face_detection.scrfd_10g()` - High accuracy face detection
- `ModelsZoo.face_detection.scrfd_2_5g()` - Balanced performance
- `ModelsZoo.face_detection.scrfd_500m()` - Fast detection
- `ModelsZoo.face_detection.retinaface_mbnet()` - Lightweight detection

### 6. Depth Estimation

**Example File**: `examples/Hailo_Depth_Estimation.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

if __name__ == "__main__":
    source = create_source("test_video.mp4")
    
    # Load depth estimation model
    inference = ModelsZoo.depth_estimation.fast_depth()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            depth_map = result.get_depth()                    # Raw depth map
            depth_normalized = result.get_depth_normalized()  # Normalized depth map
            original_shape = result.get_original_shape()      # Original image size
            
            cv2.imshow("Depth Estimation", depth_normalized)
            cv2.waitKey(1)
            
            print(f"Depth map shape: {depth_map.shape}")
            print(f"Depth value range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
            print(f"Original image size: {original_shape}")
```

**Available Models**:
- `ModelsZoo.depth_estimation.fast_depth()` - Fast depth estimation
- `ModelsZoo.depth_estimation.scdepthv3()` - High accuracy depth estimation

## Practical Usage Examples

### Example 1: Home Security Monitoring

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import DetectionVisualization
import cv2
import datetime

def security_monitoring():
    # Use camera
    source = create_source(0)
    inference = ModelsZoo.detection.yolov8s()
    visualization = DetectionVisualization()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # Check for people
            class_ids = result.get_class_ids()
            if 0 in class_ids:  # 0 means person
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"security_alert_{timestamp}.jpg", img)
                print(f"Security Alert! Intruder detected - {timestamp}")
            
            img = visualization.visualize(img, result)
            cv2.imshow("Security Monitor", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    security_monitoring()
```

### Example 2: Traffic Monitoring Analysis

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

def traffic_analysis():
    source = create_source("traffic_video.mp4")
    inference = ModelsZoo.detection.yolov8m()
    
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            class_ids = result.get_class_ids()
            boxes = result.get_boxes()
            
            vehicle_count = sum(1 for class_id in class_ids if class_id in vehicle_classes)
            person_count = sum(1 for class_id in class_ids if class_id == 0)
            
            print(f"Vehicle count: {vehicle_count}, Person count: {person_count}")
            
            # Visualize
            for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
                if class_id in vehicle_classes or class_id == 0:
                    color = (0, 255, 0) if class_id in vehicle_classes else (255, 0, 0)
                    cv2.rectangle(img, (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), color, 2)
            
            cv2.imshow("Traffic Analysis", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    traffic_analysis()
```

### Example 3: Batch Image Processing

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2
import os

def batch_image_processing():
    # Process all images in folder
    source = create_source("./product_photos/")
    inference = ModelsZoo.detection.yolov8n()
    
    os.makedirs("./detection_results", exist_ok=True)
    
    for i, img in enumerate(source):
        results = inference.predict(img)
        for result in results:
            boxes = result.get_boxes()
            scores = result.get_scores()
            class_ids = result.get_class_ids()
            
            # Draw detection results on image
            for box, score, class_id in zip(boxes, scores, class_ids):
                cv2.rectangle(img, (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(img, f"Class{class_id}: {score:.2f}", 
                          (int(box[0]), int(box[1])-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save results
            cv2.imwrite(f"./detection_results/result_{i:04d}.jpg", img)
            print(f"Processed: Image {i}, detected {len(result)} objects")

if __name__ == "__main__":
    batch_image_processing()
```

## Common Problem Solutions

### Q1: Shows "model file not found"

**Problem**: Model download failed or network issues

**Solution**:
```python
# Check network connection
import requests
try:
    response = requests.get("https://www.google.com", timeout=5)
    print("Network connection is normal")
except:
    print("Network connection problem, please check network settings")

# Manually download model (if automatic download fails)
from hailo_toolbox.models import ModelsZoo
model = ModelsZoo.detection.yolov8n()  # This will try to download the model
```

### Q2: Camera cannot be opened

**Problem**: `Cannot open camera device 0`

**Solution**:
```python
import cv2

# Test different camera IDs
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
    else:
        print(f"Camera {i} is not available")

# Use available camera ID
source = create_source(0)  # or use the found available ID
```

### Q3: Inference results are inaccurate

**Possible causes and solutions**:

1. **Input image quality issues**
```python
import cv2

# Check image quality
def check_image_quality(img):
    if img is None:
        print("Image is empty")
        return False
    
    height, width = img.shape[:2]
    if height < 100 or width < 100:
        print(f"Image too small: {width}x{height}")
        return False
    
    # Check brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < 30:
        print(f"Image too dark: brightness {brightness}")
    elif brightness > 200:
        print(f"Image too bright: brightness {brightness}")
    
    return True
```

2. **Choose appropriate model**
```python
# Choose model based on requirements
# Speed priority: yolov8n
# Balanced: yolov8s
# Accuracy priority: yolov8m, yolov8l, yolov8x

inference = ModelsZoo.detection.yolov8s()  # Recommended balanced choice
```

### Q4: Inference speed is very slow

**Optimization suggestions**:

1. **Use smaller models**
```python
# Use the fastest model
inference = ModelsZoo.detection.yolov8n()  # instead of yolov8x
```

2. **Reduce input resolution**
```python
import cv2

def resize_frame(img, max_size=640):
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
    return img

# Resize image before inference
for img in source:
    img = resize_frame(img)
    results = inference.predict(img)
    # ...
```

3. **Frame skipping**
```python
frame_skip = 2  # Process every 2nd frame
frame_count = 0

for img in source:
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    results = inference.predict(img)
    # Process results...
```

## Performance Optimization Tips

### 1. Choose Appropriate Models

```python
# Choose models based on application scenarios
# Real-time applications: choose smaller models
inference = ModelsZoo.detection.yolov8n()

# Offline analysis: can choose larger models
inference = ModelsZoo.detection.yolov8x()
```

### 2. Batch Processing Optimization

```python
# For image folders, no need for real-time display
source = create_source("./images/")
inference = ModelsZoo.detection.yolov8s()

for i, img in enumerate(source):
    results = inference.predict(img)
    # Only process results, no display
    for result in results:
        # Save or record results
        print(f"Image {i}: detected {len(result)} objects")
```

### 3. Memory Management

```python
import gc

# Periodic memory cleanup
frame_count = 0
for img in source:
    results = inference.predict(img)
    # Process results...
    
    frame_count += 1
    if frame_count % 100 == 0:
        gc.collect()  # Clean memory every 100 frames
```

## Understanding Inference Results

### Object Detection Results

```python
for result in results:
    boxes = result.get_boxes()          # Bounding boxes [x1, y1, x2, y2]
    scores = result.get_scores()        # Confidence scores [0.0-1.0]
    class_ids = result.get_class_ids()  # Class IDs [0-79 for COCO]
    
    print(f"Detected {len(result)} objects")
    for i in range(len(result)):
        print(f"Object {i}: class{class_ids[i]}, confidence{scores[i]:.3f}")
```

### Segmentation Results

```python
for result in results:
    if hasattr(result, "masks") and result.masks is not None:
        masks = result.masks            # Segmentation masks
        print(f"Mask shape: {masks.shape}")
    
    boxes = result.get_boxes_xyxy()     # Bounding boxes
    scores = result.get_scores()        # Confidence scores
```

### Pose Estimation Results

```python
for result in results:
    for person in result:
        keypoints = person.get_keypoints()  # 17 keypoint coordinates
        score = person.get_score()          # Person detection confidence
        boxes = person.get_boxes()          # Person bounding boxes
        
        print(f"Keypoint count: {len(keypoints)}")
        print(f"Person confidence: {score}")
```

## Summary

Basic steps for model inference using Hailo Toolbox:

1. **Create input source** - Use `create_source()` function
2. **Load model** - Choose appropriate model from `ModelsZoo`
3. **Process data** - Iterate through each frame of input source
4. **Get results** - Call model's `predict()` method
5. **Process output** - Use various methods of result objects to get data

### Common Code Templates

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

# Basic template
def basic_inference():
    source = create_source("your_input")
    model = ModelsZoo.task_type.model_name()
    
    for img in source:
        results = model.predict(img)
        for result in results:
            # Process results
            print("Inference completed")

# Template with visualization
def inference_with_visualization():
    source = create_source("your_input")
    model = ModelsZoo.detection.yolov8s()
    
    for img in source:
        results = model.predict(img)
        for result in results:
            # Draw results
            boxes = result.get_boxes()
            for box in boxes:
                cv2.rectangle(img, (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), (0, 255, 0), 2)
            
            cv2.imshow("Results", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    basic_inference()
```

Now you have mastered the complete skills of Hailo model inference! Refer to the specific examples in the `examples/` folder to start your AI journey!

---

**Related Documentation**: 
- [Model Conversion Guide](CONVERT.md) - Learn how to convert models
- [Developer Documentation](DEV.md) - Custom model development
- [Quick Start](GET_STAR.md) - Complete installation and usage guide
- [Example Code](../examples/) - Complete inference examples 