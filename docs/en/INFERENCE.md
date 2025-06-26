# Hailo Model Inference Guide

This document will teach you how to use converted Hailo models for AI inference. Whether you want to detect objects in images or analyze video content, there are detailed tutorials here.

## What is Model Inference?

Simply put, model inference is letting an AI model "look at" images or videos and then tell you what it "sees". For example:
- 🚗 Finding cars and pedestrians in images
- 🐱 Recognizing cats and dogs in images
- 👤 Detecting human poses and movements
- 🎭 Segmenting different regions in images

## Preparation

### 1. Make sure you have the following files

- ✅ **Converted model file** (`.hef` format)
- ✅ **Test images or videos**
- ✅ **Hailo Toolbox installed**

### 2. Check installation

```bash
# Verify tool is working properly
hailo-toolbox --version

# View help information
hailo-toolbox infer --help
```

## Basic Inference Tutorial

### Step 1: Simplest Inference

Assuming you have a YOLOv8 object detection model, the simplest inference command is:

```bash
hailo-toolbox infer your_model.hef --source test_image.jpg --task-name yolov8det
```

**Explanation**:
- `infer`: Start inference function
- `your_model.hef`: Your model file
- `--source test_image.jpg`: Image to analyze
- `--task-name yolov8det`: Tell the system this is a YOLOv8 detection model

### Step 2: View Results in Real-time

Add the `--show` parameter to see detection results in real-time:

```bash
hailo-toolbox infer your_model.hef \
    --source test_image.jpg \
    --task-name yolov8det \
    --show
```

This will pop up a window showing detection results, press any key to close.

### Step 3: Save Results

If you want to save detection results, use the `--save-dir` parameter:

```bash
hailo-toolbox infer your_model.hef \
    --source test_image.jpg \
    --task-name yolov8det \
    --save-dir ./results
```

Results will be saved in the `results` folder.

## Supported Input Types

### 1. Image Files

```bash
# Single image
hailo-toolbox infer model.hef --source photo.jpg --task-name yolov8det

# Supported formats: JPG, PNG, BMP, TIFF, WebP
hailo-toolbox infer model.hef --source image.png --task-name yolov8det
```

### 2. Image Folders

```bash
# Batch process all images in folder
hailo-toolbox infer model.hef --source ./images/ --task-name yolov8det --save-dir ./results
```

### 3. Video Files

```bash
# Video file inference
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --show

# Supported formats: MP4, AVI, MOV, MKV, WebM
hailo-toolbox infer model.hef --source movie.avi --task-name yolov8det
```

### 4. Real-time Camera Inference

```bash
# Use computer camera (device ID usually 0)
hailo-toolbox infer model.hef --source 0 --task-name yolov8det --show

# If you have multiple cameras, try other IDs
hailo-toolbox infer model.hef --source 1 --task-name yolov8det --show
```

### 5. Network Cameras

```bash
# IP camera (RTSP stream)
hailo-toolbox infer model.hef \
    --source "rtsp://username:password@192.168.1.100:554/stream" \
    --task-name yolov8det \
    --show
```

## Supported Task Types

### Object Detection (Finding Objects)

```bash
# YOLOv8 object detection
hailo-toolbox infer yolov8_detection.hef \
    --source image.jpg \
    --task-name yolov8det \
    --show
```

**What it can detect**: People, cars, animals, daily objects, etc. (80 types of objects)

### Instance Segmentation (Precise Contours)

```bash
# YOLOv8 instance segmentation
hailo-toolbox infer yolov8_segmentation.hef \
    --source image.jpg \
    --task-name yolov8seg \
    --show
```

**What it can do**: Not only find objects but also draw precise contours

### Pose Estimation (Human Keypoints)

```bash
# YOLOv8 pose estimation
hailo-toolbox infer yolov8_pose.hef \
    --source image.jpg \
    --task-name yolov8pe \
    --show
```

**What it can do**: Detect 17 human keypoints, analyze human poses and movements

## Practical Usage Examples

### Example 1: Home Security Monitoring

```bash
# Use camera to detect intruders
hailo-toolbox infer security_model.hef \
    --source 0 \
    --task-name yolov8det \
    --show \
    --save-dir ./security_logs
```

### Example 2: Traffic Monitoring

```bash
# Analyze traffic video, detect vehicles and pedestrians
hailo-toolbox infer traffic_model.hef \
    --source traffic_video.mp4 \
    --task-name yolov8det \
    --save-dir ./traffic_analysis
```

### Example 3: Batch Image Processing

```bash
# Process all product images in folder
hailo-toolbox infer product_detection.hef \
    --source ./product_photos/ \
    --task-name yolov8det \
    --save-dir ./detection_results
```

### Example 4: Pose Estimation Analysis

```bash
# Analyze human poses in video
hailo-toolbox infer pose_model.hef \
    --source workout_video.mp4 \
    --task-name yolov8pe \
    --show \
    --save-dir ./pose_analysis
```

## Inference Parameter Details

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model` | Model file path | `yolov8n.hef` |
| `--source` | Input source | `image.jpg`, `0`, `video.mp4` |

### Important Optional Parameters

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--task-name` | `yolov8det` | Task type | `yolov8det`, `yolov8seg`, `yolov8pe` |
| `--show` | No display | Display results in real-time | `--show` |
| `--save-dir` | No save | Result save directory | `--save-dir ./results` |

### Task Type Description

| Task Name | Function | Applicable Models | Output Results |
|-----------|----------|-------------------|----------------|
| `yolov8det` | Object detection | YOLOv8 detection models | Bounding boxes + classes + confidence |
| `yolov8seg` | Instance segmentation | YOLOv8 segmentation models | Segmentation masks + bounding boxes |
| `yolov8pe` | Pose estimation | YOLOv8 pose models | Human keypoints + skeleton |

## Common Problem Solutions

### Q1: Shows "model file not found"

**Problem**: `FileNotFoundError: model.hef not found`

**Solution**:
```bash
# Check if file exists
ls -la your_model.hef

# Use full path
hailo-toolbox infer /full/path/to/model.hef --source image.jpg --task-name yolov8det
```

### Q2: Camera cannot be opened

**Problem**: `Cannot open camera device 0`

**Solution**:
```bash
# Try different device IDs
hailo-toolbox infer model.hef --source 0 --task-name yolov8det  # First camera
hailo-toolbox infer model.hef --source 1 --task-name yolov8det  # Second camera

# Check available cameras on Linux
ls /dev/video*
```

### Q3: Inference results are inaccurate

**Possible causes and solutions**:

1. **Task type mismatch**
```bash
# Make sure to use correct task-name
# Detection models use yolov8det
# Segmentation models use yolov8seg  
# Pose models use yolov8pe
```

2. **Input image quality issues**
- Ensure images are clear with sufficient lighting
- Check if image size is appropriate
- Avoid overly blurry or dark images

3. **Model conversion issues**
- Re-convert model with better calibration dataset
- Check conversion parameter settings

### Q4: Inference speed is very slow

**Optimization suggestions**:

1. **Reduce input resolution**
```bash
# If original image is large, you can scale it first
# Or use smaller size models
```

2. **Check hardware connection**
- Ensure Hailo device is properly connected
- Check if drivers are working normally

3. **Reduce output saving**
```bash
# Don't save results during testing, only display
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --show
```

### Q5: Display window cannot be closed

**Solution**:
- Click on display window, then press any key
- Or press `Ctrl+C` to force exit program

## Performance Optimization Tips

### 1. Choose Appropriate Input Source

```bash
# High quality images (slower)
--source high_resolution_image.jpg

# Standard video (balanced)
--source standard_video.mp4

# Low resolution stream (faster)
--source low_res_stream.mp4
```

### 2. Reasonable Use of Display and Save

```bash
# Only display, no save (fastest)
--show

# Only save, no display (suitable for batch processing)
--save-dir ./results

# Both display and save (slowest)
--show --save-dir ./results
```

### 3. Batch Processing Optimization

```bash
# For batch processing, don't display in real-time
hailo-toolbox infer model.hef \
    --source ./image_folder/ \
    --task-name yolov8det \
    --save-dir ./batch_results
    # Note: don't add --show parameter
```

## Understanding Inference Results

### Object Detection Results

After inference completes, you will see:
- **Bounding boxes**: Rectangular boxes marking detected objects
- **Class labels**: Display object names (like "person", "car")
- **Confidence**: Display detection confidence (like 0.85 means 85% confident)

### Instance Segmentation Results

In addition to bounding boxes, you will also see:
- **Color masks**: Use different colors to mark precise object contours
- **Overlapping areas**: Can handle objects occluding each other

### Pose Estimation Results

Will display:
- **Keypoints**: 17 important body parts (like head, shoulders, wrists, etc.)
- **Skeleton connections**: Use lines to connect related keypoints
- **Confidence**: Detection confidence for each keypoint

## Advanced Usage

### Using Python API

If you're familiar with Python, you can also use it in code:

```python
from hailo_toolbox.inference import InferenceEngine

# Create inference engine
engine = InferenceEngine(
    model="your_model.hef",
    source="test_image.jpg",
    task_name="yolov8det",
    show=True,
    save_dir="./results"
)

# Run inference
engine.run()
```

### Custom Model Support

If you have custom models, you may need to implement corresponding post-processing functions. For detailed information, please refer to [Developer Documentation](DEV.md).

## Inference Process Diagram

```
[Input Source] → [Preprocessing] → [Model Inference] → [Postprocessing] → [Result Display/Save]
    ↓              ↓                ↓                  ↓                   ↓
Images/Videos   Size adjustment    AI computation    Result parsing    Bounding boxes/Masks
```

## Summary

Basic steps for model inference:

1. **Prepare model file** (`.hef` format)
2. **Prepare input data** (images, videos, or camera)
3. **Choose correct task type** (`yolov8det`, `yolov8seg`, `yolov8pe`)
4. **Run inference command**
5. **View or save results**

**Remember this universal command**:
```bash
hailo-toolbox infer your_model.hef \
    --source your_input \
    --task-name yolov8det \
    --show \
    --save-dir ./results
```

### Common Command Quick Reference

```bash
# Image detection
hailo-toolbox infer model.hef --source image.jpg --task-name yolov8det --show

# Video analysis
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --save-dir ./results

# Real-time camera
hailo-toolbox infer model.hef --source 0 --task-name yolov8det --show

# Batch processing
hailo-toolbox infer model.hef --source ./images/ --task-name yolov8det --save-dir ./results
```

Now you have mastered the basic skills of Hailo model inference! Start enjoying the convenience brought by AI!

---

**Related Documentation**: 
- [Model Conversion Guide](CONVERT.md) - Learn how to convert models
- [Developer Documentation](DEV.md) - Custom model development
- [Quick Start](GET_STAR.md) - Complete installation and usage guide 