# Hailo Toolbox Examples

This directory contains example scripts for all supported model types in Hailo Toolbox. Each example demonstrates how to load a model, perform inference on video frames, and process the results.

## Video Source

All examples use the same video source from Hailo's public dataset:
```
https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4
```

## Available Examples

### 1. Image Classification
- **File**: `Hailo_Classification.py`
- **Model**: MobileNetV1
- **Output**: Class name, confidence, Top5 predictions

### 2. Object Detection  
- **File**: `Hailo_Object_Detection.py`
- **Model**: YOLOv8n
- **Output**: Bounding boxes, confidence scores, class IDs

### 3. Instance Segmentation
- **File**: `Hailo_Instance_Segmentation.py` 
- **Model**: YOLOv8m Segmentation
- **Output**: Segmentation masks, bounding boxes, confidence scores

### 4. Pose Estimation
- **File**: `Hailo_Pose_Estimation.py`
- **Model**: YOLOv8s Pose
- **Output**: Human keypoints, person confidence, bounding boxes

### 5. Depth Estimation
- **File**: `Hailo_Depth_Estimation.py`
- **Model**: FastDepth
- **Output**: Depth maps, value ranges, original image size

### 6. Hand Landmark Detection
- **File**: `Hailo_Hand_Landmark.py`
- **Model**: Hand Landmark
- **Output**: Hand keypoint coordinates, coordinate ranges

### 7. Super Resolution
- **File**: `Hailo_Super_Resolution.py`
- **Model**: Real-ESRGAN
- **Output**: Enhanced high-resolution images, upscale factor

### 8. Face Detection
- **File**: `Hailo_Face_Detection.py`
- **Model**: SCRFD-10G
- **Output**: Face bounding boxes, confidence, facial landmarks

### 9. Face Recognition
- **File**: `Hailo_Face_Recognition.py`
- **Model**: ArcFace MobileNet
- **Output**: Face feature vectors for identification

### 10. License Plate Recognition
- **File**: `Hailo_License_Plate_Recognition.py`
- **Model**: LPRNet
- **Output**: Recognized text, confidence (implementation dependent)

### 11. Facial Landmark Detection
- **File**: `Hailo_Facial_Landmark.py`
- **Model**: TDDFA
- **Output**: 68 facial landmarks, coordinate ranges, distribution statistics

### 12. Person Re-identification
- **File**: `Hailo_Person_ReID.py`
- **Model**: OSNet-X1
- **Output**: Person feature vectors for tracking/identification

### 13. Image Denoising
- **File**: `Hailo_Image_Denoising.py`
- **Model**: DnCNN3
- **Output**: Denoised images, brightness comparison

### 14. Low Light Enhancement
- **File**: `Hailo_Low_Light_Enhancement.py`
- **Model**: Zero-DCE
- **Output**: Enhanced images, brightness improvement factor

### 15. Text-Image Retrieval
- **File**: `Hailo_Text_Image_Retrieval.py`
- **Model**: CLIP ViT-L
- **Output**: Image feature vectors for text-image similarity

### 16. Video Classification
- **File**: `Hailo_Video_Classification.py`
- **Model**: R3D-18
- **Output**: Action classes, confidence scores (requires multi-frame input)

## How to Run

Each example can be run independently:

```bash
# Run a specific example
python examples/Hailo_Object_Detection.py

# Run classification example
python examples/Hailo_Classification.py

# Run pose estimation example
python examples/Hailo_Pose_Estimation.py
```

## Example Structure

All examples follow the same basic structure:

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

if __name__ == "__main__":
    # Create video source
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load model
    inference = ModelsZoo.task_type.model_name()

    # Process video frames
    for img in source:
        results = inference.predict(img)
        for result in results:
            # Process and display results
            print("Results...")
            print("---")
```

## Key Features

- **Real-time Processing**: All examples process video frames in real-time
- **Result Methods**: Demonstrate how to access different result attributes
- **Error Handling**: Robust handling of different result formats
- **Performance Metrics**: Show timing and accuracy information where available
- **Standardized Output**: Consistent formatting for easy comparison

## Notes

1. **Network Connection**: Examples require internet access to download the video and models
2. **First Run**: Initial model loading may take time for downloading
3. **Hardware**: Some models may require specific Hailo hardware for optimal performance
4. **Dependencies**: Ensure all required packages are installed

## Related Files

- **`model_examples.py`**: Comprehensive examples with test data
- **`docs/MODEL_EXAMPLES.md`**: Detailed documentation
- **`docs/QUICK_REFERENCE.md`**: Quick reference guide

## Support

For more information, see the main documentation or visit the Hailo website.
