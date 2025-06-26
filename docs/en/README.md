# Hailo Toolbox Inference Tutorial

This is a comprehensive tutorial that will guide you through using the Hailo Toolbox framework for inference. This toolbox supports Hailo (.hef) and ONNX models with various input sources and customizable processing pipelines.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Basic Usage](#basic-usage)
- [Command Line Arguments](#command-line-arguments)
- [Inference Commands](#inference-commands)
- [Input Source Types](#input-source-types)
- [Callback Functions](#callback-functions)
- [Python API Usage](#python-api-usage)
- [Custom Callback Functions](#custom-callback-functions)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)
- [Summary](#summary)

## Installation and Setup

### Installation

Make sure you have installed Hailo Toolbox:

```bash
pip install -e .
```

### Verify Installation

Check version information:

```bash
hailo-toolbox --version
```

## Basic Usage

### Command Structure

Hailo Toolbox CLI uses a subcommand structure:

```bash
hailo-toolbox <subcommand> [arguments]
```

Supported subcommands:
- `infer`: Model inference
- `convert`: Model conversion

### Simple Inference Example

Run inference on a video file:

```bash
hailo-toolbox infer models/yolov8n.hef --source sources/test.mp4 --task-name yolov8det
```

## Command Line Arguments

### Global Arguments

#### Version Information
- `--version` / `-v` / `-V`: Display version information and exit
  ```bash
  hailo-toolbox --version
  ```

## Inference Commands

### Basic Syntax

```bash
hailo-toolbox infer <model_path> --source <input_source> [OPTIONS]
```

### Required Arguments

#### model (positional argument)
- **Purpose**: Specify model file path
- **Type**: String
- **Supported formats**: .hef and .onnx formats
- **Examples**:
  ```bash
  hailo-toolbox infer models/yolov8n.hef --source video.mp4 --task-name yolov8det
  hailo-toolbox infer models/yolov8n.onnx --source video.mp4 --task-name yolov8det
  ```

#### --source / -s (required)
- **Purpose**: Specify input source (video file, image file, folder, or camera)
- **Type**: String
- **Required**: Yes
- **Supported formats**:
  - Video files: `.mp4`, `.avi`, `.mov`, `.mkv`, etc.
  - Image files: `.jpg`, `.png`, `.bmp`, `.tiff`, etc.
  - Image folders: Directory containing image files
  - Camera: `0`, `1` (device ID)
  - IP camera: `rtsp://...`
- **Examples**:
  ```bash
  # Video file
  hailo-toolbox infer models/yolov8n.hef --source video.mp4 --task-name yolov8det
  
  # Image file
  hailo-toolbox infer models/yolov8n.hef --source image.jpg --task-name yolov8det
  
  # Image folder
  hailo-toolbox infer models/yolov8n.hef --source images/ --task-name yolov8det
  
  # Webcam
  hailo-toolbox infer models/yolov8n.hef --source 0 --task-name yolov8det
  ```

### Optional Arguments

#### --task-name / -tn
- **Purpose**: Specify task name for callback function registration lookup
- **Type**: String
- **Default**: `yolov8det`
- **Common values**: `yolov8det`, `yolov8seg`, `yolov8pe` (pose estimation)
- **Examples**:
  ```bash
  hailo-toolbox infer models/yolov8n.hef --source video.mp4 --task-name yolov8det
  hailo-toolbox infer models/yolov8n_seg.hef --source video.mp4 --task-name yolov8seg
  ```

#### --save-dir / -sd
- **Purpose**: Specify directory to save output results
- **Type**: String
- **Default**: None
- **Example**:
  ```bash
  hailo-toolbox infer models/yolov8n.hef --source video.mp4 --task-name yolov8det --save-dir output/
  ```

#### --show / -sh
- **Purpose**: Display output results in real-time (flag parameter)
- **Type**: Boolean flag
- **Default**: False
- **Example**:
  ```bash
  hailo-toolbox infer models/yolov8n.hef --source video.mp4 --task-name yolov8det --show
  ```

## Input Source Types

### Supported Input Sources

1. **Video Files**
   - Formats: MP4, AVI, MOV, MKV, WMV, etc.
   - Example: `--source video.mp4`

2. **Image Files**
   - Formats: JPG, PNG, BMP, TIFF, WEBP, etc.
   - Example: `--source image.jpg`

3. **Image Folders**
   - Format: Directory path containing image files
   - Example: `--source images/`
   - Processes all supported image files in the directory

4. **USB Cameras**
   - Format: Device ID (integer)
   - Example: `--source 0` (default camera)

5. **IP Cameras**
   - Format: RTSP stream address
   - Example: `--source rtsp://username:password@ip:port/stream`

## Callback Functions

### Built-in Callback Functions

- `yolov8det`: YOLOv8 object detection
- `yolov8seg`: YOLOv8 instance segmentation
- `yolov8pe`: YOLOv8 pose estimation

### Callback Function Responsibilities

Callback functions handle:
- **Preprocessing**: Input data preparation
- **Postprocessing**: Model output processing
- **Visualization**: Result rendering and display

## Python API Usage

### New Style API (Recommended)

```python
from hailo_toolbox.inference import InferenceEngine

# Direct parameter approach - more concise and intuitive
engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    show=True,
    save_dir="output/"
)
engine.run()
```

### Legacy API (Backward Compatible)

```python
from hailo_toolbox.inference import InferenceEngine
from hailo_toolbox.utils.config import Config

# Configuration object approach - maintains backward compatibility
config = Config()
config.model = "models/yolov8n.hef"
config.source = "video.mp4"
config.task_name = "yolov8det"
config.show = True
config.save_dir = "output/"

engine = InferenceEngine(config, "yolov8det")
engine.run()
```

### API Parameter Description

#### InferenceEngine Constructor Parameters

```python
InferenceEngine(
    config=None,                    # Configuration object (backward compatible)
    task_name="yolov8det",         # Task name
    model=None,                    # Model file path
    command=None,                  # Command type
    convert=False,                 # Whether to convert model
    source=None,                   # Data source
    output=None,                   # Output configuration
    preprocess_config=None,        # Preprocessing configuration
    postprocess_config=None,       # Postprocessing configuration
    visualization_config=None,     # Visualization configuration
    task_type=None,               # Task type
    save=False,                   # Whether to save
    save_dir=None,                # Save directory
    show=False,                   # Whether to display
    **kwargs
)
```

## Custom Callback Functions

### Registration Mechanism

```python
from hailo_toolbox.inference.core import CALLBACK_REGISTRY

# Single name registration
@CALLBACK_REGISTRY.registryPreProcessor("custom")
def custom_preprocess(image):
    return processed_image

# Multiple name registration (same function can be called with multiple names)
@CALLBACK_REGISTRY.registryPreProcessor("yolov8det", "yolov8seg")
def yolo_preprocess_func(image):
    return processed_image

# Postprocessor registration
@CALLBACK_REGISTRY.registryPostProcessor("custom")
class CustomPostProcessor:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, results, original_shape=None):
        # Process inference results
        return processed_results

# Visualizer registration
@CALLBACK_REGISTRY.registryVisualizer("custom")
class CustomVisualizer:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, original_frame, results):
        # Draw results on image
        return annotated_frame
```

## Practical Examples

### 1. Object Detection

```bash
# Command line approach
hailo-toolbox infer models/yolov8n.hef --source video.mp4 --task-name yolov8det --show --save-dir results/

# Python API approach
from hailo_toolbox.inference import InferenceEngine

engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    show=True,
    save_dir="results/"
)
engine.run()
```

### 2. Instance Segmentation

```bash
# Command line approach
hailo-toolbox infer models/yolov8n_seg.hef --source images/ --task-name yolov8seg --save-dir results/

# Python API approach
engine = InferenceEngine(
    model="models/yolov8n_seg.hef",
    source="images/",
    task_name="yolov8seg",
    save_dir="results/"
)
engine.run()
```

### 3. Pose Estimation

```bash
# Command line approach
hailo-toolbox infer models/yolov8s_pose.hef --source 0 --task-name yolov8pe --show

# Python API approach
engine = InferenceEngine(
    model="models/yolov8s_pose.hef",
    source=0,  # Camera
    task_name="yolov8pe",
    show=True
)
engine.run()
```

### 4. Custom Model

```python
from hailo_toolbox.inference import InferenceEngine
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig

# Custom preprocessing configuration
preprocess_config = PreprocessConfig(
    target_size=(224, 224),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

engine = InferenceEngine(
    model="models/custom_model.hef",
    source="test_image.jpg",
    task_name="custom",  # Requires corresponding callback functions to be registered
    preprocess_config=preprocess_config,
    show=True
)
engine.run()
```

## Best Practices

### 1. Performance Optimization
- **Batch Processing**: Use folder input for large numbers of images to leverage batch processing optimization
- **GPU Acceleration**: ONNX models automatically detect and use CUDA (if available)
- **Memory Management**: Regularly clean up resources for long-running applications

### 2. Error Handling
- **Model Validation**: Ensure model files exist and are in correct format
- **Input Validation**: Check input source validity
- **Dependency Check**: Ensure required callback functions are registered

### 3. Debugging Tips
- **Verbose Logging**: Use `--verbose` flag for detailed information
- **Step-by-step Debugging**: Test with single images first, then process videos
- **Configuration Validation**: Verify all configuration parameters are correct

## Summary

Hailo Toolbox provides a powerful and flexible inference framework that supports:

- **Multiple Model Formats**: HEF and ONNX
- **Various Input Sources**: Video, images, cameras, network streams
- **Flexible APIs**: New direct parameter API and traditional configuration object API
- **Extensible Architecture**: Support for custom callback functions and processing workflows
- **High Performance**: Optimized inference pipeline and memory management

Whether for rapid prototyping or production deployment, Hailo Toolbox can meet your deep learning inference needs. With this tutorial, you should be able to effectively use this toolbox for various inference tasks. 