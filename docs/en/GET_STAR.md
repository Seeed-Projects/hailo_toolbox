# Hailo Toolbox Quick Start Guide

This document will introduce how to install and use the Hailo Toolbox, a comprehensive toolkit designed for deep learning model conversion and inference. This guide contains complete instructions from basic installation to advanced usage.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Verify Installation](#verify-installation)
- [Project Structure](#project-structure)
- [Model Conversion](#model-conversion)
- [Model Inference](#model-inference)
- [Python API Usage](#python-api-usage)
- [Usage Examples](#usage-examples)
- [Common Issues](#common-issues)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Basic Requirements
- **Python Version**: 3.8 ≤ Python < 3.12
- **Operating System**: Linux (Ubuntu 18.04+ recommended), Windows 10+, macOS 10.15+
- **Memory**: At least 8GB RAM (16GB+ recommended)
- **Storage**: At least 2GB available space

### Hailo-Specific Requirements
- **[Hailo Dataflow Compiler](https://hailo.ai/developer-zone/software-downloads/)**: For model conversion functionality (required, X86 architecture and Linux only)
- **[HailoRT](https://hailo.ai/developer-zone/software-downloads/)**: For inference functionality (required for inference)
- **Hailo Hardware**: For hardware-accelerated inference (required)

### Python Dependencies
Core dependency packages will be automatically installed:
```
opencv-python>=4.5.0
numpy<2.0.0
requests>=2.25.0
matplotlib>=3.3.0
onnx
onnxruntime
pillow
pyyaml
tqdm
```

## Installation

### Method 1: Install from Source (Recommended)

```bash
# Clone project source code
git clone https://github.com/Seeed-Projects/hailo_toolbox.git

# Enter project directory
cd hailo_toolbox

# Install project (development mode)
pip install -e .

# Or install directly
pip install .
```

### Method 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv hailo_env

# Activate virtual environment
# Linux/macOS:
source hailo_env/bin/activate
# Windows:
hailo_env\Scripts\activate

# Install project
git clone https://github.com/Seeed-Projects/hailo_toolbox.git
cd hailo_toolbox
pip install -e .
```

## Verify Installation

After installation, verify successful installation with the following commands:

```bash
# Check version information
hailo-toolbox --version

# View help information
hailo-toolbox --help

# View conversion functionality help
hailo-toolbox convert --help

# View inference functionality help
hailo-toolbox infer --help
```

## Project Structure

```
hailo_toolbox/
├── cli/                    # Command line interface
│   ├── config.py          # Parameter configuration
│   ├── convert.py         # Model conversion CLI
│   ├── infer.py           # Model inference CLI
│   └── server.py          # API server
├── converters/            # Model converters
├── inference/             # Inference engine
│   ├── core.py           # Core inference engine and registration mechanism
│   ├── hailo_engine.py   # Hailo inference engine
│   ├── onnx_engine.py    # ONNX inference engine
│   └── pipeline.py       # Inference pipeline
├── process/              # Data processing modules
│   ├── preprocessor/     # Preprocessing modules
│   ├── postprocessor/    # Postprocessing modules
│   └── callback.py       # Callback functions
├── sources/              # Data source management
├── utils/                # Utility functions
└── models/              # Model management
```

## Model Conversion

Hailo Toolbox supports converting models from various deep learning frameworks to efficient `.hef` format for running on Hailo hardware.

### Supported Model Formats

| Framework | Format | Supported | Target Format | Notes |
|-----------|--------|-----------|---------------|-------|
| ONNX | .onnx | ✅ | .hef | Recommended format |
| TensorFlow | .h5 | ✅ | .hef | Keras models |
| TensorFlow | SavedModel.pb | ✅ | .hef | TensorFlow SavedModel |
| TensorFlow Lite | .tflite | ✅ | .hef | Mobile models |
| PyTorch | .pt (torchscript) | ✅ | .hef | TorchScript models |
| PaddlePaddle | inference model | ✅ | .hef | PaddlePaddle inference models |

### Basic Conversion Commands

```bash
# View conversion help
hailo-toolbox convert --help

# Basic conversion (ONNX to HEF)
hailo-toolbox convert model.onnx --hw-arch hailo8

# Complete conversion example
hailo-toolbox convert model.onnx \
    --hw-arch hailo8 \
    --input-shape 320,320,3 \
    --save-onnx \
    --output-dir outputs \
    --profile \
    --calib-set-path ./calibration_images
```

### Conversion Parameter Details

| Parameter | Required | Default | Description | Example |
|-----------|----------|---------|-------------|---------|
| `model` | ✅ | - | Model file path to convert | `model.onnx` |
| `--hw-arch` | ❌ | `hailo8` | Target Hailo hardware architecture | `hailo8`, `hailo8l`, `hailo15`, `hailo15l` |
| `--calib-set-path` | ❌ | None | Calibration dataset folder path | `./calibration_data/` |
| `--use-random-calib-set` | ❌ | False | Use random data for calibration | - |
| `--calib-set-size` | ❌ | None | Calibration dataset size | `100` |
| `--model-script` | ❌ | None | Custom model script path | `./custom_script.py` |
| `--end-nodes` | ❌ | None | Specify model output nodes | `output1,output2` |
| `--input-shape` | ❌ | `[640,640,3]` | Model input shape | `320,320,3` |
| `--save-onnx` | ❌ | False | Save compiled ONNX file | - |
| `--output-dir` | ❌ | Same as model | Output file save directory | `./outputs/` |
| `--profile` | ❌ | False | Generate performance analysis report | - |

## Model Inference

Hailo Toolbox provides flexible inference interfaces supporting various input sources and output formats.

### Basic Inference Commands

```bash
# View inference help
hailo-toolbox infer --help

# Basic inference example
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --show

# Complete inference example
hailo-toolbox infer yolov8.hef \
    --source 0 \
    --task-name yolov8det \
    --save-dir ./results \
    --show
```

### Inference Parameter Details

| Parameter | Required | Default | Description | Example |
|-----------|----------|---------|-------------|---------|
| `model` | ✅ | - | Model file path (.hef or .onnx) | `model.hef` |
| `--source` | ✅ | - | Input source path | See table below |
| `--task-name` | ❌ | `yolov8det` | Task name for callback function lookup | `yolov8det`, `yolov8seg`, `yolov8pe` |
| `--save-dir` | ❌ | None | Result save directory | `./results/` |
| `--show` | ❌ | False | Display results in real-time | - |

### Supported Input Source Types

| Input Source Type | Format | Example | Description |
|-------------------|--------|---------|-------------|
| Image files | jpg, png, bmp, etc. | `image.jpg` | Single image inference |
| Image folders | Directory path | `./images/` | Batch image inference |
| Video files | mp4, avi, mov, etc. | `video.mp4` | Video file inference |
| USB cameras | Device ID | `0`, `1` | Real-time camera inference |
| IP cameras | RTSP/HTTP stream | `rtsp://ip:port/stream` | Network camera inference |
| Network video streams | URL | `http://example.com/stream` | Online video stream inference |

### Available Inference Callback Functions

| Callback Function | Functionality | Applicable Models | Output |
|-------------------|---------------|-------------------|--------|
| `yolov8det` | Object detection | YOLOv8 detection models | Bounding boxes + classes + confidence |
| `yolov8seg` | Instance segmentation | YOLOv8 segmentation models | Segmentation masks + bounding boxes |
| `yolov8pe` | Pose estimation | YOLOv8 pose models | Keypoints + skeleton connections |

## Python API Usage

### Basic Usage

```python
from hailo_toolbox.inference import InferenceEngine

# New style API - direct parameters (recommended)
engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    show=True,
    save_dir="output/"
)
engine.run()

# Legacy API - configuration object (backward compatible)
from hailo_toolbox.utils.config import Config

config = Config()
config.model = "models/yolov8n.hef"
config.source = "video.mp4"
config.task_name = "yolov8det"
config.show = True

engine = InferenceEngine(config, "yolov8det")
engine.run()
```

### Custom Callback Functions

```python
from hailo_toolbox.inference.core import CALLBACK_REGISTRY, InferenceEngine
import numpy as np
import cv2

@CALLBACK_REGISTRY.registryPostProcessor("custom")
class CustomPostProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, results, original_shape=None):
        # Custom postprocessing logic
        processed_results = []
        for k, v in results.items():
            # Process model output
            processed_results.append(self.process_output(v))
        return processed_results

@CALLBACK_REGISTRY.registryVisualizer("custom")
class CustomVisualizer:
    def __init__(self, config):
        self.config = config

    def __call__(self, original_frame, results):
        # Custom visualization logic
        vis_frame = original_frame.copy()
        for result in results:
            # Draw results
            cv2.putText(vis_frame, str(result), (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return vis_frame

# Use custom callbacks
engine = InferenceEngine(
    model="models/custom_model.hef",
    source="video.mp4",
    task_name="custom",
    show=True
)
engine.run()
```

## Usage Examples

### Example 1: YOLOv8 Object Detection

```bash
# Convert YOLOv8 model
hailo-toolbox convert yolov8n.onnx \
    --hw-arch hailo8 \
    --input-shape 640,640,3 \
    --calib-set-path ./coco_samples \
    --output-dir ./models

# Use converted model for inference
hailo-toolbox infer ./models/yolov8n.hef \
    --source ./test_videos/traffic.mp4 \
    --task-name yolov8det \
    --save-dir ./results \
    --show
```

### Example 2: Real-time Camera Detection

```bash
# USB camera real-time detection
hailo-toolbox infer yolov8n.hef \
    --source 0 \
    --task-name yolov8det \
    --show

# IP camera real-time detection
hailo-toolbox infer yolov8n.hef \
    --source "rtsp://admin:password@192.168.1.100:554/stream" \
    --task-name yolov8det \
    --show
```

### Example 3: Batch Image Processing

```bash
# Process all images in folder
hailo-toolbox infer yolov8n.hef \
    --source ./test_images/ \
    --task-name yolov8det \
    --save-dir ./batch_results
```

### Example 4: Instance Segmentation

```bash
# Segmentation task
hailo-toolbox infer yolov8n_seg.hef \
    --source video.mp4 \
    --task-name yolov8seg \
    --show \
    --save-dir ./segmentation_results
```

### Example 5: Pose Estimation

```bash
# Pose estimation
hailo-toolbox infer yolov8s_pose.hef \
    --source video.mp4 \
    --task-name yolov8pe \
    --show \
    --save-dir ./pose_results
```

### Example 6: Using Python API

```python
from hailo_toolbox.inference import InferenceEngine
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig

# Custom preprocessing configuration
preprocess_config = PreprocessConfig(
    target_size=(640, 640),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Create inference engine
engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    preprocess_config=preprocess_config,
    show=True,
    save_dir="output/"
)

# Run inference
engine.run()
```

### Example 7: Server Mode (Advanced Usage)

```python
import queue
import threading
import numpy as np
from hailo_toolbox.inference import InferenceEngine

# Create inference engine
engine = InferenceEngine(
    model="models/yolov8n.hef",
    task_name="yolov8det"
)

# Start server mode
input_queue, output_queue = engine.start_server(
    enable_visualization=True,
    queue_size=30,
    server_timeout=2.0
)

# Process frames
def process_frames():
    for i in range(10):
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Send to inference queue
        frame_info = engine.shm_manager.write(frame, f"frame_{i}")
        input_queue.put(frame_info)
        
        # Get results
        try:
            result = output_queue.get(timeout=5.0)
            print(f"Processed frame {i}")
        except queue.Empty:
            print(f"Frame {i} processing timeout")

# Run processing
process_thread = threading.Thread(target=process_frames)
process_thread.start()
process_thread.join()

# Shutdown server
input_queue.put("SHUTDOWN")
```

## Common Issues

### Q1: Dependency conflicts during installation?
**A**: Recommend using virtual environment for installation:
```bash
python -m venv hailo_env
source hailo_env/bin/activate  # Linux/macOS
pip install -e .
```

### Q2: "calibration dataset required" error during model conversion?
**A**: Need to provide calibration dataset or use random calibration:
```bash
# Use calibration dataset
hailo-toolbox convert model.onnx --calib-set-path ./calibration_images

# Or use random calibration
hailo-toolbox convert model.onnx --use-random-calib-set
```

### Q3: Camera cannot be opened during inference?
**A**: Check camera permissions and device ID:
```bash
# Try different device IDs
hailo-toolbox infer model.hef --source 0 --task-name yolov8det  # First camera
hailo-toolbox infer model.hef --source 1 --task-name yolov8det  # Second camera

# Check camera devices on Linux
ls /dev/video*
```

### Q4: Slow model inference speed?
**A**: Refer to [Performance Optimization](#performance-optimization) section for suggestions.

### Q5: Support for custom callback functions?
**A**: Yes, custom callback functions can be implemented through the registration mechanism. See development documentation for details.

### Q6: How to handle different input sources?
**A**: Hailo Toolbox supports various input sources including images, videos, cameras, and network streams, automatically detecting input types.

## Performance Optimization

### 1. Hardware Optimization
- **Use Hailo Hardware Accelerators**: Get optimal inference performance
- **Choose Appropriate Hardware Architecture**: Select based on power and performance requirements
- **Optimize Input Resolution**: Balance accuracy and speed

### 2. Model Optimization
- **Quantization Optimization**: Use high-quality calibration datasets
- **Model Pruning**: Prune models before conversion
- **Batch Processing**: Use batch processing mode for image inference

### 3. System Optimization
- **Multi-threading**: Utilize multi-core CPUs for parallel processing
- **Memory Management**: Set reasonable cache sizes
- **I/O Optimization**: Use SSD storage, optimize data reading

### 4. Inference Optimization Example

```python
# Optimization configuration example
engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    # Preprocessing optimization
    preprocess_config={
        "target_size": (640, 640),
        "normalize": True,
        "batch_size": 4  # Batch processing
    },
    # Postprocessing optimization
    postprocess_config={
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "max_detections": 100
    },
    show=True
)
```

## Troubleshooting

### Debug Logging
```bash
# Enable verbose logging
export HAILO_LOG_LEVEL=DEBUG
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det

# View log files
ls *.log
cat hailo_toolbox.log
```

### Common Errors and Solutions

| Error Message | Possible Cause | Solution |
|---------------|----------------|----------|
| `Model file not found` | Incorrect model path | Check if model file path is correct |
| `Unsupported model format` | Unsupported model format | Confirm model format is in supported list |
| `CUDA out of memory` | Insufficient GPU memory | Reduce batch_size or use CPU |
| `Permission denied` | Insufficient permissions | Use sudo or check file permissions |
| `Task name not found` | Callback function not registered | Check task_name is correct or register custom callback |
| `Source not accessible` | Input source inaccessible | Check file path, camera permissions, or network connection |

### Performance Diagnostics
```python
# Performance monitoring example
import time
from hailo_toolbox.inference import InferenceEngine

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.frame_count = 0
    
    def start(self):
        self.start_time = time.time()
        self.frame_count = 0
    
    def update(self):
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"Processed {self.frame_count} frames, average FPS: {fps:.2f}")

# Use monitor
monitor = PerformanceMonitor()
monitor.start()

engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det"
)

# Call monitor.update() in inference loop
```

### Getting Help
- **GitHub Issues**: [Submit Issues](https://github.com/Seeed-Projects/hailo_toolbox/issues)
- **Documentation**: Check project documentation and README
- **Community**: Join developer community discussions

---

## Summary

Hailo Toolbox is a powerful deep learning model conversion and inference toolkit. Through this guide, you should be able to:

1. ✅ Successfully install and configure the tool
2. ✅ Convert various format deep learning models
3. ✅ Execute efficient model inference
4. ✅ Use Python API for custom development
5. ✅ Solve common problems and optimize performance

Key Features Summary:
- **Modular Architecture**: Extensible design based on registration mechanism
- **Multiple Input Sources**: Support for images, videos, cameras, network streams
- **Flexible APIs**: Support for both command line and Python APIs
- **High Performance**: Optimized inference engine and hardware acceleration support
- **Easy to Extend**: Simple custom callback function registration mechanism

If you encounter problems, please refer to the troubleshooting section or submit an Issue on GitHub. Enjoy using it!

**Update Date**: December 2024
**Version**: v2.0.0 