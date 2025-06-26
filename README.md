# Hailo Toolbox

A comprehensive deep learning model conversion and inference toolkit designed specifically for Hailo AI processors. This project aims to simplify the AI application development workflow based on Hailo devices, providing developers with a one-stop solution from model conversion to deployment inference.

## 🚀 Key Features

### Model Support
- **Multi-format Compatibility**: Supports Hailo HEF format and ONNX format models
- **Model Conversion**: Complete model conversion toolchain from various frameworks (ONNX, TensorFlow, PyTorch, PaddlePaddle, TensorFlow Lite)
- **Optimized Inference**: Inference engine optimized for Hailo hardware accelerators
- **Quantization Support**: Efficient inference for INT8 quantized models

### Vision Tasks
- **Object Detection**: YOLOv8, YOLOv5 and other mainstream detection models
- **Image Segmentation**: Semantic segmentation and instance segmentation
- **Pose Estimation**: Human keypoint detection and pose analysis
- **Image Classification**: Image classification and feature extraction

### Input Sources
- **File Input**: Image files, video files, image folders
- **Real-time Streaming**: USB cameras, IP cameras, RTSP streams
- **Network Input**: HTTP/HTTPS image and video streams
- **Multi-source Concurrent**: Support for processing multiple input sources simultaneously

### Processing Pipeline
- **Intelligent Post-processing**: Built-in post-processing algorithms for various vision tasks
- **Real-time Visualization**: Real-time rendering of bounding boxes, segmentation masks, keypoints
- **Result Saving**: Support for video saving and export of inference results
- **Performance Monitoring**: Real-time FPS statistics and performance analysis

### Extensibility
- **Registration Mechanism**: Modular architecture based on registration pattern
- **Plugin System**: Support for custom post-processing functions and visualization schemes
- **Configuration-driven**: Flexible configuration file support
- **Developer-friendly**: Clean API interface for easy secondary development

## 🏗️ Architecture

### Modular Design
```
hailo_toolbox/
├── cli/                    # Command-line interface
├── converters/            # Model converters
├── inference/             # Inference engines
├── process/              # Data processing modules
├── sources/              # Input source management
├── utils/                # Utility functions
└── models/              # Model management
```

### Core Components
- **InferenceEngine**: Core inference orchestrator with flexible callback system
- **CallbackRegistry**: Universal registry for managing callbacks by name and type
- **Multi-backend Support**: Hailo (.hef) and ONNX (.onnx) inference engines
- **Source Management**: Unified interface for various input sources
- **Processing Pipeline**: Configurable preprocessing, inference, and postprocessing

## 📦 Installation

### Requirements
- Python 3.8 ≤ version < 3.12
- Linux (recommended Ubuntu 18.04+), Windows 10+, macOS 10.15+
- Hailo Dataflow Compiler (for model conversion)
- HailoRT (for inference)

### Install from Source
```bash
git clone https://github.com/Seeed-Projects/hailo_toolbox.git
cd hailo_toolbox
pip install -e .
```

### Verify Installation
```bash
hailo-toolbox --version
hailo-toolbox --help
```

## 🚀 Quick Start

### Model Conversion
```bash
# Convert ONNX model to HEF format
hailo-toolbox convert model.onnx --hw-arch hailo8 --calib-set-path ./calibration_data

# Quick conversion with random calibration
hailo-toolbox convert model.onnx --use-random-calib-set
```

### Model Inference
```bash
# Image inference
hailo-toolbox infer model.hef --source image.jpg --task-name yolov8det --show

# Video inference
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --save-dir ./results

# Real-time camera
hailo-toolbox infer model.hef --source 0 --task-name yolov8det --show
```

### Python API
```python
from hailo_toolbox.inference import InferenceEngine

# Create inference engine
engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    show=True,
    save_dir="output/"
)

# Run inference
engine.run()
```

### Custom Callback Registration
```python
from hailo_toolbox.inference.core import CALLBACK_REGISTRY

@CALLBACK_REGISTRY.registryPostProcessor("custom_model")
class CustomPostProcessor:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, results, original_shape=None):
        # Custom post-processing logic
        return processed_results
```

## 📚 Documentation

### User Guides
- **[Quick Start Guide](docs/zh/GET_STAR.md)** / **[English](docs/en/GET_STAR.md)** - Installation and basic usage
- **[Model Conversion Guide](docs/zh/CONVERT.md)** / **[English](docs/en/CONVERT.md)** - How to convert models to HEF format
- **[Model Inference Guide](docs/zh/INFERENCE.md)** / **[English](docs/en/INFERENCE.md)** - How to run inference with converted models
- **[Input Sources Guide](docs/zh/SOURCE.md)** / **[English](docs/en/SOURCE.md)** - Supported input sources and configuration

### Developer Documentation
- **[Developer Guide](docs/zh/DEV.md)** / **[English](docs/en/DEV.md)** - How to implement custom models and callbacks
- **[Project Introduction](docs/zh/INTRODUCE.md)** / **[English](docs/en/INTRODUCE.md)** - Detailed project overview and architecture

## 🎯 Use Cases

### Industrial Applications
- Quality inspection and defect detection
- Security monitoring and anomaly detection
- Robotic vision guidance and control

### Commercial Applications
- Retail analytics and customer behavior analysis
- Intelligent transportation and traffic monitoring
- Medical imaging analysis and assisted diagnosis

### Research and Education
- Deep learning model validation
- Rapid prototyping and testing
- Computer vision algorithm research platform

## 🛠️ Supported Models

### Built-in Task Types
- `yolov8det`: YOLOv8 object detection
- `yolov8seg`: YOLOv8 instance segmentation  
- `yolov8pe`: YOLOv8 pose estimation

### Input Formats
- **ONNX** (.onnx) - Recommended universal format
- **TensorFlow** (.h5, saved_model.pb)
- **TensorFlow Lite** (.tflite)
- **PyTorch** (.pt - TorchScript)
- **PaddlePaddle** (inference models)

### Hardware Support
- Hailo-8 AI processor
- Hailo-8L (low-power variant)
- Hailo-15 AI processor
- Hailo-15L (low-power variant)

## 🔧 Command Line Interface

### Model Conversion
```bash
hailo-toolbox convert <model_file> [OPTIONS]
  --hw-arch ARCH              Hardware architecture (hailo8, hailo8l, hailo15, hailo15l)
  --calib-set-path PATH       Calibration dataset path
  --use-random-calib-set      Use random calibration data
  --input-shape SHAPE         Model input shape (e.g., 640,640,3)
  --output-dir DIR            Output directory
  --save-onnx                 Save optimized ONNX model
  --profile                   Generate performance profile
```

### Model Inference
```bash
hailo-toolbox infer <model_file> [OPTIONS]
  --source SOURCE             Input source (image, video, camera, folder)
  --task-name NAME            Task name for callback lookup
  --save-dir DIR              Results save directory
  --show                      Display results in real-time
```

## 🤝 Contributing

We welcome community contributions! Please see our contributing guidelines:

1. **Report Issues**: Submit bug reports or feature requests
2. **Code Contributions**: Fork the project and submit pull requests
3. **Documentation**: Improve documentation and examples
4. **Testing**: Add test cases and performance benchmarks

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🔗 Links

- **GitHub Repository**: [https://github.com/Seeed-Projects/hailo_toolbox](https://github.com/Seeed-Projects/hailo_toolbox)
- **Issues**: [https://github.com/Seeed-Projects/hailo_toolbox/issues](https://github.com/Seeed-Projects/hailo_toolbox/issues)
- **Hailo AI**: [https://hailo.ai](https://hailo.ai)

## 📞 Support

- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference
- **Community**: Join our developer community discussions

---

*Making AI inference simpler and more efficient with Hailo Toolbox!*