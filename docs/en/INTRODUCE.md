# Hailo Tool Box - Intelligent Vision Inference Toolbox

## Project Overview

Hailo Tools is a comprehensive deep learning model conversion and inference toolbox specifically designed for Hailo AI processors. This project aims to simplify the AI application development process based on Hailo devices, providing developers with a one-stop solution from model conversion to deployment inference. Through a highly modular architecture design and registration mechanism, users can easily extend and customize functionality, quickly validate and deploy various vision AI models.

## 🚀 Core Features

### Model Support
- **Multi-format Compatibility**: Supports Hailo HEF format and ONNX format models
- **Model Conversion**: Provides complete model conversion toolchain
- **Optimized Inference**: Inference engine optimized for Hailo hardware accelerators
- **Quantization Support**: Supports efficient inference of INT8 quantized models

### Vision Task Support
- **Object Detection**: YOLOv8, YOLOv5 and other mainstream detection models
- **Image Segmentation**: Semantic segmentation and instance segmentation
- **Pose Estimation**: Human keypoint detection and pose analysis
- **Classification Recognition**: Image classification and feature extraction

### Diverse Input Sources
- **File Input**: Image files, video files, image folders
- **Real-time Streams**: USB cameras, IP cameras, RTSP streams
- **Network Input**: HTTP/HTTPS image and video streams
- **Multi-source Concurrency**: Supports simultaneous processing of multiple input sources

### Post-processing and Visualization
- **Intelligent Post-processing**: Built-in post-processing algorithms for various vision tasks
- **Real-time Visualization**: Supports real-time rendering of bounding boxes, segmentation masks, keypoints
- **Result Saving**: Supports video saving and export of inference results
- **Performance Monitoring**: Real-time FPS statistics and performance analysis

### Extensible Design
- **Registration Mechanism**: Modular architecture based on registration pattern
- **Plugin System**: Supports custom post-processing functions and visualization schemes
- **Configuration-driven**: Flexible configuration file support
- **Developer-friendly**: Clean API interfaces, convenient for secondary development

## 🏗️ Project Architecture

### Modular Design
This project adopts a highly modular architecture design, with each functional module managed through a registration mechanism. Main modules include:

```
hailo_tools/
├── cli/                    # Command line interface module
│   ├── infer.py           # Inference command line tool
│   └── convert.py         # Conversion command line tool
├── sources/               # Input source management module
│   ├── base.py           # Input source base class definition
│   ├── file.py           # File input source implementation
│   ├── webcam.py         # Camera input source
│   ├── ip_camera.py      # IP camera input source
│   └── multi.py          # Multi-source manager
├── process/              # Inference processing module
│   ├── inference.py      # Inference engine core
│   ├── postprocess.py    # Post-processing algorithm library
│   └── visualization.py  # Visualization rendering engine
└── utils/               # Utility function module
    ├── config.py        # Configuration management
    ├── logging.py       # Logging system
    └── registry.py      # Registration manager
```

### Core Design Patterns

#### 1. Registration Mechanism (Registry Pattern)
- **Module Registration**: All functional modules are managed uniformly through the registry
- **Dynamic Loading**: Supports runtime dynamic loading and registration of new modules
- **Decoupled Design**: Loose coupling between modules, easy to maintain and extend

#### 2. Factory Pattern (Factory Pattern)
- **Source Factory**: Automatically creates corresponding input source instances based on input type
- **Processor Factory**: Creates processors based on model type and task type
- **Unified Interface**: Provides unified creation interface, simplifying usage complexity

#### 3. Strategy Pattern (Strategy Pattern)
- **Synchronization Strategy**: Different synchronization strategies for multi-source input
- **Post-processing Strategy**: Post-processing strategies for different vision tasks
- **Visualization Strategy**: Diverse result visualization schemes

## 🛠️ Technology Stack

### Core Dependencies
- **Python 3.8+**: Main development language
- **OpenCV 4.5+**: Computer vision processing library
- **NumPy**: Numerical computation and array operations
- **ONNX Runtime**: ONNX model inference engine
- **Hailo SDK**: Hailo hardware acceleration support

### Development Tools
- **Click**: Command line interface framework
- **PyYAML**: Configuration file parsing
- **Tqdm**: Progress bar display
- **Pillow**: Image processing extension
- **Requests**: HTTP request handling

### Testing Framework
- **Pytest**: Unit testing framework
- **Coverage**: Code coverage statistics
- **Mock**: Testing simulation tools

## 📋 Use Cases

### Industrial Applications
- **Quality Inspection**: Industrial product defect detection and quality control
- **Safety Monitoring**: Intelligent monitoring systems and anomaly detection
- **Automated Production**: Robot vision guidance and control

### Commercial Applications
- **Retail Analytics**: Customer flow statistics and behavior analysis
- **Intelligent Transportation**: Vehicle detection and traffic monitoring
- **Medical Imaging**: Medical image analysis and auxiliary diagnosis

### Educational Research
- **Algorithm Validation**: Deep learning model effectiveness validation
- **Prototype Development**: Rapid prototype construction and testing
- **Academic Research**: Computer vision algorithm research platform

## 🔧 Quick Start

### Environment Setup
```bash
# Clone project
git clone https://github.com/your-repo/hailo_tools.git
cd hailo_tools

# Install dependencies
pip install -e .

# Verify installation
hailo-toolbox --version
```

### Basic Usage
```bash
# Run object detection
hailo-toolbox infer models/yolov8n.hef -c yolov8det --source video.mp4

# Real-time camera inference
hailo-toolbox infer models/yolov8n.hef -c yolov8det --source 0 --show

# Batch image processing
hailo-toolbox infer models/yolov8n.hef -c yolov8det --source images/ --save
```

### Custom Extensions
```python
# Register custom post-processing function
from hailo_tools.utils.registry import register_callback

@register_callback("custom_detection")
def custom_detection_callback(outputs, frame, model_info):
    """
    Custom detection postprocessing function
    
    Args:
        outputs: Model inference outputs
        frame: Input frame
        model_info: Model metadata
    
    Returns:
        Processed frame with visualizations
    """
    # Implement custom post-processing logic
    processed_frame = apply_custom_processing(outputs, frame)
    return processed_frame
```

## 📊 Performance Features

### High-performance Inference
- **Hardware Acceleration**: Fully utilizes parallel computing capabilities of Hailo AI processors
- **Memory Optimization**: Intelligent memory management, reducing memory usage and copy overhead
- **Batch Processing Support**: Supports batch inference to improve overall throughput

### Concurrent Processing
- **Multi-threaded Architecture**: Independent threads for input reading, inference processing, result output
- **Asynchronous Processing**: Asynchronous I/O operations avoid blocking waits
- **Load Balancing**: Intelligent task scheduling and load distribution

### Real-time Performance
- **Low Latency**: Optimized inference pipeline, minimizing end-to-end latency
- **High Frame Rate**: Supports real-time processing of high frame rate video streams
- **Adaptive Adjustment**: Automatically adjusts processing parameters based on hardware performance

## 🌐 Platform Compatibility

### Operating System Support
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+
- **Windows**: Windows 10/11 (64-bit)
- **Embedded Linux**: Supports ARM architecture embedded platforms

### Hardware Requirements
- **CPU**: x86_64 or ARM64 architecture
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: Minimum 10GB available space
- **Hailo Device**: Hailo-8 or Hailo-15 AI processor

## 📖 Documentation Navigation

- [Quick Start Guide](GET_STAR.md) - Beginner tutorial
- [Development Documentation](DEV.md) - Detailed development guide  
- [Input Source Documentation](SOURCE.md) - Input source usage instructions
- [Inference Tutorial](INFERENCE.md) - Inference functionality detailed explanation
- [Model Conversion](CONVERT.md) - Model conversion guide

## 🤝 Contribution Guidelines

We welcome community contributions! Please participate in the project in the following ways:

1. **Submit Issues**: Report bugs or propose feature requests
2. **Code Contributions**: Fork the project and submit Pull Requests
3. **Documentation Improvement**: Improve documentation and example code
4. **Test Cases**: Add test cases and performance benchmarks

## 📄 Open Source License

This project adopts the [MIT License](../LICENSE) open source license, allowing free use, modification, and distribution.

## 📞 Technical Support

- **GitHub Issues**: [Project Issues Page](https://github.com/your-repo/hailo_tools/issues)
- **Email Support**: your.email@example.com
- **Technical Documentation**: [Online Documentation Site](https://your-docs-site.com)

---

*Hailo Tools - Making AI inference simpler and more efficient!* 