 # Hailo Toolbox

> **Language**: 🇺🇸 English | [🇨🇳 中文](README_zh.md)

A comprehensive deep learning model conversion and inference toolkit designed specifically for Hailo AI processors. This project aims to simplify the AI application development workflow based on Hailo devices, providing developers with a one-stop solution from model conversion to deployment inference.

- [Quick Start](docs/en/GET_STAR.md)
- [Developer Documentation](docs/en/DEV.md)
- [Model Conversion](docs/en/CONVERT.md)
- [Model Inference](docs/en/INFERENCE.md)


## 📦 Installation

### Hardware Prepare

|                                               reComputer AI R2140                                              |                                               reComputer AI Industrial R2145                                              |
| :----------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| ![Raspberry Pi AI Kit](https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/i/m/image114993560.jpeg) | ![reComputer R1100](https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/2/-/2-114993595-recomputer-ai-industrial-r2135-12.jpg) |
| [**Purchase Now**](https://www.seeedstudio.com/reComputer-AI-R2130-12-p-6368.html?utm_source=PiAICourse&utm_medium=github&utm_campaign=Course) | [**Purchase Now**](https://www.seeedstudio.com/reComputer-AI-Industrial-R2135-12-p-6432.html?utm_source=PiAICourse&utm_medium=github&utm_campaign=Course) |


### Requirements
- Python 3.8 ≤ version < 3.12
- Linux (recommended Ubuntu 18.04+), Windows 10+, macOS 10.15+

### Hailo Dataflow Compiler

> **Note:** You only need to install Hailo DFC on your own machine if you need to convert your own model

- Hailo Dataflow Compiler (for model conversion) [Installation Tutorial](https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/)

### HailoRT 

> **Note:** You can use command like below if you use reComputer

```bash
sudo apt update && sudo apt full-upgrade -y

sudo apt install hailo-all -y
```

> **Note:** Other hardwares please follow tutorial below

- HailoRT (for inference) [Installation Tutorial](https://wiki.seeedstudio.com/benchmark_of_multistream_inference_on_raspberrypi5_with_hailo8/#prepare-software)

### Install from Source
```bash
# Get project code
git clone https://github.com/Seeed-Projects/hailo_toolbox.git
cd hailo_toolbox
# Install hailo-toolbox
pip install -e .
```

### Install from Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv hailo_env
# Activate virtual environment
source hailo_env/bin/activate
# Get project code
git clone https://github.com/Seeed-Projects/hailo_toolbox.git
cd hailo_toolbox
# Install hailo-toolbox
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
```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

# Create input source
source = create_source("video.mp4")  # or camera: 0, or image: "image.jpg"

# Load model from ModelsZoo
inference = ModelsZoo.detection.yolov8s()

# Process frames
for img in source:
    results = inference.predict(img)
    for result in results:
        boxes = result.get_boxes()
        scores = result.get_scores()
        class_ids = result.get_class_ids()
        print(f"Detected {len(result)} objects")
```

### 📖 Complete Examples
Detailed usage examples for each supported task:

```bash
# Browse all available examples
ls examples/Hailo_*.py

# Object detection with visualization
python examples/Hailo_Object_Detection.py

# Human pose estimation
python examples/Hailo_Pose_Estimation.py

# Face detection and landmarks
python examples/Hailo_Face_Detection.py
```

> 📚 **Learn More**: See [`examples/README.md`](examples/README.md) for detailed documentation of all supported tasks and models.

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

### 🚀 Getting Started
- **[Examples Directory](examples/)** - Complete working examples for all supported tasks
- **[Examples README](examples/README.md)** - Detailed guide to all available examples
- **[Quick Start Guide](docs/en/GET_STAR.md)** / **[中文](docs/zh/GET_STAR.md)** - Installation and basic usage

### 📖 User Guides  
- **[Model Conversion Guide](docs/en/CONVERT.md)** / **[中文](docs/zh/CONVERT.md)** - How to convert models to HEF format
- **[Model Inference Guide](docs/en/INFERENCE.md)** / **[中文](docs/zh/INFERENCE.md)** - How to run inference with converted models
- **[Input Sources Guide](docs/en/SOURCE.md)** / **[中文](docs/zh/SOURCE.md)** - Supported input sources and configuration

### 🔧 Developer Documentation
- **[Developer Guide](docs/en/DEV.md)** / **[中文](docs/zh/DEV.md)** - How to implement custom models and callbacks
- **[Project Introduction](docs/en/INTRODUCE.md)** / **[中文](docs/zh/INTRODUCE.md)** - Detailed project overview and architecture



### 🚀 Quick Examples

```bash
# Run object detection example
python examples/Hailo_Object_Detection.py

# Run pose estimation example  
python examples/Hailo_Pose_Estimation.py

# Run face detection example
python examples/Hailo_Face_Detection.py

# View all examples
ls examples/Hailo_*.py
```

> 💡 **Tip**: Check the [`examples/`](examples/) directory for complete working examples of each task type. Each example includes model loading, inference, and result processing.


## 🤝 Contributing

We welcome community contributions! Please see our contributing guidelines:

1. **Report Issues**: Submit bug reports or feature requests
2. **Code Contributions**: Fork the project and submit pull requests
3. **Documentation**: Improve documentation and examples
4. **Testing**: Add test cases and performance benchmarks



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

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.