 # Hailo Toolbox

A comprehensive deep learning model conversion and inference toolkit designed specifically for Hailo AI processors. This project aims to simplify the AI application development workflow based on Hailo devices, providing developers with a one-stop solution from model conversion to deployment inference.

- [Quick Start](docs/en/GET_STAR.md)
- [Developer Documentation](docs/en/DEV.md)
- [Model Conversion](docs/en/CONVERT.md)
- [Model Inference](docs/en/INFERENCE.md)


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