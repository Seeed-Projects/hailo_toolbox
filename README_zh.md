# Hailo 工具箱

> **Language**: [🇺🇸 English](README.md) | 🇨🇳 中文

专为 Hailo AI 处理器设计的综合深度学习模型转换和推理工具包。本项目旨在简化基于 Hailo 设备的 AI 应用开发工作流程，为开发者提供从模型转换到部署推理的一站式解决方案。


- [快速上手](docs/zh/GET_STAR.md)
- [开发者文档](docs/zh/DEV.md)
- [模型转换](docs/zh/CONVERT.md)
- [模型推理](docs/zh/INFERENCE.md)


## 📦 安装

### 要求
- Python 3.8 ≤ 版本 < 3.12
- Linux（推荐 Ubuntu 18.04+）、Windows 10+、macOS 10.15+
- Hailo Dataflow Compiler（用于模型转换）[安装教程](https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/)
- HailoRT（用于推理）[安装教程](https://wiki.seeedstudio.com/benchmark_on_rpi5_and_cm4_running_yolov8s_with_rpi_ai_kit/)

### 从源码安装
```bash
# 获取项目代码
git clone https://github.com/Seeed-Projects/hailo_toolbox.git
cd hailo_toolbox
# 安装hailo-toolbox
pip install -e .
```
### 从虚拟环境安装(推荐)

```bash
# 创建虚拟环境
python -m venv hailo_env
# 激活虚拟环境
source hailo_env/bin/activate
# 获取项目代码
git clone https://github.com/Seeed-Projects/hailo_toolbox.git
cd hailo_toolbox
# 安装hailo-toolbox
pip install -e .
```

### 验证安装
```bash
hailo-toolbox --version
hailo-toolbox --help
```

## 🚀 快速开始

### 模型转换
```bash
# 将 ONNX 模型转换为 HEF 格式
hailo-toolbox convert model.onnx --hw-arch hailo8 --calib-set-path ./calibration_data

# 使用随机校准进行快速转换
hailo-toolbox convert model.onnx --use-random-calib-set
```

### 模型推理
```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

# 创建输入源
source = create_source("video.mp4")  # 或摄像头：0，或图像："image.jpg"

# 从 ModelsZoo 加载模型
inference = ModelsZoo.detection.yolov8s()

# 处理帧
for img in source:
    results = inference.predict(img)
    for result in results:
        boxes = result.get_boxes()
        scores = result.get_scores()
        class_ids = result.get_class_ids()
        print(f"检测到 {len(result)} 个对象")
```

### 📖 完整示例
每个支持任务的详细使用示例：

```bash
# 浏览所有可用示例
ls examples/Hailo_*.py

# 带可视化的目标检测
python examples/Hailo_Object_Detection.py

# 人体姿态估计
python examples/Hailo_Pose_Estimation.py

# 人脸检测和关键点
python examples/Hailo_Face_Detection.py
```

> 📚 **了解更多**：查看 [`examples/README.md`](examples/README.md) 获取所有支持任务和模型的详细文档。

### 自定义回调注册
```python
from hailo_toolbox.inference.core import CALLBACK_REGISTRY

@CALLBACK_REGISTRY.registryPostProcessor("custom_model")
class CustomPostProcessor:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, results, original_shape=None):
        # 自定义后处理逻辑
        return processed_results
```

## 📚 文档

### 🚀 入门指南
- **[示例目录](examples/)** - 所有支持任务的完整工作示例
- **[示例 README](examples/README.md)** - 所有可用示例的详细指南
- **[快速开始指南](docs/zh/GET_STAR.md)** / **[English](docs/en/GET_STAR.md)** - 安装和基本使用

### 📖 用户指南  
- **[模型转换指南](docs/zh/CONVERT.md)** / **[English](docs/en/CONVERT.md)** - 如何将模型转换为 HEF 格式
- **[模型推理指南](docs/zh/INFERENCE.md)** / **[English](docs/en/INFERENCE.md)** - 如何使用转换后的模型运行推理
- **[输入源指南](docs/zh/SOURCE.md)** / **[English](docs/en/SOURCE.md)** - 支持的输入源和配置

### 🔧 开发者文档
- **[开发者指南](docs/zh/DEV.md)** / **[English](docs/en/DEV.md)** - 如何实现自定义模型和回调
- **[项目介绍](docs/zh/INTRODUCE.md)** / **[English](docs/en/INTRODUCE.md)** - 详细的项目概述和架构



### 🚀 快速示例

```bash
# 运行目标检测示例
python examples/Hailo_Object_Detection.py

# 运行姿态估计示例  
python examples/Hailo_Pose_Estimation.py

# 运行人脸检测示例
python examples/Hailo_Face_Detection.py

# 查看所有示例
ls examples/Hailo_*.py
```

> 💡 **提示**：查看 [`examples/`](examples/) 目录获取每种任务类型的完整工作示例。每个示例都包含模型加载、推理和结果处理。


## 🤝 贡献

我们欢迎社区贡献！请查看我们的贡献指南：

1. **报告问题**：提交错误报告或功能请求
2. **代码贡献**：Fork 项目并提交拉取请求
3. **文档**：改进文档和示例
4. **测试**：添加测试用例和性能基准



## 🔗 链接

- **GitHub 仓库**：[https://github.com/Seeed-Projects/hailo_toolbox](https://github.com/Seeed-Projects/hailo_toolbox)
- **问题**：[https://github.com/Seeed-Projects/hailo_toolbox/issues](https://github.com/Seeed-Projects/hailo_toolbox/issues)
- **Hailo AI**：[https://hailo.ai](https://hailo.ai)

## 📞 支持

- **GitHub Issues**：用于错误报告和功能请求
- **文档**：全面的指南和 API 参考
- **社区**：加入我们的开发者社区讨论

---

*使用 Hailo 工具箱让 AI 推理更简单、更高效！* 

## 📄 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](LICENSE)。