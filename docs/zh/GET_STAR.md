# Hailo Toolbox 快速开始指南

本文档将介绍如何安装和使用Hailo Toolbox工具，这是一个专为深度学习模型转换和推理设计的综合工具包。本指南包含从基础安装到高级使用的完整说明。

## 目录

- [系统要求](#系统要求)
- [安装](#安装)
- [验证安装](#验证安装)
- [项目结构](#项目结构)
- [模型转换](#模型转换)
- [模型推理](#模型推理)

## 系统要求

### 基础要求
- **Python版本**: 3.8 ≤ Python < 3.12
- **操作系统**: Linux (推荐Ubuntu 18.04+), Windows 10+
- **内存**: 至少8GB RAM（推荐16GB+）
- **存储**: 至少2GB可用空间

### Hailo特定要求
- **[Hailo Dataflow Compiler](https://hailo.ai/developer-zone/software-downloads/)**: 用于模型转换功能（如何使用转换功能则必须安装，仅支持X86架构且Linux系统），可参考[安装教程](https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/)
- **[HailoRT](https://hailo.ai/developer-zone/software-downloads/)**: 用于推理功能（使用推理功能时必须安装），可参考[教程安装](https://wiki.seeedstudio.com/benchmark_on_rpi5_and_cm4_running_yolov8s_with_rpi_ai_kit/)
- **Hailo硬件**: 用于硬件加速推理（使用推理功能时必须安装）

### Python依赖包
核心依赖包会在安装时自动安装：
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

## 安装Hailo-toolbox

### 方式一：从源码安装（推荐）

```bash
# 克隆项目源码
git clone https://github.com/Seeed-Projects/hailo_toolbox.git

# 进入项目目录
cd hailo_toolbox

# 安装项目（开发模式）
pip install -e .

# 或者直接安装
pip install .
```

### 方式二：创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv hailo_env

# 激活虚拟环境
# Linux/macOS:
source hailo_env/bin/activate
# Windows:
hailo_env\Scripts\activate

# 安装项目
git clone https://github.com/Seeed-Projects/hailo_toolbox.git
cd hailo_toolbox
pip install -e .
```

## 验证安装

安装完成后，可以通过以下命令验证安装是否成功：

```bash
# 查看版本信息
hailo-toolbox --version

# 查看帮助信息
hailo-toolbox --help

# 查看转换功能帮助
hailo-toolbox convert --help

# 查看推理功能帮助
hailo-toolbox infer --help
```


## 模型转换

Hailo Toolbox支持将多种深度学习框架的模型转换为高效的`.hef`格式，以便在Hailo硬件上运行。

### 支持的模型格式

| 框架 | 格式 | 是否支持 | 目标格式 | 备注 |
|------|------|----------|----------|------|
| ONNX | .onnx | ✅ | .hef | 推荐格式 |
| TensorFlow | .h5 | ✅ | .hef | Keras模型 |
| TensorFlow | SavedModel.pb | ✅ | .hef | TensorFlow SavedModel |
| TensorFlow Lite | .tflite | ✅ | .hef | 移动端模型 |
| PyTorch | .pt (torchscript) | ✅ | .hef | TorchScript模型 |
| PaddlePaddle | inference model | ✅ | .hef | PaddlePaddle推理模型 |

### 基本转换命令

```bash
# 查看转换帮助
hailo-toolbox convert --help

# 基础转换（ONNX到HEF）
hailo-toolbox convert model.onnx --hw-arch hailo8

# 完整转换示例
hailo-toolbox convert model.onnx \
    --hw-arch hailo8 \
    --input-shape 320,320,3 \
    --save-onnx \
    --output-dir outputs \
    --profile \
    --calib-set-path ./calibration_images
```

### 转换参数详解

| 参数 | 必选 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `model` | ✅ | - | 待转换的模型文件路径 | `model.onnx` |
| `--hw-arch` | ❌ | `hailo8` | 目标Hailo硬件架构 | `hailo8`, `hailo8l`, `hailo15`, `hailo15l` |
| `--calib-set-path` | ❌ | None | 校准数据集文件夹路径 | `./calibration_data/` |
| `--use-random-calib-set` | ❌ | False | 使用随机数据进行校准 | - |
| `--calib-set-size` | ❌ | None | 校准数据集大小 | `100` |
| `--model-script` | ❌ | None | 自定义模型脚本路径 | `./custom_script.py` |
| `--end-nodes` | ❌ | None | 指定模型输出节点 | `output1,output2` |
| `--input-shape` | ❌ | `[640,640,3]` | 模型输入shape | `320,320,3` |
| `--save-onnx` | ❌ | False | 保存编译后的ONNX文件 | - |
| `--output-dir` | ❌ | 模型同目录 | 输出文件保存目录 | `./outputs/` |
| `--profile` | ❌ | False | 生成性能分析报告 | - |

## 模型推理

Hailo Toolbox提供灵活的推理接口，支持多种输入源和输出格式。

### 推理示例

```bash
# 查看推理帮助
cd examples

# 基础推理示例
python Hailo_Object_Detection.py
```

### 支持的输入源类型

| 输入源类型 | 格式 | 示例 | 说明 |
|------------|------|------|------|
| 图片文件 | jpg, png, bmp等 | `image.jpg` | 单张图片推理 |
| 图片文件夹 | 目录路径 | `./images/` | 批量图片推理 |
| 视频文件 | mp4, avi, mov等 | `video.mp4` | 视频文件推理 |
| USB摄像头 | 设备ID | `0`, `1` | 实时摄像头推理 |
| IP摄像头 | RTSP/HTTP流 | `rtsp://ip:port/stream` | 网络摄像头推理 |
| 网络视频流 | URL | `http://example.com/stream` | 在线视频流推理 |


### 代码解释

为了帮助理解

```python
from hailo_toolbox import create_source     # 加载图像源的API
from hailo_toolbox.models import ModelsZoo  # 模型库
from hailo_toolbox.process.visualization import DetectionVisualization  # 已经实现的目标检测可视化工具
import cv2  # opencv工具

if __name__ == "__main__":
    # 创建模型输入源
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load YOLOv8n detection model
    # 加载目标检测任务下的yolov8s模型
    inference = ModelsZoo.detection.yolov8s()
    # 加载可视化模块
    visualization = DetectionVisualization()

    # 读取图像源，按帧读取
    for img in source:
        # 将图像交给模型进行推理预测，推理模块会根据模型的配置进行对应的预处理与后处理，并将处理结果包装成可直接使用的数据
        results = inference.predict(img)
        # 依次获取每张图像的推理结果，模型接受多张图像同时推理，所以返回的结果是每张图像的处理结果
        for result in results:
            # 可视化推理结果
            img = visualization.visualize(img, result)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)
            # print(f"Detected {len(result)} objects")
            # 获取当前图像的所预测目标框
            boxes = result.get_boxes()
            # 获取当前图像的所预测置信度
            scores = result.get_scores()
            # 获取当前图像所预测的类ID
            class_ids = result.get_class_ids()

            # Show first 5 detection results
            for i in range(min(5, len(result))):
                print(
                    f"  Object{i}: bbox{boxes[i]}, score{scores[i]:.3f}, class{class_ids[i]}"
                )
            print("---")


```

