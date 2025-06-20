# Hailo Toolbox 快速开始指南

本文档将介绍如何安装和使用Hailo Toolbox工具，这是一个专为深度学习模型转换和推理设计的综合工具包。本指南包含从基础安装到高级使用的完整说明。

## 目录

- [系统要求](#系统要求)
- [安装](#安装)
- [验证安装](#验证安装)
- [项目结构](#项目结构)
- [模型转换](#模型转换)
- [模型推理](#模型推理)
- [API服务模式](#api服务模式)
- [使用示例](#使用示例)
- [常见问题](#常见问题)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

## 系统要求

### 基础要求
- **Python版本**: 3.8 ≤ Python < 3.12
- **操作系统**: Linux (推荐Ubuntu 18.04+), Windows 10+, macOS 10.15+
- **内存**: 至少8GB RAM（推荐16GB+）
- **存储**: 至少2GB可用空间

### Hailo特定要求
- **[Hailo Dataflow Compiler](https://hailo.ai/developer-zone/software-downloads/)**: 用于模型转换功能（必须安装，仅支持X86架构且Linux系统）
- **[HailoRT](https://hailo.ai/developer-zone/software-downloads/)**: 用于推理功能（推理时必须安装）
- **Hailo硬件**: 用于硬件加速推理（必须安装）

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

## 安装

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

## 项目结构

```
hailo_toolbox/
├── cli/                    # 命令行接口
│   ├── config.py          # 参数配置
│   ├── convert.py         # 模型转换CLI
│   ├── infer.py           # 模型推理CLI
│   └── server.py          # API服务器
├── converters/            # 模型转换器
├── inference/             # 推理引擎
│   ├── base.py           # 基础推理类
│   ├── hailo_engine.py   # Hailo推理引擎
│   ├── onnx_engine.py    # ONNX推理引擎
│   └── pipeline.py       # 推理管道
├── process/              # 数据处理模块
├── sources/              # 数据源管理
├── utils/                # 工具函数
└── models/              # 模型管理
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

### 转换最佳实践

1. **准备校准数据集**：
   ```bash
   # 创建校准数据集目录
   mkdir calibration_images
   # 添加代表性图像（建议50-200张）
   cp representative_images/* calibration_images/
   ```

2. **选择合适的硬件架构**：
   - `hailo8`: 标准Hailo-8芯片
   - `hailo8l`: Hailo-8L低功耗版本
   - `hailo15`: 新一代Hailo-15芯片
   - `hailo15l`: Hailo-15L版本

3. **优化输入尺寸**：
   ```bash
   # 对于目标检测模型，常用尺寸：
   hailo-toolbox convert yolov8n.onnx --input-shape 640,640,3 --hw-arch hailo8
   
   # 对于分类模型：
   hailo-toolbox convert resnet50.onnx --input-shape 224,224,3 --hw-arch hailo8
   ```

## 模型推理

Hailo Toolbox提供灵活的推理接口，支持多种输入源和输出格式。

### 基本推理命令

```bash
# 查看推理帮助
hailo-toolbox infer --help

# 基础推理示例
hailo-toolbox infer model.hef --source video.mp4 --infer-name yolov8det --show

# 完整推理示例
hailo-toolbox infer yolov8.hef \
    --source 0 \
    --infer-name yolov8det \
    --save-dir ./results \
    --show
```

### 推理参数详解

| 参数 | 必选 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `model` | ✅ | - | 模型文件路径(.hef或.onnx) | `model.hef` |
| `--source` | ✅ | - | 输入源路径 | 见下表 |
| `--infer-name` | ❌ | `yolov8det` | 推理回调函数名称 | `yolov8det`, `yolov8seg`, `yolov8pose` |
| `--save-dir` | ❌ | None | 结果保存目录 | `./results/` |
| `--show` | ❌ | False | 实时显示结果 | - |

### 支持的输入源类型

| 输入源类型 | 格式 | 示例 | 说明 |
|------------|------|------|------|
| 图片文件 | jpg, png, bmp等 | `image.jpg` | 单张图片推理 |
| 图片文件夹 | 目录路径 | `./images/` | 批量图片推理 |
| 视频文件 | mp4, avi, mov等 | `video.mp4` | 视频文件推理 |
| USB摄像头 | 设备ID | `0`, `1` | 实时摄像头推理 |
| IP摄像头 | RTSP/HTTP流 | `rtsp://ip:port/stream` | 网络摄像头推理 |
| 网络视频流 | URL | `http://example.com/stream` | 在线视频流推理 |

### 可用的推理回调函数

| 回调函数名 | 功能 | 适用模型 | 输出 |
|------------|------|----------|------|
| `yolov8det` | 目标检测 | YOLOv8检测模型 | 边界框+类别+置信度 |
| `yolov8seg` | 实例分割 | YOLOv8分割模型 | 分割掩码+边界框 |
| `yolov8pose` | 姿态估计 | YOLOv8姿态模型 | 关键点+骨架连接 |

## API服务模式

Hailo Toolbox提供RESTful API服务，支持远程调用和集成。

### 启动API服务器

```bash
# 启动基础服务器
hailo-toolbox server --host 0.0.0.0 --port 8080

# 启动服务器并加载配置
hailo-toolbox server --config server_config.yaml --verbose
```

### API端点

| 端点 | 方法 | 功能 | 参数 |
|------|------|------|------|
| `/api/models` | GET | 获取已加载的模型列表 | - |
| `/api/models` | POST | 加载新模型 | `model_path`, `model_id` |
| `/api/models/<id>` | DELETE | 卸载模型 | - |
| `/api/sources` | GET | 获取数据源列表 | - |
| `/api/sources` | POST | 创建数据源 | `source_type`, `config` |
| `/api/pipelines` | GET | 获取推理管道列表 | - |
| `/api/pipelines` | POST | 创建推理管道 | `source_id`, `model_id` |
| `/api/infer` | POST | 执行推理 | `image_data`, `model_id` |

### API使用示例

```python
import requests
import json

# 加载模型
response = requests.post('http://localhost:8080/api/models', 
                        json={'model_path': 'yolov8n.hef', 'model_id': 'yolo'})

# 执行推理
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8080/api/infer', 
                           files=files, 
                           data={'model_id': 'yolo'})
    
results = response.json()
print(results)
```

## 使用示例

### 示例1：YOLOv8目标检测

```bash
# 转换YOLOv8模型
hailo-toolbox convert yolov8n.onnx \
    --hw-arch hailo8 \
    --input-shape 640,640,3 \
    --calib-set-path ./coco_samples \
    --output-dir ./models

# 使用转换后的模型进行推理
hailo-toolbox infer ./models/yolov8n.hef \
    --source ./test_videos/traffic.mp4 \
    --infer-name yolov8det \
    --save \
    --save-dir ./results \
    --show
```

### 示例2：实时摄像头检测

```bash
# USB摄像头实时检测
hailo-toolbox infer yolov8n.hef \
    --source 0 \
    --infer-name yolov8det \
    --show

# IP摄像头实时检测
hailo-toolbox infer yolov8n.hef \
    --source "rtsp://admin:password@192.168.1.100:554/stream" \
    --infer-name yolov8det \
    --show
```

### 示例3：批量图片处理

```bash
# 处理文件夹中的所有图片
hailo-toolbox infer yolov8n.hef \
    --source ./test_images/ \
    --infer-name yolov8det \
    --save \
    --save-dir ./batch_results
```

### 示例4：使用Python API

```python
from hailo_toolbox.inference import InferencePipeline
from hailo_toolbox.sources import FileSource

# 创建推理管道
pipeline = InferencePipeline(
    model_path="yolov8n.hef",
    callback_name="yolov8det"
)

# 创建文件源
source = FileSource("test_video.mp4")

# 运行推理
for frame in source:
    results = pipeline.infer(frame)
    print(f"检测到 {len(results.detections)} 个目标")
```

## 常见问题

### Q1: 安装时出现依赖冲突怎么办？
**A**: 建议使用虚拟环境安装：
```bash
python -m venv hailo_env
source hailo_env/bin/activate  # Linux/macOS
pip install -e .
```

### Q2: 转换模型时出现"calibration dataset required"错误？
**A**: 需要提供校准数据集或使用随机校准：
```bash
# 使用校准数据集
hailo-toolbox convert model.onnx --calib-set-path ./calibration_images

# 或使用随机校准
hailo-toolbox convert model.onnx --use-random-calib-set
```

### Q3: 推理时摄像头无法打开？
**A**: 检查摄像头权限和设备ID：
```bash
# 尝试不同的设备ID
hailo-toolbox infer model.hef --source 0  # 第一个摄像头
hailo-toolbox infer model.hef --source 1  # 第二个摄像头

# 在Linux下检查摄像头设备
ls /dev/video*
```

### Q4: 模型推理速度慢怎么办？
**A**: 参考[性能优化](#性能优化)部分的建议。

### Q5: 支持自定义回调函数吗？
**A**: 是的，可以通过继承`InferenceCallback`类来实现自定义回调函数。

## 性能优化

### 1. 硬件优化
- **使用Hailo硬件加速器**：获得最佳推理性能
- **选择合适的硬件架构**：根据功耗和性能需求选择
- **优化输入分辨率**：平衡精度和速度

### 2. 模型优化
- **量化优化**：使用高质量的校准数据集
- **模型剪枝**：在转换前对模型进行剪枝
- **批处理**：对于图片推理，使用批处理模式

### 3. 系统优化
- **多线程处理**：利用多核CPU进行并行处理
- **内存管理**：合理设置缓存大小
- **I/O优化**：使用SSD存储，优化数据读取

### 4. 推理优化示例

```python
# 优化配置示例
config = {
    "batch_size": 4,           # 批处理大小
    "num_threads": 8,          # 线程数
    "cache_size": 100,         # 缓存大小
    "device": "hailo",         # 使用Hailo设备
    "precision": "int8"        # 使用int8精度
}
```

## 故障排除

### 日志调试
```bash
# 启用详细日志
export HAILO_LOG_LEVEL=DEBUG
hailo-toolbox infer model.hef --source video.mp4 --infer-name yolov8det

# 查看日志文件
ls *.log
cat hailo_toolbox.log
```

### 常见错误及解决方案

| 错误信息 | 可能原因 | 解决方案 |
|----------|----------|----------|
| `Model file not found` | 模型路径错误 | 检查模型文件路径是否正确 |
| `Unsupported model format` | 模型格式不支持 | 确认模型格式是否在支持列表中 |
| `CUDA out of memory` | GPU内存不足 | 减少batch_size或使用CPU |
| `Permission denied` | 权限不足 | 使用sudo或检查文件权限 |
| `Port already in use` | 端口被占用 | 更换端口或停止占用进程 |

### 获取帮助
- **GitHub Issues**: [提交问题](https://github.com/Seeed-Projects/hailo_toolbox/issues)
- **文档**: 查看项目文档和README
- **社区**: 加入开发者社区讨论

---

## 总结

Hailo Toolbox是一个功能强大的深度学习模型转换和推理工具包。通过本指南，您应该能够：

1. ✅ 成功安装和配置工具
2. ✅ 转换各种格式的深度学习模型
3. ✅ 执行高效的模型推理
4. ✅ 使用API服务模式
5. ✅ 解决常见问题和优化性能

如果遇到问题，请参考故障排除部分或在GitHub上提交Issue。祝您使用愉快！

**更新日期**: 2024年6月
**版本**: v1.0.0




