# Hailo Toolbox 快速开始指南

本文档将介绍如何安装和使用Hailo Toolbox工具，这是一个专为深度学习模型转换和推理设计的综合工具包。本指南包含从基础安装到高级使用的完整说明。

## 目录

- [系统要求](#系统要求)
- [安装](#安装)
- [验证安装](#验证安装)
- [项目结构](#项目结构)
- [模型转换](#模型转换)
- [模型推理](#模型推理)
- [Python API使用](#python-api使用)
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
- **Hailo硬件**: 用于硬件加速推理（必选）

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
│   ├── core.py           # 核心推理引擎和注册机制
│   ├── hailo_engine.py   # Hailo推理引擎
│   ├── onnx_engine.py    # ONNX推理引擎
│   └── pipeline.py       # 推理管道
├── process/              # 数据处理模块
│   ├── preprocessor/     # 预处理模块
│   ├── postprocessor/    # 后处理模块
│   └── callback.py       # 回调函数
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

## 模型推理

Hailo Toolbox提供灵活的推理接口，支持多种输入源和输出格式。

### 基本推理命令

```bash
# 查看推理帮助
hailo-toolbox infer --help

# 基础推理示例
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --show

# 完整推理示例
hailo-toolbox infer yolov8.hef \
    --source 0 \
    --task-name yolov8det \
    --save-dir ./results \
    --show
```

### 推理参数详解

| 参数 | 必选 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `model` | ✅ | - | 模型文件路径(.hef或.onnx) | `model.hef` |
| `--source` | ✅ | - | 输入源路径 | 见下表 |
| `--task-name` | ❌ | `yolov8det` | 任务名称，用于回调函数查找 | `yolov8det`, `yolov8seg`, `yolov8pe` |
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
| `yolov8pe` | 姿态估计 | YOLOv8姿态模型 | 关键点+骨架连接 |

## Python API使用

### 基本使用方式

```python
from hailo_toolbox.inference import InferenceEngine

# 新式API - 直接参数（推荐）
engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    show=True,
    save_dir="output/"
)
engine.run()

# 旧式API - 配置对象（向后兼容）
from hailo_toolbox.utils.config import Config

config = Config()
config.model = "models/yolov8n.hef"
config.source = "video.mp4"
config.task_name = "yolov8det"
config.show = True

engine = InferenceEngine(config, "yolov8det")
engine.run()
```

### 自定义回调函数

```python
from hailo_toolbox.inference.core import CALLBACK_REGISTRY, InferenceEngine
import numpy as np
import cv2

@CALLBACK_REGISTRY.registryPostProcessor("custom")
class CustomPostProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, results, original_shape=None):
        # 自定义后处理逻辑
        processed_results = []
        for k, v in results.items():
            # 处理模型输出
            processed_results.append(self.process_output(v))
        return processed_results

@CALLBACK_REGISTRY.registryVisualizer("custom")
class CustomVisualizer:
    def __init__(self, config):
        self.config = config

    def __call__(self, original_frame, results):
        # 自定义可视化逻辑
        vis_frame = original_frame.copy()
        for result in results:
            # 绘制结果
            cv2.putText(vis_frame, str(result), (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return vis_frame

# 使用自定义回调
engine = InferenceEngine(
    model="models/custom_model.hef",
    source="video.mp4",
    task_name="custom",
    show=True
)
engine.run()
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
    --task-name yolov8det \
    --save-dir ./results \
    --show
```

### 示例2：实时摄像头检测

```bash
# USB摄像头实时检测
hailo-toolbox infer yolov8n.hef \
    --source 0 \
    --task-name yolov8det \
    --show

# IP摄像头实时检测
hailo-toolbox infer yolov8n.hef \
    --source "rtsp://admin:password@192.168.1.100:554/stream" \
    --task-name yolov8det \
    --show
```

### 示例3：批量图片处理

```bash
# 处理文件夹中的所有图片
hailo-toolbox infer yolov8n.hef \
    --source ./test_images/ \
    --task-name yolov8det \
    --save-dir ./batch_results
```

### 示例4：实例分割

```bash
# 分割任务
hailo-toolbox infer yolov8n_seg.hef \
    --source video.mp4 \
    --task-name yolov8seg \
    --show \
    --save-dir ./segmentation_results
```

### 示例5：姿态估计

```bash
# 姿态估计
hailo-toolbox infer yolov8s_pose.hef \
    --source video.mp4 \
    --task-name yolov8pe \
    --show \
    --save-dir ./pose_results
```

### 示例6：使用Python API

```python
from hailo_toolbox.inference import InferenceEngine
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig

# 自定义预处理配置
preprocess_config = PreprocessConfig(
    target_size=(640, 640),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 创建推理引擎
engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    preprocess_config=preprocess_config,
    show=True,
    save_dir="output/"
)

# 运行推理
engine.run()
```

### 示例7：服务器模式（高级用法）

```python
import queue
import threading
import numpy as np
from hailo_toolbox.inference import InferenceEngine

# 创建推理引擎
engine = InferenceEngine(
    model="models/yolov8n.hef",
    task_name="yolov8det"
)

# 启动服务器模式
input_queue, output_queue = engine.start_server(
    enable_visualization=True,
    queue_size=30,
    server_timeout=2.0
)

# 处理帧
def process_frames():
    for i in range(10):
        # 创建测试帧
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 发送到推理队列
        frame_info = engine.shm_manager.write(frame, f"frame_{i}")
        input_queue.put(frame_info)
        
        # 获取结果
        try:
            result = output_queue.get(timeout=5.0)
            print(f"处理完成帧 {i}")
        except queue.Empty:
            print(f"帧 {i} 处理超时")

# 运行处理
process_thread = threading.Thread(target=process_frames)
process_thread.start()
process_thread.join()

# 关闭服务器
input_queue.put("SHUTDOWN")
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
hailo-toolbox infer model.hef --source 0 --task-name yolov8det  # 第一个摄像头
hailo-toolbox infer model.hef --source 1 --task-name yolov8det  # 第二个摄像头

# 在Linux下检查摄像头设备
ls /dev/video*
```

### Q4: 模型推理速度慢怎么办？
**A**: 参考[性能优化](#性能优化)部分的建议。

### Q5: 支持自定义回调函数吗？
**A**: 是的，可以通过注册机制实现自定义回调函数，详见开发文档。

### Q6: 如何处理不同的输入源？
**A**: Hailo Toolbox支持多种输入源，包括图片、视频、摄像头和网络流，会自动检测输入类型。

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
engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det",
    # 预处理优化
    preprocess_config={
        "target_size": (640, 640),
        "normalize": True,
        "batch_size": 4  # 批处理
    },
    # 后处理优化
    postprocess_config={
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "max_detections": 100
    },
    show=True
)
```

## 故障排除

### 日志调试
```bash
# 启用详细日志
export HAILO_LOG_LEVEL=DEBUG
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det

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
| `Task name not found` | 回调函数未注册 | 检查task_name是否正确或注册自定义回调 |
| `Source not accessible` | 输入源无法访问 | 检查文件路径、摄像头权限或网络连接 |

### 性能诊断
```python
# 性能监控示例
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
            print(f"处理了 {self.frame_count} 帧，平均FPS: {fps:.2f}")

# 使用监控器
monitor = PerformanceMonitor()
monitor.start()

engine = InferenceEngine(
    model="models/yolov8n.hef",
    source="video.mp4",
    task_name="yolov8det"
)

# 在推理循环中调用monitor.update()
```

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
4. ✅ 使用Python API进行自定义开发
5. ✅ 解决常见问题和优化性能

关键特性总结：
- **模块化架构**: 基于注册机制的可扩展设计
- **多种输入源**: 支持图片、视频、摄像头、网络流
- **灵活的API**: 同时支持命令行和Python API
- **高性能**: 优化的推理引擎和硬件加速支持
- **易于扩展**: 简单的自定义回调函数注册机制

如果遇到问题，请参考故障排除部分或在GitHub上提交Issue。祝您使用愉快！

**更新日期**: 2024年12月
**版本**: v2.0.0




