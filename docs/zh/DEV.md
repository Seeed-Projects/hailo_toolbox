# Hailo Toolbox 开发者指南

本文档面向开发者，介绍如何为 Hailo Toolbox 实现自定义模型的推理处理模块。

## 概述

Hailo Toolbox 采用模块化架构，通过注册机制管理各个处理模块。要支持新的自定义模型，您需要实现相应的处理模块并注册到系统中。

## 核心模块说明

### 模块分类

| 模块类型 | 是否必需 | 作用 | 实现复杂度 |
|----------|----------|------|------------|
| **PreProcessor** | 🔶 可选 | 图像预处理，转换为模型输入格式 | 简单 |
| **PostProcessor** | ✅ 必需 | 模型输出后处理，解析推理结果 | 中等 |
| **Visualizer** | 🔶 可选 | 结果可视化，在图像上绘制检测框等 | 简单 |
| **CollateInfer** | 🔶 可选 | 推理结果整理，格式化模型原始输出 | 简单 |
| **Source** | ❌ 无需 | 数据源管理，已有通用实现 | - |

### 模块职责详解

#### PreProcessor（预处理器）- 可选实现
- **作用**: 将输入图像转换为模型所需格式
- **输入**: 原始图像 (H, W, C) BGR格式
- **输出**: 预处理后的张量，通常为 (N, C, H, W) 格式
- **说明**: 系统已内置通用预处理器，可通过 `PreprocessConfig` 配置。只有特殊需求时才需要自定义实现。
- **主要任务**:
  - 图像尺寸调整
  - 颜色空间转换 (BGR→RGB)
  - 数据归一化和标准化
  - 维度转换 (HWC→CHW)

#### PostProcessor（后处理器）- 必需实现
- **作用**: 处理模型原始输出，转换为可用的结果
- **输入**: 模型推理输出字典
- **输出**: 结构化的检测/分类结果列表
- **说明**: 每个模型的输出格式不同，必须实现对应的后处理逻辑。
- **主要任务**:
  - 解码模型输出
  - 置信度过滤
  - 非极大值抑制 (NMS)
  - 坐标转换

#### Visualizer（可视化器）- 可选实现
- **作用**: 在图像上绘制推理结果
- **输入**: 原始图像 + 后处理结果
- **输出**: 带有可视化标注的图像
- **主要任务**:
  - 绘制边界框
  - 显示类别标签和置信度
  - 渲染分割掩码或关键点

#### CollateInfer（结果整理）- 可选实现
- **作用**: 整理推理引擎的原始输出
- **输入**: 推理引擎原始输出字典
- **输出**: 格式化后的输出字典
- **主要任务**:
  - 维度调整
  - 数据类型转换
  - 多输出合并

## 注册机制

### 回调类型枚举

```python
from hailo_toolbox.inference.core import CallbackType

class CallbackType(Enum):
    PRE_PROCESSOR = "pre_processor"    # 预处理器
    POST_PROCESSOR = "post_processor"  # 后处理器
    VISUALIZER = "visualizer"          # 可视化器
    COLLATE_INFER = "collate_infer"    # 推理结果整理
    SOURCE = "source"                  # 数据源（通常无需自定义）
```

### 注册方式

```python
from hailo_toolbox.inference.core import CALLBACK_REGISTRY

# 方式1: 装饰器注册（推荐）
@CALLBACK_REGISTRY.registryPreProcessor("my_model")
def my_preprocess(image):
    return processed_image

# 方式2: 多名称注册（一个实现支持多个模型）
@CALLBACK_REGISTRY.registryPostProcessor("model_v1", "model_v2")
class MyPostProcessor:
    def __call__(self, results): pass

# 方式3: 直接注册
CALLBACK_REGISTRY.register_callback("my_model", CallbackType.PRE_PROCESSOR, preprocess_func)
```

## 快速实现示例

以下是一个完整的自定义模型实现示例：

```python
"""
自定义模型实现示例
适用于目标检测类型的模型
"""
from hailo_toolbox.inference.core import InferenceEngine, CALLBACK_REGISTRY
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig
import yaml
import numpy as np
import cv2

# 必须实现
@CALLBACK_REGISTRY.registryPostProcessor("custom")
class CustomPostProcessor:
    def __init__(self, config):
        self.config = config
        self.get_classes()

    def get_classes(self):
        with open("examples/ImageNet.yaml", "r") as f:
            self.classes = yaml.load(f, Loader=yaml.FullLoader)

    def __call__(self, results, original_shape=None):
        class_name = []
        for k, v in results.items():
            class_name.append(self.classes[np.argmax(v)])
        return class_name

# 可选实现
@CALLBACK_REGISTRY.registryVisualizer("custom")
class CustomVisualizer:
    def __init__(self, config):
        self.config = config

    def __call__(self, original_frame, results):

        for v in results:
            cv2.putText(
                original_frame,
                f"CLASS: {v}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        return original_frame


if __name__ == "__main__":
    # 配置输入shape
    preprocess_config = PreprocessConfig(
        target_size=(224, 224),
    )

    engine = InferenceEngine(
        model="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/efficientnet_s.hef",
        source="/home/hk/github/hailo_tools/sources/test640.mp4",
        preprocess_config=preprocess_config,
        task_name="custom",
        show=True,
    )
    engine.run()

```

## 使用自定义模型

实现并注册模块后，就可以使用自定义模型了：

```python
from hailo_toolbox.inference import InferenceEngine

# 创建推理引擎
engine = InferenceEngine(
    model="models/my_custom_model.hef",  # 或 .onnx
    source="test_video.mp4",
    task_name="my_detection_model",      # 与注册时的名称一致
    show=True,
    save_dir="output/"
)

# 运行推理
engine.run()
```

## 最小实现要求

如果您只想快速验证模型，最少只需实现后处理器：

```python
# 最简后处理器
@CALLBACK_REGISTRY.registryPostProcessor("simple_model")
def simple_postprocess(results, original_shape=None):
    # 返回空结果（用于测试）
    return []

# 使用内置预处理器配置
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig

preprocess_config = PreprocessConfig(
    target_size=(640, 640),  # 模型输入尺寸
    normalize=False           # 是否归一化
)

engine = InferenceEngine(
    model="model.hef",
    source="video.mp4", 
    task_name="simple_model",
    preprocess_config=preprocess_config  # 使用内置预处理器
)
engine.run()
```

## 调试技巧

1. **添加日志**: 在关键步骤添加日志输出
```python
import logging
logger = logging.getLogger(__name__)

def __call__(self, image):
    logger.info(f"Input shape: {image.shape}")
    processed = self.process(image)
    logger.info(f"Output shape: {processed.shape}")
    return processed
```

2. **保存中间结果**: 调试时保存预处理后的图像
```python
def __call__(self, image):
    processed = self.process(image)
    # 调试时保存
    if self.debug:
        cv2.imwrite("debug_preprocessed.jpg", processed[0].transpose(1,2,0)*255)
    return processed
```

3. **单步测试**: 先用单张图像测试各个模块
```python
# 测试预处理
preprocessor = MyPreProcessor()
test_image = cv2.imread("test.jpg")
processed = preprocessor(test_image)
print(f"Preprocessed shape: {processed.shape}")
```

## 常见问题

**Q: 如何确定模型的输入输出格式？**
A: 可以使用 ONNX 工具查看模型信息，或参考模型的官方文档。

**Q: 预处理器输出的维度不对怎么办？**
A: 检查模型期望的输入格式，通常为 (N, C, H, W) 或 (N, H, W, C)。

**Q: 后处理器如何处理多输出模型？**
A: 遍历 results 字典中的所有输出，根据每个输出的含义分别处理。

**Q: 可以不实现可视化器吗？**
A: 可以，可视化器是可选的。不实现时系统会使用默认的空实现。

**Q: 可以不实现预处理器吗？**
A: 可以，系统提供了内置的通用预处理器。通过 `PreprocessConfig` 配置即可满足大多数模型的预处理需求。

**Q: 什么时候需要自定义预处理器？**
A: 当模型有特殊的预处理需求时，比如特殊的归一化方式、数据增强、或复杂的输入格式转换。

通过以上指南，您应该能够快速为自定义模型实现必要的处理模块。建议先实现最小功能版本（只需后处理器），验证流程后再逐步完善各个模块的功能。