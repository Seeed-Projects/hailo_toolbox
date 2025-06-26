# Hailo 模型推理指南

本文档将教您如何使用转换后的 Hailo 模型进行 AI 推理。无论您是想检测图片中的物体，还是分析视频内容，这里都有详细的教程。

## 什么是模型推理？

简单来说，模型推理就是让 AI 模型"看"图片或视频，然后告诉您它"看到"了什么。比如：
- 🚗 在图片中找到汽车、行人
- 🐱 识别图片中的猫、狗
- 👤 检测人体姿态和动作
- 🎭 分割图像中的不同区域

## 准备工作

### 1. 确保您有以下文件

- ✅ **转换后的模型文件**（`.hef` 格式）
- ✅ **测试图片或视频**
- ✅ **Hailo Toolbox 已安装**

### 2. 检查安装

```bash
# 验证工具是否正常
hailo-toolbox --version

# 查看帮助信息
hailo-toolbox infer --help
```

## 基础推理教程

### 第一步：最简单的推理

假设您有一个 YOLOv8 目标检测模型，最简单的推理命令是：

```bash
hailo-toolbox infer your_model.hef --source test_image.jpg --task-name yolov8det
```

**解释**:
- `infer`: 启动推理功能
- `your_model.hef`: 您的模型文件
- `--source test_image.jpg`: 要分析的图片
- `--task-name yolov8det`: 告诉系统这是 YOLOv8 检测模型

### 第二步：实时查看结果

添加 `--show` 参数可以实时看到检测结果：

```bash
hailo-toolbox infer your_model.hef \
    --source test_image.jpg \
    --task-name yolov8det \
    --show
```

这样会弹出一个窗口显示检测结果，按任意键关闭。

### 第三步：保存结果

如果想保存检测结果，使用 `--save-dir` 参数：

```bash
hailo-toolbox infer your_model.hef \
    --source test_image.jpg \
    --task-name yolov8det \
    --save-dir ./results
```

结果会保存在 `results` 文件夹中。

## 支持的输入类型

### 1. 图片文件

```bash
# 单张图片
hailo-toolbox infer model.hef --source photo.jpg --task-name yolov8det

# 支持的格式：JPG, PNG, BMP, TIFF, WebP
hailo-toolbox infer model.hef --source image.png --task-name yolov8det
```

### 2. 图片文件夹

```bash
# 批量处理文件夹中的所有图片
hailo-toolbox infer model.hef --source ./images/ --task-name yolov8det --save-dir ./results
```

### 3. 视频文件

```bash
# 视频文件推理
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --show

# 支持的格式：MP4, AVI, MOV, MKV, WebM
hailo-toolbox infer model.hef --source movie.avi --task-name yolov8det
```

### 4. 摄像头实时推理

```bash
# 使用电脑摄像头（设备ID通常是0）
hailo-toolbox infer model.hef --source 0 --task-name yolov8det --show

# 如果有多个摄像头，尝试其他ID
hailo-toolbox infer model.hef --source 1 --task-name yolov8det --show
```

### 5. 网络摄像头

```bash
# IP摄像头（RTSP流）
hailo-toolbox infer model.hef \
    --source "rtsp://username:password@192.168.1.100:554/stream" \
    --task-name yolov8det \
    --show
```

## 支持的任务类型

### 目标检测（找物体）

```bash
# YOLOv8 目标检测
hailo-toolbox infer yolov8_detection.hef \
    --source image.jpg \
    --task-name yolov8det \
    --show
```

**能检测什么**：人、车、动物、日常物品等80种物体

### 实例分割（精确轮廓）

```bash
# YOLOv8 实例分割
hailo-toolbox infer yolov8_segmentation.hef \
    --source image.jpg \
    --task-name yolov8seg \
    --show
```

**能做什么**：不仅找到物体，还能画出精确的轮廓

### 姿态估计（人体关键点）

```bash
# YOLOv8 姿态估计
hailo-toolbox infer yolov8_pose.hef \
    --source image.jpg \
    --task-name yolov8pe \
    --show
```

**能做什么**：检测人体17个关键点，分析人的姿态和动作

## 实际使用示例

### 示例 1：家庭安防监控

```bash
# 使用摄像头检测入侵者
hailo-toolbox infer security_model.hef \
    --source 0 \
    --task-name yolov8det \
    --show \
    --save-dir ./security_logs
```

### 示例 2：交通监控

```bash
# 分析交通视频，检测车辆和行人
hailo-toolbox infer traffic_model.hef \
    --source traffic_video.mp4 \
    --task-name yolov8det \
    --save-dir ./traffic_analysis
```

### 示例 3：批量图片处理

```bash
# 处理文件夹中的所有产品图片
hailo-toolbox infer product_detection.hef \
    --source ./product_photos/ \
    --task-name yolov8det \
    --save-dir ./detection_results
```

### 示例 4：姿态估计分析

```bash
# 分析视频中的人体姿态
hailo-toolbox infer pose_model.hef \
    --source workout_video.mp4 \
    --task-name yolov8pe \
    --show \
    --save-dir ./pose_analysis
```

## 推理参数详解

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `model` | 模型文件路径 | `yolov8n.hef` |
| `--source` | 输入源 | `image.jpg`, `0`, `video.mp4` |

### 重要可选参数

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--task-name` | `yolov8det` | 任务类型 | `yolov8det`, `yolov8seg`, `yolov8pe` |
| `--show` | 不显示 | 实时显示结果 | `--show` |
| `--save-dir` | 不保存 | 结果保存目录 | `--save-dir ./results` |

### 任务类型说明

| 任务名称 | 功能 | 适用模型 | 输出结果 |
|----------|------|----------|----------|
| `yolov8det` | 目标检测 | YOLOv8检测模型 | 边界框+类别+置信度 |
| `yolov8seg` | 实例分割 | YOLOv8分割模型 | 分割掩码+边界框 |
| `yolov8pe` | 姿态估计 | YOLOv8姿态模型 | 人体关键点+骨架 |

## 常见问题解决

### Q1: 提示"找不到模型文件"

**问题**: `FileNotFoundError: model.hef not found`

**解决方法**:
```bash
# 检查文件是否存在
ls -la your_model.hef

# 使用完整路径
hailo-toolbox infer /full/path/to/model.hef --source image.jpg --task-name yolov8det
```

### Q2: 摄像头无法打开

**问题**: `Cannot open camera device 0`

**解决方法**:
```bash
# 尝试不同的设备ID
hailo-toolbox infer model.hef --source 0 --task-name yolov8det  # 第一个摄像头
hailo-toolbox infer model.hef --source 1 --task-name yolov8det  # 第二个摄像头

# 在Linux下检查可用摄像头
ls /dev/video*
```

### Q3: 推理结果不准确

**可能原因和解决方法**:

1. **任务类型不匹配**
```bash
# 确保使用正确的task-name
# 检测模型用 yolov8det
# 分割模型用 yolov8seg  
# 姿态模型用 yolov8pe
```

2. **输入图像质量问题**
- 确保图像清晰、光线充足
- 检查图像尺寸是否合适
- 避免过度模糊或过暗的图像

3. **模型转换问题**
- 重新转换模型，使用更好的校准数据集
- 检查转换时的参数设置

### Q4: 推理速度很慢

**优化建议**:

1. **降低输入分辨率**
```bash
# 如果原图很大，可以先缩放
# 或者使用较小尺寸的模型
```

2. **检查硬件连接**
- 确保 Hailo 设备正确连接
- 检查驱动程序是否正常

3. **减少输出保存**
```bash
# 测试时不保存结果，只显示
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --show
```

### Q5: 显示窗口无法关闭

**解决方法**:
- 点击显示窗口，然后按任意键
- 或者按 `Ctrl+C` 强制退出程序

## 性能优化技巧

### 1. 选择合适的输入源

```bash
# 高质量图像（较慢）
--source high_resolution_image.jpg

# 标准视频（平衡）
--source standard_video.mp4

# 低分辨率流（较快）
--source low_res_stream.mp4
```

### 2. 合理使用显示和保存

```bash
# 只显示，不保存（最快）
--show

# 只保存，不显示（适合批处理）
--save-dir ./results

# 既显示又保存（最慢）
--show --save-dir ./results
```

### 3. 批处理优化

```bash
# 批量处理时，不要实时显示
hailo-toolbox infer model.hef \
    --source ./image_folder/ \
    --task-name yolov8det \
    --save-dir ./batch_results
    # 注意：不加 --show 参数
```

## 理解推理结果

### 目标检测结果

推理完成后，您会看到：
- **边界框**: 用矩形框标出检测到的物体
- **类别标签**: 显示物体的名称（如"person"、"car"）
- **置信度**: 显示检测的可信程度（如0.85表示85%确信）

### 实例分割结果

除了边界框外，还会看到：
- **彩色掩码**: 用不同颜色标出物体的精确轮廓
- **重叠区域**: 可以处理物体相互遮挡的情况

### 姿态估计结果

会显示：
- **关键点**: 人体的17个重要部位（如头部、肩膀、手腕等）
- **骨架连接**: 用线条连接相关的关键点
- **置信度**: 每个关键点的检测可信度

## 进阶使用

### 使用 Python API

如果您熟悉 Python，也可以在代码中使用：

```python
from hailo_toolbox.inference import InferenceEngine

# 创建推理引擎
engine = InferenceEngine(
    model="your_model.hef",
    source="test_image.jpg",
    task_name="yolov8det",
    show=True,
    save_dir="./results"
)

# 运行推理
engine.run()
```

### 自定义模型支持

如果您有自定义模型，可能需要实现对应的后处理函数。详细信息请参考 [开发者文档](DEV.md)。

## 推理流程图

```
[输入源] → [预处理] → [模型推理] → [后处理] → [结果显示/保存]
    ↓         ↓          ↓          ↓           ↓
  图片/视频   尺寸调整    AI计算     结果解析    边界框/掩码
```

## 总结

模型推理的基本步骤：

1. **准备模型文件** (`.hef` 格式)
2. **准备输入数据** (图片、视频或摄像头)
3. **选择正确的任务类型** (`yolov8det`、`yolov8seg`、`yolov8pe`)
4. **运行推理命令**
5. **查看或保存结果**

**记住这个万能命令**:
```bash
hailo-toolbox infer your_model.hef \
    --source your_input \
    --task-name yolov8det \
    --show \
    --save-dir ./results
```

### 常用命令速查

```bash
# 图片检测
hailo-toolbox infer model.hef --source image.jpg --task-name yolov8det --show

# 视频分析
hailo-toolbox infer model.hef --source video.mp4 --task-name yolov8det --save-dir ./results

# 实时摄像头
hailo-toolbox infer model.hef --source 0 --task-name yolov8det --show

# 批量处理
hailo-toolbox infer model.hef --source ./images/ --task-name yolov8det --save-dir ./results
```

现在您已经掌握了 Hailo 模型推理的基本技能！开始享受 AI 带来的便利吧！

---

**相关文档**: 
- [模型转换指南](CONVERT.md) - 学习如何转换模型
- [开发者文档](DEV.md) - 自定义模型开发
- [快速开始](GET_STAR.md) - 完整的安装和使用指南
