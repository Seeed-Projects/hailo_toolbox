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

- ✅ **Hailo Toolbox 已安装**
- ✅ **Python 环境准备好**
- ✅ **测试图片或视频**（或使用摄像头）

### 2. 检查安装

```python
# 在Python中验证安装
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox import create_source
print("Hailo Toolbox 已正确安装！")
```

## 基础推理教程

### 第一步：理解基本结构

所有的推理都遵循相同的模式：

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

# 1. 创建输入源
source = create_source("your_input_source")

# 2. 加载模型
model = ModelsZoo.task_type.model_name()

# 3. 处理每一帧
for img in source:
    results = model.predict(img)
    for result in results:
        # 4. 处理结果
        print("处理结果...")
```

### 第二步：选择输入源

```python
# 图片文件
source = create_source("test_image.jpg")

# 视频文件
source = create_source("video.mp4")

# 摄像头（设备ID通常是0）
source = create_source(0)

# 网络摄像头
source = create_source("rtsp://username:password@192.168.1.100:554/stream")

# 图片文件夹
source = create_source("./images/")

# 网络视频
source = create_source("https://example.com/video.mp4")
```

## 支持的任务类型和示例

### 1. 目标检测（找物体）

**示例文件**: `examples/Hailo_Object_Detection.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import DetectionVisualization
import cv2

if __name__ == "__main__":
    # 创建输入源
    source = create_source("test_video.mp4")  # 或使用 0 表示摄像头
    
    # 加载YOLOv8检测模型
    inference = ModelsZoo.detection.yolov8s()
    visualization = DetectionVisualization()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # 可视化结果
            img = visualization.visualize(img, result)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)
            
            # 获取检测结果
            boxes = result.get_boxes()      # 边界框
            scores = result.get_scores()    # 置信度
            class_ids = result.get_class_ids()  # 类别ID
            
            print(f"检测到 {len(result)} 个物体")
            # 显示前5个检测结果
            for i in range(min(5, len(result))):
                print(f"  物体{i}: 边界框{boxes[i]}, 置信度{scores[i]:.3f}, 类别{class_ids[i]}")
```

**能检测什么**：人、车、动物、日常物品等80种物体

**可用模型**：
- `ModelsZoo.detection.yolov8n()` - 最快速度
- `ModelsZoo.detection.yolov8s()` - 平衡速度和精度
- `ModelsZoo.detection.yolov8m()` - 更高精度
- `ModelsZoo.detection.yolov8l()` - 高精度
- `ModelsZoo.detection.yolov8x()` - 最高精度

### 2. 实例分割（精确轮廓）

**示例文件**: `examples/Hailo_Instance_Segmentation.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import SegmentationVisualization
import cv2

if __name__ == "__main__":
    source = create_source("test_video.mp4")
    
    # 加载YOLOv8分割模型
    inference = ModelsZoo.segmentation.yolov8s_seg()
    visualization = SegmentationVisualization()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # 可视化分割结果
            img = visualization.visualize(img, result)
            cv2.imshow("Segmentation", img)
            cv2.waitKey(1)
            
            # 获取分割结果
            if hasattr(result, "masks") and result.masks is not None:
                print(f"分割掩码形状: {result.masks.shape}")
            
            boxes = result.get_boxes_xyxy()  # 边界框
            scores = result.get_scores()     # 置信度
            class_ids = result.get_class_ids()  # 类别ID
```

**能做什么**：不仅找到物体，还能画出精确的轮廓

**可用模型**：
- `ModelsZoo.segmentation.yolov8n_seg()` - 快速分割
- `ModelsZoo.segmentation.yolov8s_seg()` - 标准分割
- `ModelsZoo.segmentation.yolov8m_seg()` - 高精度分割

### 3. 姿态估计（人体关键点）

**示例文件**: `examples/Hailo_Pose_Estimation.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import KeypointVisualization
import cv2

if __name__ == "__main__":
    source = create_source("test_video.mp4")
    
    # 加载YOLOv8姿态估计模型
    inference = ModelsZoo.pose_estimation.yolov8s_pose()
    visualization = KeypointVisualization()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # 可视化姿态结果
            img = visualization.visualize(img, result)
            cv2.imshow("Pose Estimation", img)
            cv2.waitKey(1)
            
            print(f"检测到 {len(result)} 个人")
            # 显示前3个人的姿态信息
            for i, person in enumerate(result):
                if i >= 3:  # 只显示前3个
                    break
                keypoints = person.get_keypoints()  # 关键点坐标
                score = person.get_score()          # 人体置信度
                boxes = person.get_boxes()          # 边界框
                
                print(f"  人{i}: {len(keypoints)}个关键点, 置信度{score[0]:.3f}")
```

**能做什么**：检测人体17个关键点，分析人的姿态和动作

**可用模型**：
- `ModelsZoo.pose_estimation.yolov8s_pose()` - 标准姿态估计
- `ModelsZoo.pose_estimation.yolov8m_pose()` - 高精度姿态估计

### 4. 图像分类（识别主要物体）

**示例文件**: `examples/Hailo_Classification.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

if __name__ == "__main__":
    source = create_source("test_image.jpg")
    
    # 加载分类模型
    inference = ModelsZoo.classification.resnet18()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # 获取分类结果
            class_name = result.get_class_name()        # 最可能的类别
            confidence = result.get_score()             # 置信度
            top5_names = result.get_top_5_class_names() # 前5个类别
            top5_scores = result.get_top_5_scores()     # 前5个分数
            
            print(f"分类结果: {class_name} (置信度: {confidence:.3f})")
            print(f"前5个类别: {top5_names}")
            print(f"前5个分数: {[f'{score:.3f}' for score in top5_scores]}")
```

**可用模型**：
- `ModelsZoo.classification.mobilenetv1()` - 轻量级分类
- `ModelsZoo.classification.resnet18()` - 经典分类模型

### 5. 人脸检测

**示例文件**: `examples/Hailo_Face_Detection.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

def visualize_face_detection(img, boxes, scores, landmarks):
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        # 绘制人脸框
        cv2.rectangle(img, (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    source = create_source("test_video.mp4")
    
    # 加载人脸检测模型
    inference = ModelsZoo.face_detection.scrfd_10g()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            print(f"检测到 {len(result)} 张人脸")
            
            boxes = result.get_boxes(pixel_coords=True)      # 人脸框
            scores = result.get_scores()                     # 置信度
            landmarks = result.get_landmarks(pixel_coords=True)  # 面部关键点
            
            img = visualize_face_detection(img, boxes, scores, landmarks)
            cv2.imshow("Face Detection", img)
            cv2.waitKey(1)
```

**可用模型**：
- `ModelsZoo.face_detection.scrfd_10g()` - 高精度人脸检测
- `ModelsZoo.face_detection.scrfd_2_5g()` - 平衡性能
- `ModelsZoo.face_detection.scrfd_500m()` - 快速检测
- `ModelsZoo.face_detection.retinaface_mbnet()` - 轻量级检测

### 6. 深度估计

**示例文件**: `examples/Hailo_Depth_Estimation.py`

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

if __name__ == "__main__":
    source = create_source("test_video.mp4")
    
    # 加载深度估计模型
    inference = ModelsZoo.depth_estimation.fast_depth()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            depth_map = result.get_depth()                    # 原始深度图
            depth_normalized = result.get_depth_normalized()  # 归一化深度图
            original_shape = result.get_original_shape()      # 原始图像尺寸
            
            cv2.imshow("Depth Estimation", depth_normalized)
            cv2.waitKey(1)
            
            print(f"深度图形状: {depth_map.shape}")
            print(f"深度值范围: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
            print(f"原始图像尺寸: {original_shape}")
```

**可用模型**：
- `ModelsZoo.depth_estimation.fast_depth()` - 快速深度估计
- `ModelsZoo.depth_estimation.scdepthv3()` - 高精度深度估计

## 实际使用示例

### 示例 1：家庭安防监控

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import DetectionVisualization
import cv2
import datetime

def security_monitoring():
    # 使用摄像头
    source = create_source(0)
    inference = ModelsZoo.detection.yolov8s()
    visualization = DetectionVisualization()
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            # 检查是否有人
            class_ids = result.get_class_ids()
            if 0 in class_ids:  # 0 表示人
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"security_alert_{timestamp}.jpg", img)
                print(f"安全警报！检测到入侵者 - {timestamp}")
            
            img = visualization.visualize(img, result)
            cv2.imshow("Security Monitor", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    security_monitoring()
```

### 示例 2：交通监控分析

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

def traffic_analysis():
    source = create_source("traffic_video.mp4")
    inference = ModelsZoo.detection.yolov8m()
    
    vehicle_classes = [2, 3, 5, 7]  # 汽车、摩托车、公交车、卡车
    
    for img in source:
        results = inference.predict(img)
        for result in results:
            class_ids = result.get_class_ids()
            boxes = result.get_boxes()
            
            vehicle_count = sum(1 for class_id in class_ids if class_id in vehicle_classes)
            person_count = sum(1 for class_id in class_ids if class_id == 0)
            
            print(f"车辆数量: {vehicle_count}, 行人数量: {person_count}")
            
            # 可视化
            for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
                if class_id in vehicle_classes or class_id == 0:
                    color = (0, 255, 0) if class_id in vehicle_classes else (255, 0, 0)
                    cv2.rectangle(img, (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), color, 2)
            
            cv2.imshow("Traffic Analysis", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    traffic_analysis()
```

### 示例 3：批量图片处理

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2
import os

def batch_image_processing():
    # 处理文件夹中的所有图片
    source = create_source("./product_photos/")
    inference = ModelsZoo.detection.yolov8n()
    
    os.makedirs("./detection_results", exist_ok=True)
    
    for i, img in enumerate(source):
        results = inference.predict(img)
        for result in results:
            boxes = result.get_boxes()
            scores = result.get_scores()
            class_ids = result.get_class_ids()
            
            # 在图像上绘制检测结果
            for box, score, class_id in zip(boxes, scores, class_ids):
                cv2.rectangle(img, (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(img, f"Class{class_id}: {score:.2f}", 
                          (int(box[0]), int(box[1])-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存结果
            cv2.imwrite(f"./detection_results/result_{i:04d}.jpg", img)
            print(f"处理完成: 图片 {i}, 检测到 {len(result)} 个物体")

if __name__ == "__main__":
    batch_image_processing()
```

## 常见问题解决

### Q1: 提示"找不到模型文件"

**问题**: 模型下载失败或网络问题

**解决方法**:
```python
# 检查网络连接
import requests
try:
    response = requests.get("https://www.google.com", timeout=5)
    print("网络连接正常")
except:
    print("网络连接有问题，请检查网络设置")

# 手动下载模型（如果自动下载失败）
from hailo_toolbox.models import ModelsZoo
model = ModelsZoo.detection.yolov8n()  # 这会尝试下载模型
```

### Q2: 摄像头无法打开

**问题**: `Cannot open camera device 0`

**解决方法**:
```python
import cv2

# 测试不同的摄像头ID
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"摄像头 {i} 可用")
        cap.release()
    else:
        print(f"摄像头 {i} 不可用")

# 使用可用的摄像头ID
source = create_source(0)  # 或者使用找到的可用ID
```

### Q3: 推理结果不准确

**可能原因和解决方法**:

1. **输入图像质量问题**
```python
import cv2

# 检查图像质量
def check_image_quality(img):
    if img is None:
        print("图像为空")
        return False
    
    height, width = img.shape[:2]
    if height < 100 or width < 100:
        print(f"图像太小: {width}x{height}")
        return False
    
    # 检查亮度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < 30:
        print(f"图像太暗: 亮度 {brightness}")
    elif brightness > 200:
        print(f"图像太亮: 亮度 {brightness}")
    
    return True
```

2. **选择合适的模型**
```python
# 根据需求选择模型
# 速度优先：yolov8n
# 平衡：yolov8s
# 精度优先：yolov8m, yolov8l, yolov8x

inference = ModelsZoo.detection.yolov8s()  # 推荐的平衡选择
```

### Q4: 推理速度很慢

**优化建议**:

1. **使用更小的模型**
```python
# 使用最快的模型
inference = ModelsZoo.detection.yolov8n()  # 而不是 yolov8x
```

2. **降低输入分辨率**
```python
import cv2

def resize_frame(img, max_size=640):
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
    return img

# 在推理前调整图像大小
for img in source:
    img = resize_frame(img)
    results = inference.predict(img)
    # ...
```

3. **跳帧处理**
```python
frame_skip = 2  # 每2帧处理一次
frame_count = 0

for img in source:
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    results = inference.predict(img)
    # 处理结果...
```

## 性能优化技巧

### 1. 选择合适的模型

```python
# 根据应用场景选择模型
# 实时应用：选择较小的模型
inference = ModelsZoo.detection.yolov8n()

# 离线分析：可以选择更大的模型
inference = ModelsZoo.detection.yolov8x()
```

### 2. 批处理优化

```python
# 对于图片文件夹，不需要实时显示
source = create_source("./images/")
inference = ModelsZoo.detection.yolov8s()

for i, img in enumerate(source):
    results = inference.predict(img)
    # 只处理结果，不显示
    for result in results:
        # 保存或记录结果
        print(f"图片 {i}: 检测到 {len(result)} 个物体")
```

### 3. 内存管理

```python
import gc

# 定期清理内存
frame_count = 0
for img in source:
    results = inference.predict(img)
    # 处理结果...
    
    frame_count += 1
    if frame_count % 100 == 0:
        gc.collect()  # 每100帧清理一次内存
```

## 理解推理结果

### 目标检测结果

```python
for result in results:
    boxes = result.get_boxes()          # 边界框 [x1, y1, x2, y2]
    scores = result.get_scores()        # 置信度分数 [0.0-1.0]
    class_ids = result.get_class_ids()  # 类别ID [0-79 for COCO]
    
    print(f"检测到 {len(result)} 个物体")
    for i in range(len(result)):
        print(f"物体 {i}: 类别{class_ids[i]}, 置信度{scores[i]:.3f}")
```

### 分割结果

```python
for result in results:
    if hasattr(result, "masks") and result.masks is not None:
        masks = result.masks            # 分割掩码
        print(f"掩码形状: {masks.shape}")
    
    boxes = result.get_boxes_xyxy()     # 边界框
    scores = result.get_scores()        # 置信度
```

### 姿态估计结果

```python
for result in results:
    for person in result:
        keypoints = person.get_keypoints()  # 17个关键点坐标
        score = person.get_score()          # 人体检测置信度
        boxes = person.get_boxes()          # 人体边界框
        
        print(f"关键点数量: {len(keypoints)}")
        print(f"人体置信度: {score}")
```

## 总结

使用 Hailo Toolbox 进行模型推理的基本步骤：

1. **创建输入源** - 使用 `create_source()` 函数
2. **加载模型** - 从 `ModelsZoo` 选择合适的模型
3. **处理数据** - 遍历输入源的每一帧
4. **获取结果** - 调用模型的 `predict()` 方法
5. **处理输出** - 使用结果对象的各种方法获取数据

### 常用代码模板

```python
from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

# 基础模板
def basic_inference():
    source = create_source("your_input")
    model = ModelsZoo.task_type.model_name()
    
    for img in source:
        results = model.predict(img)
        for result in results:
            # 处理结果
            print("推理完成")

# 带可视化的模板
def inference_with_visualization():
    source = create_source("your_input")
    model = ModelsZoo.detection.yolov8s()
    
    for img in source:
        results = model.predict(img)
        for result in results:
            # 绘制结果
            boxes = result.get_boxes()
            for box in boxes:
                cv2.rectangle(img, (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), (0, 255, 0), 2)
            
            cv2.imshow("Results", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    basic_inference()
```

现在您已经掌握了 Hailo 模型推理的完整技能！参考 `examples/` 文件夹中的具体示例开始您的 AI 之旅吧！

---

**相关文档**: 
- [模型转换指南](CONVERT.md) - 学习如何转换模型
- [开发者文档](DEV.md) - 自定义模型开发
- [快速开始](GET_STAR.md) - 完整的安装和使用指南
- [示例代码](../examples/) - 完整的推理示例
