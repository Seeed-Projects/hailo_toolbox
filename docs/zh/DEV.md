本文档面向开发者，主要介绍如何推理自定义模型以及目前`hailo-toolbox`还未支持的模型。

## 概述

Hailo Toolbox 提供了一个模块化的架构，允许开发者轻松扩展和自定义推理流程。整个推理流程包含以下核心模块：

1. **图像预处理模块 (PreProcessor)** - 负责输入数据的预处理
2. **推理引擎模块 (InferenceEngine)** - 负责模型推理
3. **推理结果整理模块 (CollateInfer)** - 负责推理结果的数据整理
4. **模型后处理模块 (PostProcessor)** - 负责推理结果的后处理
5. **可视化模块 (Visualizer)** - 负责结果的可视化展示

## 架构设计

Hailo Toolbox 采用基于注册器的设计模式，通过 `CALLBACK_REGISTRY` 统一管理所有模块。这种设计具有以下优势：

- **模块化**: 每个模块职责单一，便于开发和维护
- **可扩展性**: 可以轻松添加新的处理模块
- **可复用性**: 模块可以在不同任务间复用
- **解耦性**: 模块间依赖关系清晰，降低耦合度

## 需要实现的模块

在使用模块时需要使用注册器将需要使用的模块进行注册，在调用时只需要声明注册时设置的`name`即可发起调用。支持多种注册方式：

### 装饰器注册方式（推荐）

```python
# 单名称注册
@CALLBACK_REGISTRY.registryPreProcessor("custom")
def pre_process(result):
    pass

# 多名称注册（同一函数可以用多个名称调用）
@CALLBACK_REGISTRY.register(["yolov8det", "yolov8seg"], CallbackType.PRE_PROCESSOR)
def yolo_preprocess_func(data):
    pass
```

### 直接注册方式

```python
# 直接注册单个名称
CALLBACK_REGISTRY.register_callback("custom", CallbackType.PRE_PROCESSOR, pre_process)

# 直接注册多个名称
CALLBACK_REGISTRY.register_callback(["name1", "name2"], CallbackType.POST_PROCESSOR, post_process)
```
### 完整示例代码

```python
"""
Custom Model Inference Example

This example demonstrates how to implement custom processing modules
for a new model type that is not yet supported by hailo-toolbox.

Author: Your Name
Date: 2024-01-01
"""

import numpy as np
import cv2
import logging
from typing import Any, Dict, List, Tuple, Optional, Union

from hailo_toolbox.utils.config import Config
from hailo_toolbox.inference.core import CALLBACK_REGISTRY, InferenceEngine
from hailo_toolbox.process.exceptions import ImageProcessingError, PostProcessingError


# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@CALLBACK_REGISTRY.registryPreProcessor("custom")
def custom_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Custom preprocessing function for input images.
    
    This function handles image preprocessing operations such as:
    - Resizing to model input dimensions
    - Normalization
    - Color space conversion
    - Data type conversion
    
    Args:
        image (np.ndarray): Input image in BGR format with shape (H, W, C)
        
    Returns:
        np.ndarray: Preprocessed image ready for model inference
        
    Raises:
        ImageProcessingError: If preprocessing fails
        
    Note:
        If the HEF model includes preprocessing operations (e.g., normalization),
        this function may not be needed. The model will expect uint8 input.
    """
    try:
        # Validate input image
        if image is None or image.size == 0:
            raise ImageProcessingError("Empty or invalid image provided")
            
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ImageProcessingError(f"Expected 3-channel image, got shape: {image.shape}")
        
        logger.debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")
        
        # Example preprocessing steps
        # 1. Resize to model input size (e.g., 640x640 for YOLO models)
        target_size = (640, 640)  # (width, height)
        processed_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 2. Color space conversion if needed (BGR to RGB)
        # processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # 3. Normalization (if not included in HEF model)
        # processed_image = processed_image.astype(np.float32) / 255.0
        # processed_image = (processed_image - mean) / std
        
        # 4. Ensure correct data type
        # If HEF includes normalization, keep as uint8
        processed_image = processed_image.astype(np.uint8)
        
        logger.debug(f"Processed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")
        
        return processed_image
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ImageProcessingError(f"Preprocessing failed: {str(e)}")


@CALLBACK_REGISTRY.registryPostProcessor("custom")
def custom_postprocess(inference_results: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Custom postprocessing function for inference results.
    
    This function processes the raw model outputs to extract meaningful results.
    Common operations include:
    - Decoding bounding boxes
    - Applying NMS (Non-Maximum Suppression)
    - Converting coordinates
    - Filtering by confidence threshold
    
    Args:
        inference_results (Dict[str, np.ndarray]): Raw inference outputs from the model
            Keys are output layer names, values are numpy arrays
            
    Returns:
        Dict[str, Any]: Processed results containing:
            - detections: List of detected objects
            - scores: Confidence scores
            - classes: Class predictions
            - boxes: Bounding box coordinates
            
    Raises:
        PostProcessingError: If postprocessing fails
    """
    try:
        logger.debug(f"Inference results keys: {list(inference_results.keys())}")
        
        # Example for object detection model
        # Adjust based on your model's output format
        
        # Extract outputs (example names, adjust for your model)
        if 'output_layer1' in inference_results:
            predictions = inference_results['output_layer1']
        else:
            # Use the first available output if specific name not found
            predictions = list(inference_results.values())[0]
            
        logger.debug(f"Predictions shape: {predictions.shape}")
        
        # Example postprocessing for YOLO-like models
        detections = []
        confidence_threshold = 0.5
        nms_threshold = 0.4
        
        # Process predictions (adjust based on your model's output format)
        # This is a generic example - modify for your specific model
        for i in range(predictions.shape[0]):  # Batch dimension
            batch_predictions = predictions[i]
            
            # Extract boxes, scores, and classes
            # Adjust indices based on your model's output format
            if len(batch_predictions.shape) == 2:  # Shape: (num_detections, 5+num_classes)
                boxes = batch_predictions[:, :4]  # x1, y1, x2, y2
                scores = batch_predictions[:, 4]   # Confidence scores
                classes = np.argmax(batch_predictions[:, 5:], axis=1)  # Class predictions
                
                # Filter by confidence threshold
                valid_indices = scores > confidence_threshold
                filtered_boxes = boxes[valid_indices]
                filtered_scores = scores[valid_indices]
                filtered_classes = classes[valid_indices]
                
                # Apply NMS if needed
                if len(filtered_boxes) > 0:
                    nms_indices = cv2.dnn.NMSBoxes(
                        filtered_boxes.tolist(),
                        filtered_scores.tolist(),
                        confidence_threshold,
                        nms_threshold
                    )
                    
                    if len(nms_indices) > 0:
                        nms_indices = nms_indices.flatten()
                        final_boxes = filtered_boxes[nms_indices]
                        final_scores = filtered_scores[nms_indices]
                        final_classes = filtered_classes[nms_indices]
                        
                        # Convert to detection format
                        for j in range(len(final_boxes)):
                            detection = {
                                'bbox': final_boxes[j].tolist(),  # [x1, y1, x2, y2]
                                'score': float(final_scores[j]),
                                'class_id': int(final_classes[j]),
                                'class_name': f'class_{final_classes[j]}'  # Map to actual class names
                            }
                            detections.append(detection)
        
        # Prepare final results
        results = {
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': 0.0,  # Will be filled by inference engine
            'postprocess_time': 0.0,  # Will be filled by inference engine
        }
        
        logger.info(f"Postprocessing completed: {len(detections)} detections found")
        return results
        
    except Exception as e:
        logger.error(f"Postprocessing failed: {str(e)}")
        raise PostProcessingError(f"Postprocessing failed: {str(e)}")


@CALLBACK_REGISTRY.registryCollateInfer("custom")
def custom_collate_inference(raw_outputs: Any) -> Dict[str, np.ndarray]:
    """
    Custom inference result collation function.
    
    This function is called immediately after model inference to organize
    the raw outputs into a standardized format. It's particularly useful
    for models that include NMS in the HEF file, which can produce
    variable-length outputs.
    
    Args:
        raw_outputs: Raw outputs from the inference engine
        
    Returns:
        Dict[str, np.ndarray]: Organized inference results
        
    Note:
        This function is optional and only needed for models with
        non-standard output formats or variable-length outputs.
    """
    try:
        logger.debug("Collating inference results")
        
        # If raw_outputs is already in the correct format, return as is
        if isinstance(raw_outputs, dict):
            return raw_outputs
            
        # Convert to dictionary format if needed
        # Adjust based on your model's output structure
        if isinstance(raw_outputs, (list, tuple)):
            collated_results = {}
            for i, output in enumerate(raw_outputs):
                collated_results[f'output_{i}'] = output
            return collated_results
        elif isinstance(raw_outputs, np.ndarray):
            return {'output_0': raw_outputs}
        else:
            logger.warning(f"Unexpected output type: {type(raw_outputs)}")
            return {'output_0': raw_outputs}
            
    except Exception as e:
        logger.error(f"Collation failed: {str(e)}")
        return {'output_0': raw_outputs}  # Fallback


@CALLBACK_REGISTRY.registryVisualizer("custom")
def custom_visualize(image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
    """
    Custom visualization function for inference results.
    
    This function draws the inference results on the input image for
    visual inspection and debugging purposes.
    
    Args:
        image (np.ndarray): Original input image
        results (Dict[str, Any]): Processed inference results
        
    Returns:
        np.ndarray: Annotated image with visualizations
    """
    try:
        # Create a copy to avoid modifying the original image
        vis_image = image.copy()
        
        # Extract detections from results
        detections = results.get('detections', [])
        
        # Define colors for different classes
        colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw each detection
        for detection in detections:
            bbox = detection['bbox']
            score = detection['score']
            class_id = detection['class_id']
            class_name = detection.get('class_name', f'class_{class_id}')
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_name}: {score:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add summary information
        summary = f'Detections: {len(detections)}'
        cv2.putText(vis_image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        logger.debug(f"Visualization completed for {len(detections)} detections")
        return vis_image
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return image  # Return original image if visualization fails


def create_optimized_config() -> Config:
    """
    Create an optimized configuration for custom model inference.
    
    Returns:
        Config: Configured Config object with optimized settings
    """
    # Initialize configuration with empty dict to avoid errors
    config = Config({})
    
    # Core configuration
    config.callback = "custom"  # Use our custom registered modules
    config.model = "models/custom_model.hef"  # Path to HEF model file
    config.source = "test_video.mp4"  # Input source (video file, camera, or image)
    
    # Output configuration
    config.output = "output_results.mp4"  # Output file path
    config.save = True  # Save results to file
    config.show = True  # Display real-time results
    
    # Performance configuration
    config.task_type = "detection"  # Task type for optimization
    
    return config


def main():
    """
    Main function demonstrating custom model inference usage.
    
    This function shows how to:
    1. Configure the inference engine
    2. Run inference with custom modules
    3. Handle errors and logging
    """
    try:
        logger.info("Starting custom model inference")
        
        # Create optimized configuration
        config = create_optimized_config()
        
        # Log configuration for debugging
        logger.info("Configuration:")
        logger.info(str(config))
        
        # Initialize inference engine with custom callback
        engine = InferenceEngine(config, "custom")
        
        # Run inference
        logger.info("Starting inference process...")
        engine.run()
        
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
```

## 模块详细说明

### 1. 图像预处理模块 (PreProcessor)

#### 何时需要实现
- 模型需要特殊的输入格式转换
- HEF模型**未包含**预处理操作（如归一化）
- 需要自定义的图像变换操作

#### 实现要点
```python
@CALLBACK_REGISTRY.registryPreProcessor("your_model_name")
def your_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Custom preprocessing function.
    
    Args:
        image: Input image in BGR format, shape (H, W, C), dtype uint8
        
    Returns:
        Preprocessed image ready for model input
    """
    # 1. Image validation
    if image is None or image.size == 0:
        raise ImageProcessingError("Invalid image input")
    
    # 2. Resize to model input size
    target_size = (640, 640)  # Adjust to your model's input size
    processed = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 3. Normalization (only if NOT included in HEF)
    if not hef_includes_normalization:
        processed = processed.astype(np.float32) / 255.0
        # Apply mean and std if needed
        # processed = (processed - mean) / std
    
    # 4. Data type conversion
    return processed.astype(np.uint8 if hef_includes_normalization else np.float32)
```

#### 内置变换工具
Hailo Toolbox 提供了丰富的预处理变换工具在 `hailo_toolbox/process/preprocessor/transforms.py` 中：

- `ResizeTransform`: 图像尺寸调整
- `NormalizationTransform`: 归一化处理
- `DataTypeTransform`: 数据类型转换
- `PaddingTransform`: 填充操作
- `CropTransform`: 裁剪操作

#### 最佳实践
1. **错误处理**: 总是验证输入图像的有效性
2. **性能优化**: 使用 OpenCV 的优化函数
3. **内存管理**: 避免不必要的内存拷贝
4. **日志记录**: 记录预处理的关键信息用于调试

### 2. 推理结果整理模块 (CollateInfer)

#### 何时需要实现
- 模型包含NMS操作，输出长度可变
- 模型有多个输出需要重组
- 原始输出格式需要标准化

#### 实现要点
```python
@CALLBACK_REGISTRY.registryCollateInfer("your_model_name")
def your_collate(raw_outputs: Any) -> Dict[str, np.ndarray]:
    """
    Collate raw inference outputs into standardized format.
    
    Args:
        raw_outputs: Raw outputs from inference engine
        
    Returns:
        Dictionary with organized outputs
    """
    # Handle different output formats
    if isinstance(raw_outputs, dict):
        return raw_outputs  # Already in correct format
    elif isinstance(raw_outputs, (list, tuple)):
        # Multiple outputs case
        return {f'output_{i}': output for i, output in enumerate(raw_outputs)}
    elif isinstance(raw_outputs, np.ndarray):
        # Single output case
        return {'predictions': raw_outputs}
    else:
        # Fallback
        return {'output': raw_outputs}
```

#### 常见场景
1. **NMS输出处理**: 变长检测结果的标准化
2. **多尺度输出**: 不同分辨率特征图的整合
3. **分类+回归**: 分类和回归分支的分离

### 3. 模型后处理模块 (PostProcessor)

#### 实现要点
后处理模块是最复杂的部分，需要根据具体算法实现：

```python
@CALLBACK_REGISTRY.registryPostProcessor("your_model_name")
def your_postprocess(inference_results: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Process inference results to extract meaningful information.
    
    Args:
        inference_results: Raw inference outputs
        
    Returns:
        Processed results with detections, scores, etc.
    """
    # Extract model outputs
    predictions = inference_results['predictions']
    
    # Apply thresholding
    confidence_threshold = 0.5
    valid_detections = predictions[predictions[:, 4] > confidence_threshold]
    
    # Apply NMS (if not in model)
    if not model_has_nms:
        nms_indices = apply_nms(valid_detections, nms_threshold=0.4)
        final_detections = valid_detections[nms_indices]
    else:
        final_detections = valid_detections
    
    # Convert to standard format
    results = format_detections(final_detections)
    
    return results
```

#### 不同任务的后处理

**目标检测**:
- 边界框解码
- 置信度过滤
- NMS应用
- 坐标系转换

**图像分类**:
- Softmax应用
- Top-K结果提取
- 类别名称映射

**语义分割**:
- 像素级预测解码
- 类别掩码生成
- 后处理滤波

**实例分割**:
- 掩码解码
- 实例分离
- 轮廓提取

### 4. 可视化模块 (Visualizer)

#### 实现要点
```python
@CALLBACK_REGISTRY.registryVisualizer("your_model_name")
def your_visualize(image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
    """
    Visualize inference results on the input image.
    
    Args:
        image: Original input image
        results: Processed inference results
        
    Returns:
        Image with visualizations
    """
    vis_image = image.copy()
    
    # Draw detections
    for detection in results.get('detections', []):
        # Draw bounding box
        draw_bbox(vis_image, detection['bbox'], detection['class_name'], detection['score'])
    
    # Add summary information
    add_summary_text(vis_image, results)
    
    return vis_image
```

#### 可视化元素
- **边界框**: 不同类别使用不同颜色
- **标签**: 类别名称和置信度
- **掩码**: 半透明覆盖层
- **关键点**: 特殊标记点
- **统计信息**: 检测数量、处理时间等

## 配置参数详解

### Config 类主要参数

```python
config = Config({})

# 核心配置
config.model = "path/to/model.hef"      # HEF模型文件路径
config.callback = "your_callback_name"  # 注册的回调名称
config.source = "input.mp4"            # 输入源（视频/图片/摄像头）

# 输出配置
config.output = "output.mp4"           # 输出文件路径
config.save = True                     # 是否保存结果
config.show = True                     # 是否实时显示

# 任务配置
config.task_type = "detection"         # 任务类型
config.preprocess = "custom_preprocess" # 自定义预处理
config.postprocess = "custom_postprocess" # 自定义后处理
config.visualization = "custom_visualize" # 自定义可视化
```

### 输入源配置

```python
# 视频文件
config.source = "video.mp4"

# 图片文件
config.source = "image.jpg"

# 图片文件夹
config.source = "images/"

# 摄像头 (设备ID)
config.source = 0

# RTSP流
config.source = "rtsp://camera_ip:port/stream"

# 图片列表
config.source = ["img1.jpg", "img2.jpg", "img3.jpg"]
```

## 性能优化建议

### 1. 预处理优化
- 使用 OpenCV 的优化函数
- 避免不必要的数据类型转换
- 批量处理多张图片
- 使用 GPU 加速（如果可用）

### 2. 后处理优化
- 向量化操作替代循环
- 使用 NumPy 的高效函数
- 预分配内存空间
- 避免频繁的内存分配

### 3. 可视化优化
- 只在需要时进行可视化
- 使用简单的绘制操作
- 避免复杂的文本渲染
- 考虑降低可视化分辨率

### 4. 内存管理
```python
# 良好的内存管理示例
def optimized_postprocess(inference_results):
    # 预分配数组
    max_detections = 1000
    detections = np.zeros((max_detections, 6), dtype=np.float32)
    
    # 复用变量
    valid_count = 0
    
    # 避免中间变量
    predictions = inference_results['predictions']
    valid_indices = predictions[:, 4] > 0.5
    
    # 直接操作
    detections[:valid_count] = predictions[valid_indices]
    
    return format_results(detections[:valid_count])
```

## 错误处理和调试

### 1. 异常处理模式
```python
from hailo_toolbox.process.exceptions import ImageProcessingError, PostProcessingError

def safe_preprocess(image):
    try:
        # 预处理逻辑
        return processed_image
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ImageProcessingError(f"Preprocessing failed: {str(e)}")
```

### 2. 调试技巧
```python
# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 保存中间结果
def debug_postprocess(inference_results):
    # 保存原始输出
    np.save('debug_raw_output.npy', inference_results['predictions'])
    
    # 处理逻辑
    processed = process_predictions(inference_results)
    
    # 保存处理后结果
    with open('debug_processed.json', 'w') as f:
        json.dump(processed, f, indent=2)
    
    return processed
```

### 3. 常见问题诊断

#### 问题1: 预处理后图像异常
```python
# 检查点1: 图像尺寸和数据类型
print(f"Image shape: {image.shape}, dtype: {image.dtype}")
print(f"Value range: [{image.min()}, {image.max()}]")

# 检查点2: 归一化参数
if needs_normalization:
    print(f"Mean: {mean}, Std: {std}")
    normalized = (image - mean) / std
    print(f"Normalized range: [{normalized.min()}, {normalized.max()}]")
```

#### 问题2: 后处理结果异常
```python
# 检查推理输出
print(f"Inference output keys: {list(inference_results.keys())}")
for key, value in inference_results.items():
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    print(f"  Value range: [{value.min()}, {value.max()}]")

# 检查检测结果
detections = extract_detections(inference_results)
print(f"Number of detections: {len(detections)}")
if detections:
    print(f"Score range: [{min(d['score'] for d in detections)}, {max(d['score'] for d in detections)}]")
```

#### 问题3: 性能问题
```python
import time

def profile_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@profile_function
def your_postprocess(inference_results):
    # 你的后处理逻辑
    pass
```

## 单元测试

### 测试框架设置
```python
import unittest
import numpy as np
from unittest.mock import Mock, patch

class TestCustomProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.sample_inference_results = {
            'predictions': np.random.rand(1, 100, 85).astype(np.float32)
        }
    
    def test_preprocess_valid_input(self):
        """Test preprocessing with valid input."""
        result = custom_preprocess(self.sample_image)
        
        # Check output shape
        self.assertEqual(result.shape, (640, 640, 3))
        
        # Check data type
        self.assertEqual(result.dtype, np.uint8)
        
        # Check value range
        self.assertTrue(0 <= result.min() <= result.max() <= 255)
    
    def test_preprocess_invalid_input(self):
        """Test preprocessing with invalid input."""
        with self.assertRaises(ImageProcessingError):
            custom_preprocess(None)
        
        with self.assertRaises(ImageProcessingError):
            custom_preprocess(np.array([]))
    
    def test_postprocess_valid_input(self):
        """Test postprocessing with valid input."""
        result = custom_postprocess(self.sample_inference_results)
        
        # Check result structure
        self.assertIn('detections', result)
        self.assertIn('num_detections', result)
        
        # Check detection format
        if result['detections']:
            detection = result['detections'][0]
            self.assertIn('bbox', detection)
            self.assertIn('score', detection)
            self.assertIn('class_id', detection)
    
    def test_visualization_output(self):
        """Test visualization output."""
        mock_results = {
            'detections': [
                {
                    'bbox': [100, 100, 200, 200],
                    'score': 0.9,
                    'class_id': 0,
                    'class_name': 'test_class'
                }
            ]
        }
        
        result = custom_visualize(self.sample_image, mock_results)
        
        # Check that image is returned
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.sample_image.shape)

if __name__ == '__main__':
    unittest.main()
```

### 集成测试
```python
def test_end_to_end_inference():
    """Test complete inference pipeline."""
    # Setup
    config = create_optimized_config()
    config.source = "test_image.jpg"  # Use a known test image
    
    # Mock the model to avoid requiring actual HEF file
    with patch('hailo_toolbox.inference.hailo_engine.HailoInference') as mock_engine:
        mock_engine.return_value.infer.return_value = {
            'predictions': np.random.rand(1, 100, 85).astype(np.float32)
        }
        
        # Run inference
        engine = InferenceEngine(config, "custom")
        results = engine.run()
        
        # Verify results
        assert results is not None
        assert 'detections' in results
```

## 部署和生产环境

### 1. 容器化部署
```dockerfile
FROM hailo/hailo-ai-dev:latest

# Copy your custom modules
COPY custom_inference.py /app/
COPY models/ /app/models/

# Install additional dependencies
RUN pip install -r requirements.txt

# Set working directory
WORKDIR /app

# Run inference
CMD ["python", "custom_inference.py"]
```

### 2. 生产环境配置
```python
# production_config.py
import os
from hailo_toolbox.utils.config import Config

def create_production_config():
    config = Config({})
    
    # Use environment variables for configuration
    config.model = os.getenv('MODEL_PATH', 'models/default.hef')
    config.source = os.getenv('INPUT_SOURCE', 'camera')
    config.callback = os.getenv('CALLBACK_NAME', 'custom')
    
    # Performance settings
    config.batch_size = int(os.getenv('BATCH_SIZE', '1'))
    config.num_threads = int(os.getenv('NUM_THREADS', '4'))
    
    # Logging settings
    config.log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    return config
```

### 3. 监控和日志
```python
import logging
import time
from functools import wraps

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.info(f"{func.__name__} completed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{func.__name__} failed after {execution_time:.4f}s: {str(e)}")
            raise
    return wrapper

# Apply monitoring to your functions
@monitor_performance
@CALLBACK_REGISTRY.registryPostProcessor("monitored_custom")
def monitored_postprocess(inference_results):
    return custom_postprocess(inference_results)
```

## 总结

本文档提供了在 Hailo Toolbox 中实现自定义模型推理的完整指南。通过模块化的设计，开发者可以：

1. **灵活扩展**: easily 添加新的模型支持
2. **代码复用**: 在不同项目间共享处理模块
3. **性能优化**: 针对特定需求优化处理流程
4. **易于维护**: 清晰的模块分离便于调试和维护

遵循本文档的最佳实践，可以构建高效、可靠的AI推理应用。
