# Hailo Toolbox Developer Guide

This document is for developers who want to implement custom model inference processing modules for Hailo Toolbox.

## Overview

Hailo Toolbox adopts a modular architecture and manages various processing modules through a registration mechanism. To support new custom models, you need to implement corresponding processing modules and register them in the system.

## Core Module Description

### Module Classification

| Module Type | Required | Function | Implementation Complexity |
|-------------|----------|----------|---------------------------|
| **PreProcessor** | 🔶 Optional | Image preprocessing, convert to model input format | Simple |
| **PostProcessor** | ✅ Required | Model output postprocessing, parse inference results | Medium |
| **Visualizer** | 🔶 Optional | Result visualization, draw detection boxes etc. | Simple |
| **CollateInfer** | 🔶 Optional | Inference result organization, format model raw output | Simple |
| **Source** | ❌ Not needed | Data source management, generic implementation available | - |

### Module Responsibility Details

#### PreProcessor (Preprocessor) - Optional Implementation
- **Function**: Convert input images to model required format
- **Input**: Original image (H, W, C) BGR format
- **Output**: Preprocessed tensor, usually (N, C, H, W) format
- **Note**: System provides built-in generic preprocessor, configurable via `PreprocessConfig`. Only needed for special requirements.
- **Main Tasks**:
  - Image size adjustment
  - Color space conversion (BGR→RGB)
  - Data normalization and standardization
  - Dimension conversion (HWC→CHW)

#### PostProcessor (Postprocessor) - Required Implementation
- **Function**: Process model raw output, convert to usable results
- **Input**: Model inference output dictionary
- **Output**: Structured detection/classification result list
- **Note**: Each model has different output formats, must implement corresponding postprocessing logic.
- **Main Tasks**:
  - Decode model output
  - Confidence filtering
  - Non-maximum suppression (NMS)
  - Coordinate transformation

#### Visualizer (Visualizer) - Optional Implementation
- **Function**: Draw inference results on images
- **Input**: Original image + postprocessing results
- **Output**: Image with visualization annotations
- **Main Tasks**:
  - Draw bounding boxes
  - Display class labels and confidence
  - Render segmentation masks or keypoints

#### CollateInfer (Result Organization) - Optional Implementation
- **Function**: Organize inference engine raw output
- **Input**: Inference engine raw output dictionary
- **Output**: Formatted output dictionary
- **Main Tasks**:
  - Dimension adjustment
  - Data type conversion
  - Multi-output merging

## Registration Mechanism

### Callback Type Enumeration

```python
from hailo_toolbox.inference.core import CallbackType

class CallbackType(Enum):
    PRE_PROCESSOR = "pre_processor"    # Preprocessor
    POST_PROCESSOR = "post_processor"  # Postprocessor
    VISUALIZER = "visualizer"          # Visualizer
    COLLATE_INFER = "collate_infer"    # Inference result organization
    SOURCE = "source"                  # Data source (usually no need to customize)
```

### Registration Methods

```python
from hailo_toolbox.inference.core import CALLBACK_REGISTRY

# Method 1: Decorator registration (recommended)
@CALLBACK_REGISTRY.registryPreProcessor("my_model")
def my_preprocess(image):
    return processed_image

# Method 2: Multi-name registration (one implementation supports multiple models)
@CALLBACK_REGISTRY.registryPostProcessor("model_v1", "model_v2")
class MyPostProcessor:
    def __call__(self, results): pass

# Method 3: Direct registration
CALLBACK_REGISTRY.register_callback("my_model", CallbackType.PRE_PROCESSOR, preprocess_func)
```

## Quick Implementation Example

Here's a complete custom model implementation example:

```python
"""
Custom model implementation example
Suitable for object detection type models
"""
from hailo_toolbox.inference.core import InferenceEngine, CALLBACK_REGISTRY
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig
import yaml
import numpy as np
import cv2

# Required implementation
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

# Optional implementation
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
    # Configure input shape
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

## Using Custom Models

After implementing and registering modules, you can use custom models:

```python
from hailo_toolbox.inference import InferenceEngine

# Create inference engine
engine = InferenceEngine(
    model="models/my_custom_model.hef",  # or .onnx
    source="test_video.mp4",
    task_name="my_detection_model",      # Must match registration name
    show=True,
    save_dir="output/"
)

# Run inference
engine.run()
```

## Minimum Implementation Requirements

If you just want to quickly verify a model, you only need to implement the postprocessor:

```python
# Simplest postprocessor
@CALLBACK_REGISTRY.registryPostProcessor("simple_model")
def simple_postprocess(results, original_shape=None):
    # Return empty results (for testing)
    return []

# Use built-in preprocessor configuration
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig

preprocess_config = PreprocessConfig(
    target_size=(640, 640),  # Model input size
    normalize=False           # Whether to normalize
)

engine = InferenceEngine(
    model="model.hef",
    source="video.mp4", 
    task_name="simple_model",
    preprocess_config=preprocess_config  # Use built-in preprocessor
)
engine.run()
```

## Debugging Tips

1. **Add Logging**: Add log output at key steps
```python
import logging
logger = logging.getLogger(__name__)

def __call__(self, image):
    logger.info(f"Input shape: {image.shape}")
    processed = self.process(image)
    logger.info(f"Output shape: {processed.shape}")
    return processed
```

2. **Save Intermediate Results**: Save preprocessed images during debugging
```python
def __call__(self, image):
    processed = self.process(image)
    # Save during debugging
    if self.debug:
        cv2.imwrite("debug_preprocessed.jpg", processed[0].transpose(1,2,0)*255)
    return processed
```

3. **Step-by-step Testing**: Test each module with a single image first
```python
# Test preprocessing
preprocessor = MyPreProcessor()
test_image = cv2.imread("test.jpg")
processed = preprocessor(test_image)
print(f"Preprocessed shape: {processed.shape}")
```

## Common Questions

**Q: How to determine model input/output format?**
A: You can use ONNX tools to view model information, or refer to the model's official documentation.

**Q: What if preprocessor output dimensions are wrong?**
A: Check the model's expected input format, usually (N, C, H, W) or (N, H, W, C).

**Q: How to handle multi-output models in postprocessor?**
A: Iterate through all outputs in the results dictionary and process each according to its meaning.

**Q: Can I not implement the visualizer?**
A: Yes, the visualizer is optional. The system will use a default empty implementation if not provided.

**Q: Can I not implement the preprocessor?**
A: Yes, the system provides a built-in generic preprocessor. Most model preprocessing needs can be met through `PreprocessConfig` configuration.

**Q: When do I need a custom preprocessor?**
A: When the model has special preprocessing requirements, such as special normalization methods, data augmentation, or complex input format conversions.

Through this guide, you should be able to quickly implement necessary processing modules for custom models. It's recommended to start with a minimal functional version (only postprocessor required), verify the workflow, and then gradually improve each module's functionality. 