# Hailo Toolbox Developer Guide

This document is for developers who want to implement custom model inference processing modules for Hailo Toolbox.

## Overview

Hailo Toolbox adopts a modular architecture and manages various processing modules through a registration mechanism. To support new custom models, you need to implement corresponding processing modules and register them in the system.

## Core Module Description

### Module Classification

| Module Type | Required | Function | Implementation Complexity |
|-------------|----------|----------|---------------------------|
| **PreProcessor** | 🔶 Optional | Image preprocessing, convert to model input format | Simple |
| **PostProcessor** | ✅ Required | Model output post-processing, parse inference results | Medium |
| **EventProcessor** | 🔶 Optional | Event processing, execute extended functionality based on prediction results | Simple-Medium |
| **Visualizer** | 🔶 Optional | Result visualization, draw detection boxes on images | Simple |
| **CollateInfer** | 🔶 Optional | Inference result organization, format model raw output | Simple |
| **Source** | ❌ Not needed | Data source management, generic implementation available | - |

### Detailed Module Responsibilities

#### PreProcessor (Preprocessor) - Optional Implementation
- **Function**: Convert input images to model-required format
- **Input**: Raw image (H, W, C) BGR format
- **Output**: Preprocessed tensor, usually (N, C, H, W) format
- **Note**: System has built-in generic preprocessor, configurable via `PreprocessConfig`. Custom implementation only needed for special requirements.
- **Main Tasks**:
  - Image size adjustment
  - Color space conversion (BGR→RGB)
  - Data normalization and standardization
  - Dimension conversion (HWC→CHW)

#### PostProcessor (Post-processor) - Required Implementation
- **Function**: Process model raw output, convert to usable results
- **Input**: Model inference output dictionary
- **Output**: Structured detection/classification result list
- **Note**: Each model has different output formats, must implement corresponding post-processing logic.
- **Main Tasks**:
  - Decode model output
  - Confidence filtering
  - Non-Maximum Suppression (NMS)
  - Coordinate transformation

#### EventProcessor (Event Processor) - Optional Implementation
- **Function**: Execute extended functionality and business logic based on model prediction results
- **Input**: Structured results from post-processor output
- **Output**: Processed results or triggered events
- **Note**: Used to implement business logic based on AI prediction results, such as statistics, alerts, data recording, etc.
- **Main Tasks**:
  - Object counting statistics (e.g., pedestrian count, vehicle count)
  - Anomaly detection and alerting
  - Data recording and log output
  - Business rule judgment and execution
  - Integration with external systems (databases, APIs, etc.)

#### Visualizer (Visualizer) - Optional Implementation
- **Function**: Draw inference results on images
- **Input**: Original image + post-processing results
- **Output**: Image with visualization annotations
- **Main Tasks**:
  - Draw bounding boxes
  - Display category labels and confidence
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
    POST_PROCESSOR = "post_processor"  # Post-processor
    EVENT_PROCESSOR = "event_processor" # Event processor
    VISUALIZER = "visualizer"          # Visualizer
    COLLATE_INFER = "collate_infer"    # Inference result organization
    SOURCE = "source"                  # Data source (usually no need for custom)
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

# Method 3: Event processor registration
@CALLBACK_REGISTRY.registryEventProcessor("my_model")
class MyEventProcessor:
    def __call__(self, results): pass

# Method 4: Direct registration
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

# Optional implementation - Event processor
@CALLBACK_REGISTRY.registryEventProcessor("custom")
class CustomEventProcessor:
    def __init__(self, config=None):
        self.config = config
        self.frame_count = 0
        self.detection_history = []

    def __call__(self, results):
        """
        Execute business logic based on prediction results
        
        Args:
            results: Structured results from post-processor output
            
        Returns:
            Processed results or None
        """
        self.frame_count += 1
        
        # Example 1: Simple result logging
        print(f"Frame {self.frame_count}: Detected classes: {results}")
        
        # Example 2: Detection history statistics
        self.detection_history.append(len(results))
        
        # Example 3: Anomaly detection logic
        if len(results) > 5:  # If more than 5 objects detected
            print(f"⚠️  Alert: High object count detected: {len(results)}")
        
        # Example 4: Periodic statistical reports
        if self.frame_count % 100 == 0:
            avg_detections = np.mean(self.detection_history[-100:])
            print(f"📊 Statistics: Average detections in last 100 frames: {avg_detections:.2f}")
        
        # Example 5: Can return processed results or trigger events
        return {
            'frame_id': self.frame_count,
            'detection_count': len(results),
            'results': results
        }

# Optional implementation - Visualizer
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

**Q: Can I skip implementing the preprocessor?**
A: Yes, the system provides a built-in generic preprocessor. Configuration via `PreprocessConfig` can meet most model preprocessing requirements.

**Q: When do I need to implement a custom preprocessor?**
A: When the model has special preprocessing requirements, such as special normalization methods, data augmentation, or complex input format conversions.

**Q: What is the EventProcessor for?**
A: EventProcessor is an optional module used to execute custom business logic based on AI prediction results, such as object counting, anomaly alerts, data recording, etc. It executes after the post-processor.

**Q: Will EventProcessor affect inference performance?**
A: If implemented properly, the impact is minimal. It's recommended to avoid time-consuming operations in event processors, and consider asynchronous processing for complex business logic.

**Q: How can I access the original image in the EventProcessor?**
A: Currently, EventProcessor only receives post-processing results. If you need to access the original image, you can implement related logic in the visualizer, or consider extending the EventProcessor interface.

Through this guide, you should be able to quickly implement necessary processing modules for custom models. It's recommended to start with a minimal functional version (only postprocessor required), verify the workflow, and then gradually improve each module's functionality.

## EventProcessor Detailed Guide

### Overview

EventProcessor is an optional extension module that allows developers to execute custom business logic based on model prediction results. It is positioned after the post-processor in the inference pipeline and can receive structured prediction results to execute various extended functionalities.

### Typical Application Scenarios

#### 1. Object Counting Statistics
```python
@CALLBACK_REGISTRY.registryEventProcessor("person_counter")
class PersonCounterProcessor:
    def __init__(self, config=None):
        self.person_count = 0
        self.total_frames = 0
        
    def __call__(self, results):
        self.total_frames += 1
        
        # Count detected persons
        current_person_count = sum(1 for obj in results if obj.get('class') == 'person')
        self.person_count += current_person_count
        
        # Output statistics every 100 frames
        if self.total_frames % 100 == 0:
            avg_persons = self.person_count / self.total_frames
            print(f"Average persons per frame: {avg_persons:.2f}")
```

#### 2. Anomaly Detection and Alerting
```python
@CALLBACK_REGISTRY.registryEventProcessor("security_monitor")
class SecurityMonitorProcessor:
    def __init__(self, config=None):
        self.alert_threshold = config.get('alert_threshold', 3) if config else 3
        self.consecutive_alerts = 0
        
    def __call__(self, results):
        # Check for anomalous situations (e.g., too many people)
        person_count = sum(1 for obj in results if obj.get('class') == 'person')
        
        if person_count > self.alert_threshold:
            self.consecutive_alerts += 1
            if self.consecutive_alerts >= 5:  # 5 consecutive frames exceed threshold
                self.send_alert(f"Abnormal gathering detected: {person_count} people")
                self.consecutive_alerts = 0
        else:
            self.consecutive_alerts = 0
            
    def send_alert(self, message):
        # Send alert (email, SMS, API call, etc.)
        print(f"🚨 ALERT: {message}")
        # Integrate with actual alert system here
```

#### 3. Data Recording and Analysis
```python
@CALLBACK_REGISTRY.registryEventProcessor("data_logger")
class DataLoggerProcessor:
    def __init__(self, config=None):
        self.log_file = config.get('log_file', 'detection_log.csv') if config else 'detection_log.csv'
        self.init_log_file()
        
    def init_log_file(self):
        import csv
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame_id', 'object_count', 'objects'])
    
    def __call__(self, results):
        import csv
        import datetime
        
        timestamp = datetime.datetime.now().isoformat()
        frame_id = getattr(self, 'frame_counter', 0)
        self.frame_counter = frame_id + 1
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, 
                frame_id, 
                len(results), 
                str(results)
            ])
```

#### 4. Business Rule Execution
```python
@CALLBACK_REGISTRY.registryEventProcessor("traffic_monitor")
class TrafficMonitorProcessor:
    def __init__(self, config=None):
        self.vehicle_count = 0
        self.traffic_state = "normal"
        
    def __call__(self, results):
        # Count vehicles
        vehicles = [obj for obj in results if obj.get('class') in ['car', 'truck', 'bus']]
        current_vehicle_count = len(vehicles)
        
        # Update traffic state
        if current_vehicle_count > 20:
            self.traffic_state = "heavy"
        elif current_vehicle_count > 10:
            self.traffic_state = "moderate"
        else:
            self.traffic_state = "light"
            
        # Execute corresponding business logic
        self.update_traffic_light_timing()
        
    def update_traffic_light_timing(self):
        # Adjust traffic light timing based on traffic state
        if self.traffic_state == "heavy":
            print("Adjusting traffic light: Extend green light duration")
        elif self.traffic_state == "light":
            print("Adjusting traffic light: Shorten green light duration")
```

### Implementation Guidelines

#### 1. Basic Structure
```python
@CALLBACK_REGISTRY.registryEventProcessor("your_processor_name")
class YourEventProcessor:
    def __init__(self, config=None):
        """
        Initialize event processor
        
        Args:
            config: Configuration parameters, can be None
        """
        self.config = config
        # Initialize your state variables
        
    def __call__(self, results):
        """
        Process prediction results
        
        Args:
            results: Structured results from post-processor output
            
        Returns:
            Optional: Return processed results or status information
        """
        # Implement your business logic
        pass
```

#### 2. State Management
Event processors typically need to maintain state information:
- Frame counters
- Historical data
- Statistical information
- Configuration parameters

#### 3. Performance Considerations
- Avoid time-consuming operations in event processors
- Consider asynchronous processing for I/O operations
- Regularly clean up historical data to prevent memory leaks

### Configuration Example

```python
# Pass event processor configuration when creating InferenceEngine
event_processor_config = {
    'alert_threshold': 5,
    'log_file': 'custom_log.csv',
    'enable_alerts': True
}

engine = InferenceEngine(
    model="model.hef",
    source="video.mp4",
    task_name="custom",
    event_processor_config=event_processor_config,
    show=True
)
```

Through event processors, you can easily combine AI inference results with actual business requirements to implement intelligent application logic. 