# Hailo Model Conversion Guide

This document will guide you step-by-step on how to convert common deep learning models to `.hef` format that can run on Hailo hardware. Even if you are a beginner, you can easily complete model conversion.

## What is Model Conversion?

Simply put, model conversion is "translating" your trained model into a language that Hailo AI chips can understand. It's like translating Chinese to English.

### Why Do We Need Conversion?

- **Compatibility**: Different AI chips have different "languages", conversion is needed for them to run
- **Optimization**: The conversion process optimizes for Hailo chips, improving running speed
- **Compression**: Converted models are usually smaller, taking up less storage space

## Supported Model Formats

### Input Formats (Your Models)

| Framework | File Format | Common Use | Example Filename |
|-----------|-------------|------------|------------------|
| **ONNX** | `.onnx` | Universal format, recommended | `yolov8n.onnx` |
| **TensorFlow** | `.h5` | Keras models | `model.h5` |
| **TensorFlow** | `saved_model.pb` | TensorFlow saved models | `saved_model.pb` |
| **TensorFlow Lite** | `.tflite` | Mobile models | `model.tflite` |
| **PyTorch** | `.pt` | TorchScript models | `model.pt` |
| **PaddlePaddle** | Inference models | Baidu PaddlePaddle models | `inference.pdmodel` |

### Output Format (After Conversion)

- **`.hef`**: Hailo Executable Format, Hailo's proprietary optimized format

## Preparation

### 1. Confirm System Requirements

```bash
# Check operating system (must be Linux)
uname -a

# Check Python version (needs 3.8-3.11)
python3 --version
```

### 2. Install Necessary Software

```bash
# Install Hailo Toolbox
pip install -e .

# Verify installation
hailo-toolbox --version
```

### 3. Prepare Model Files

Make sure you have:
- ✅ Model file (e.g., `model.onnx`)
- ✅ Know the model's input size (e.g., 640x640)
- ✅ Calibration dataset (recommended, optional)

## Basic Conversion Tutorial

### Step 1: Simplest Conversion

If you have an ONNX model, the simplest conversion command is:

```bash
hailo-toolbox convert your_model.onnx
```

**Explanation**:
- `hailo-toolbox convert`: Start conversion tool
- `your_model.onnx`: Your model filename

After conversion completes, a `your_model.hef` file will be generated in the same directory.

### Step 2: Specify Output Directory

For better file management, it's recommended to specify an output directory:

```bash
hailo-toolbox convert your_model.onnx --output-dir ./converted_models
```

This way the converted files will be saved in the `converted_models` folder.

### Step 3: Specify Hardware Architecture

Choose the corresponding architecture based on your Hailo device:

```bash
# Hailo-8 chip (most common)
hailo-toolbox convert your_model.onnx --hw-arch hailo8

# Hailo-8L chip
hailo-toolbox convert your_model.onnx --hw-arch hailo8l

# Hailo-15 chip
hailo-toolbox convert your_model.onnx --hw-arch hailo15

# Hailo-15L chip
hailo-toolbox convert your_model.onnx --hw-arch hailo15l
```

## Advanced Conversion Options

### Using Calibration Dataset (Recommended)

Calibration datasets can improve the accuracy of converted models:

```bash
hailo-toolbox convert your_model.onnx \
    --hw-arch hailo8 \
    --calib-set-path ./calibration_images \
    --output-dir ./converted_models
```

**Calibration Dataset Requirements**:
- 📁 Folder containing representative images
- 🖼️ Image formats: JPG, PNG, etc.
- 📊 Quantity: Recommend 100-1000 images
- 🎯 Content: Images similar to actual usage scenarios

### Specify Input Size

If conversion encounters size-related errors, you can manually specify:

```bash
hailo-toolbox convert your_model.onnx \
    --input-shape 640,640,3 \
    --hw-arch hailo8
```

**Input Size Format**:
- `640,640,3`: Width,Height,Channels
- `224,224,3`: Common classification model size
- `320,320,3`: Lightweight detection model size

### Use Random Calibration (Quick Testing)

If you don't have a calibration dataset, you can use random data:

```bash
hailo-toolbox convert your_model.onnx \
    --use-random-calib-set \
    --hw-arch hailo8
```

⚠️ **Note**: Random calibration may have lower accuracy, only suitable for quick testing.

## Practical Conversion Examples

### Example 1: YOLOv8 Object Detection Model

```bash
# Download YOLOv8 ONNX model (assuming you already have it)
# Conversion command
hailo-toolbox convert yolov8n.onnx \
    --hw-arch hailo8 \
    --input-shape 640,640,3 \
    --calib-set-path ./coco_samples \
    --output-dir ./converted_models \
    --save-onnx
```

**Parameter Explanation**:
- `--input-shape 640,640,3`: YOLOv8's standard input size
- `--calib-set-path ./coco_samples`: Use COCO dataset samples for calibration
- `--save-onnx`: Save optimized ONNX file

### Example 2: Image Classification Model

```bash
hailo-toolbox convert efficientnet.onnx \
    --hw-arch hailo8 \
    --input-shape 224,224,3 \
    --calib-set-path ./imagenet_samples \
    --output-dir ./converted_models
```

### Example 3: Quick Test Conversion

```bash
# Fastest conversion method (for quick validation)
hailo-toolbox convert test_model.onnx \
    --use-random-calib-set \
    --hw-arch hailo8
```

## Conversion Parameter Details

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model` | Input model file path | `yolov8n.onnx` |

### Common Optional Parameters

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--hw-arch` | `hailo8` | Hardware architecture | `hailo8`, `hailo15` |
| `--output-dir` | Same as model | Output directory | `./models` |
| `--input-shape` | Auto-detect | Input size | `640,640,3` |
| `--calib-set-path` | None | Calibration dataset path | `./calib_images` |
| `--use-random-calib-set` | `False` | Use random calibration | - |
| `--save-onnx` | `False` | Save optimized ONNX | - |
| `--profile` | `False` | Generate performance report | - |

## Common Problem Solutions

### Q1: Conversion shows "model file not found"

**Problem**: `FileNotFoundError: model.onnx not found`

**Solution**:
```bash
# Check if file exists
ls -la your_model.onnx

# Use absolute path
hailo-toolbox convert /full/path/to/your_model.onnx
```

### Q2: Conversion shows need calibration dataset

**Problem**: `calibration dataset required`

**Solution**:
```bash
# Method 1: Use random calibration (quick)
hailo-toolbox convert model.onnx --use-random-calib-set

# Method 2: Prepare calibration dataset
mkdir calibration_images
# Put some representative images
hailo-toolbox convert model.onnx --calib-set-path ./calibration_images
```

### Q3: Conversion is very slow

**Reason**: Model conversion is compute-intensive and takes time

**Optimization Suggestions**:
- Use fewer calibration images (around 100)
- Use random calibration for quick testing
- Ensure system has sufficient memory

### Q4: Converted model is large

**Solution**:
```bash
# Check file size
ls -lh *.hef

# If too large, try:
# 1. Use smaller input size
hailo-toolbox convert model.onnx --input-shape 320,320,3

# 2. Check for unnecessary output nodes
hailo-toolbox convert model.onnx --end-nodes output1,output2
```

## Verify Conversion Results

### Check if Conversion Succeeded

```bash
# Check generated files
ls -la *.hef

# View file information
file your_model.hef
```

### Quick Test Converted Model

```bash
# Use converted model for inference testing
hailo-toolbox infer your_model.hef \
    --source test_image.jpg \
    --task-name yolov8det \
    --show
```

## Performance Optimization Suggestions

### 1. Calibration Dataset Optimization

- **Quantity**: 100-500 images usually sufficient
- **Quality**: Choose images similar to actual scenarios
- **Diversity**: Include different lighting, angles, backgrounds

### 2. Input Size Selection

```bash
# High accuracy (slower)
--input-shape 640,640,3

# Balanced (recommended)
--input-shape 416,416,3

# High speed (faster)
--input-shape 320,320,3
```

### 3. Hardware Architecture Selection

- **Hailo-8**: High performance, suitable for complex models
- **Hailo-8L**: Low power version
- **Hailo-15**: Latest architecture, higher performance

## Conversion Process Diagram

```
[Original Model] → [Conversion Tool] → [Calibration] → [Optimization] → [HEF Model]
    ↓               ↓                   ↓              ↓               ↓
 .onnx file    hailo-toolbox       Calibration data  Hardware opt   .hef file
```

## Summary

Basic model conversion process:

1. **Prepare model file** (.onnx format recommended)
2. **Prepare calibration data** (optional but recommended)
3. **Run conversion command** 
4. **Verify conversion results**
5. **Test inference performance**

**Remember this universal command**:
```bash
hailo-toolbox convert your_model.onnx \
    --hw-arch hailo8 \
    --calib-set-path ./calibration_images \
    --output-dir ./converted_models
```

After successful conversion, you can run your AI model at high speed on Hailo devices!

---

**Next Step**: Learn how to use converted models for inference, please refer to [Model Inference Guide](INFERENCE.md). 