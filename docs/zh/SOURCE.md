# DL Toolbox Sources Module

深度学习模型输入流模块，支持多种视频和图像源的统一接口。

## 功能特性

### 支持的源类型

1. **图像源 (ImageSource)**
   - 单张图片文件 (JPG, PNG, BMP, TIFF, WebP, GIF)
   - 图片文件夹 (自动扫描所有支持格式)
   - 网络图片URL

2. **视频源 (VideoSource)**
   - 本地视频文件 (MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V)
   - 网络视频流 (HTTP/HTTPS)

3. **摄像头源**
   - **USB摄像头 (WebcamSource)**: 标准USB摄像头和集成摄像头
   - **IP摄像头 (IPCameraSource)**: RTSP流支持，包含缓冲和重连机制
   - **MIPI摄像头 (MIPICameraSource)**: 支持嵌入式平台 (Jetson, Raspberry Pi)

4. **多源管理 (MultiSourceManager)**
   - 同时处理多个输入源
   - 支持不同同步模式 (latest, nearest, wait_all)
   - 线程池管理和帧缓冲

### 核心特性

- **自动源类型检测**: 根据输入自动识别源类型
- **统一接口**: 所有源类型使用相同的API
- **错误处理**: 完善的错误处理和恢复机制
- **性能优化**: 多线程处理和智能缓冲
- **可扩展性**: 易于添加新的源类型

## 安装

```bash
# 安装基础依赖
pip install -r requirements.txt

# 对于特定平台的可选依赖，请参考requirements.txt中的注释
```

## 快速开始

### 基本使用

```python
from hailo_toolbox.sources import create_source

# 自动检测源类型并创建
source = create_source("path/to/video.mp4")

# 使用上下文管理器
with source:
    for frame in source:
        print(f"Frame shape: {frame.shape}")
        # 处理帧...
        break
```

### 手动创建特定源类型

```python
from hailo_toolbox.sources import ImageSource, VideoSource, WebcamSource

# 图像源
image_source = ImageSource("img_src", {
    "source_path": "path/to/images/",
    "resolution": (640, 480),
    "loop": True
})

# 视频源
video_source = VideoSource("vid_src", {
    "file_path": "path/to/video.mp4",
    "resolution": (1280, 720),
    "fps": 30,
    "loop": True
})

# 摄像头源
webcam_source = WebcamSource("cam_src", {
    "device_id": 0,
    "resolution": (1920, 1080),
    "fps": 30
})
```

### 多源配置

```python
from hailo_toolbox.sources import MultiSourceManager

# 配置多个源
sources_config = [
    {
        "type": "WEBCAM",
        "id": "camera_1", 
        "device_id": 0,
        "resolution": (640, 480)
    },
    {
        "type": "FILE",
        "id": "video_1",
        "file_path": "path/to/video.mp4",
        "resolution": (640, 480)
    }
]

# 创建多源管理器
multi_source = MultiSourceManager("multi", {
    "sources": sources_config,
    "sync_mode": "latest",  # 或 "nearest", "wait_all"
    "fps": 30
})

# 读取同步帧
with multi_source:
    success, frames_dict = multi_source.read()
    if success:
        for source_id, frame in frames_dict.items():
            print(f"Source {source_id}: {frame.shape}")
```

## 详细配置

### ImageSource 配置

```python
config = {
    "source_path": "path/to/image_or_folder",
    "supported_formats": [".jpg", ".png", ".bmp"],  # 支持的格式
    "loop": True,                    # 文件夹循环
    "sort_files": True,              # 文件排序
    "timeout": 10,                   # URL超时(秒)
    "resolution": (640, 480),        # 输出分辨率
}
```

### VideoSource 配置

```python
config = {
    "file_path": "path/to/video.mp4",
    "loop": False,                   # 视频循环
    "start_frame": 0,                # 起始帧
    "fps": 30,                       # 目标FPS
    "resolution": (1280, 720),       # 输出分辨率
    "timeout": 10,                   # 网络流超时(秒)
}
```

### WebcamSource 配置

```python
config = {
    "device_id": 0,                  # 设备ID
    "fps": 30,                       # 帧率
    "resolution": (1920, 1080),      # 分辨率
    "api_preference": "AUTO",        # API偏好 (AUTO, DSHOW, V4L2等)
}
```

### IPCameraSource 配置

```python
config = {
    "url": "rtsp://username:password@ip:port/stream",
    "buffer_size": 30,               # 缓冲区大小
    "reconnect_attempts": 3,         # 重连尝试次数
    "reconnect_delay": 5,            # 重连延迟(秒)
    "timeout": 10,                   # 连接超时(秒)
    "resolution": (1280, 720),       # 输出分辨率
}
```

### MIPICameraSource 配置

```python
config = {
    "pipeline_type": "gstreamer",    # 管道类型 (gstreamer, jetson, picamera2)
    "sensor_id": 0,                  # 传感器ID
    "fps": 30,                       # 帧率
    "resolution": (1920, 1080),      # 分辨率
    "flip_method": 0,                # 翻转方法
    "sensor_mode": 0,                # 传感器模式
    "custom_pipeline": None,         # 自定义GStreamer管道
}
```

### MultiSourceManager 配置

```python
config = {
    "sources": [...],                # 源配置列表
    "sync_mode": "latest",           # 同步模式 (latest, nearest, wait_all)
    "max_queue_size": 30,            # 最大队列大小
    "timeout": 10,                   # 操作超时(秒)
    "fps": 30,                       # 目标FPS
}
```

## 源类型自动检测

模块会根据输入自动检测源类型：

```python
from hailo_toolbox.sources import detect_source_type, SourceType

# 图像文件
detect_source_type("image.jpg")          # -> SourceType.IMAGE
detect_source_type("images_folder/")     # -> SourceType.IMAGE

# 视频文件  
detect_source_type("video.mp4")          # -> SourceType.FILE

# 摄像头
detect_source_type(0)                    # -> SourceType.WEBCAM

# IP摄像头
detect_source_type("rtsp://...")         # -> SourceType.IP_CAMERA

# MIPI摄像头
detect_source_type("/dev/video0")        # -> SourceType.MIPI_CAMERA
detect_source_type("v4l2://...")         # -> SourceType.MIPI_CAMERA

# 网络URL
detect_source_type("http://...jpg")      # -> SourceType.IMAGE
detect_source_type("http://...mp4")      # -> SourceType.FILE

# 多源
detect_source_type([0, 1, "video.mp4"]) # -> SourceType.MULTI
```

## 错误处理

所有源都实现了完善的错误处理：

```python
source = create_source("invalid_source")

try:
    if source.open():
        success, frame = source.read()
        if not success:
            print("读取失败")
    else:
        print("打开失败")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    source.close()
```

## 性能优化建议

1. **使用适当的分辨率**: 避免不必要的高分辨率
2. **合理设置FPS**: 根据实际需求设置帧率
3. **缓冲区大小**: 对于网络源，适当增加缓冲区
4. **多线程**: MultiSourceManager自动处理多线程
5. **资源释放**: 使用上下文管理器或手动调用close()

## 测试

运行完整的测试套件：

```bash
# 运行所有测试
python test_sources.py

# 自定义测试参数
python test_sources.py --duration 10 --max-frames 20 --verbose

# 运行特定测试
python -m pytest tests/test_sources.py -v
```

## 平台特定说明

### Jetson平台 (MIPI摄像头)

```bash
# 安装Jetson工具
sudo apt-get install nvidia-jetpack

# 使用CSI摄像头
source = create_source("csi://0", config={
    "pipeline_type": "jetson",
    "resolution": (1920, 1080),
    "fps": 30
})
```

### Raspberry Pi (MIPI摄像头)

```bash
# 安装picamera2
pip install picamera2

# 使用Pi摄像头
source = create_source("/dev/video0", config={
    "pipeline_type": "picamera2",
    "resolution": (1640, 1232),
    "fps": 30
})
```

## 扩展开发

添加新的源类型：

```python
from hailo_toolbox.sources.base import BaseSource, SourceType

class CustomSource(BaseSource):
    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(source_id, config)
        self.source_type = SourceType.CUSTOM
        
    def open(self) -> bool:
        # 实现打开逻辑
        pass
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # 实现读取逻辑
        pass
        
    def close(self) -> None:
        # 实现关闭逻辑
        pass
```

## 常见问题

### Q: 如何处理网络摄像头连接不稳定？
A: IPCameraSource内置了重连机制，可以通过配置`reconnect_attempts`和`reconnect_delay`参数调整。

### Q: 多源同步有什么模式？
A: 支持三种模式：
- `latest`: 获取每个源的最新帧
- `nearest`: 获取时间戳最接近的帧
- `wait_all`: 等待所有源都有帧可用

### Q: 如何优化内存使用？
A: 可以通过设置较小的`max_queue_size`和适当的分辨率来控制内存使用。

### Q: 支持哪些视频格式？
A: 支持OpenCV能处理的所有格式，包括MP4, AVI, MOV, MKV, WebM等。

## 许可证

本项目采用MIT许可证。详见LICENSE文件。 