# DL Toolbox Sources Module

Deep learning model input stream module, supporting unified interface for various video and image sources.

## Features

### Supported Source Types

1. **Image Sources (ImageSource)**
   - Single image files (JPG, PNG, BMP, TIFF, WebP, GIF)
   - Image folders (automatically scan all supported formats)
   - Network image URLs

2. **Video Sources (VideoSource)**
   - Local video files (MP4, AVI, MOV, MKV, WebM, FLV, WMV, M4V)
   - Network video streams (HTTP/HTTPS)

3. **Camera Sources**
   - **USB Cameras (WebcamSource)**: Standard USB cameras and integrated cameras
   - **IP Cameras (IPCameraSource)**: RTSP stream support with buffering and reconnection mechanisms
   - **MIPI Cameras (MIPICameraSource)**: Support for embedded platforms (Jetson, Raspberry Pi)

4. **Multi-Source Management (MultiSourceManager)**
   - Process multiple input sources simultaneously
   - Support different synchronization modes (latest, nearest, wait_all)
   - Thread pool management and frame buffering

### Core Features

- **Automatic Source Type Detection**: Automatically identify source type based on input
- **Unified Interface**: All source types use the same API
- **Error Handling**: Comprehensive error handling and recovery mechanisms
- **Performance Optimization**: Multi-threading and intelligent buffering
- **Extensibility**: Easy to add new source types

## Installation

```bash
# Install basic dependencies
pip install -r requirements.txt

# For platform-specific optional dependencies, refer to comments in requirements.txt
```

## Quick Start

### Basic Usage

```python
from hailo_toolbox.sources import create_source

# Automatically detect source type and create
source = create_source("path/to/video.mp4")

# Use context manager
with source:
    for frame in source:
        print(f"Frame shape: {frame.shape}")
        # Process frame...
        break
```

### Manual Creation of Specific Source Types

```python
from hailo_toolbox.sources import ImageSource, VideoSource, WebcamSource

# Image source
image_source = ImageSource("img_src", {
    "source_path": "path/to/images/",
    "resolution": (640, 480),
    "loop": True
})

# Video source
video_source = VideoSource("vid_src", {
    "file_path": "path/to/video.mp4",
    "resolution": (1280, 720),
    "fps": 30,
    "loop": True
})

# Camera source
webcam_source = WebcamSource("cam_src", {
    "device_id": 0,
    "resolution": (1920, 1080),
    "fps": 30
})
```

### Multi-Source Configuration

```python
from hailo_toolbox.sources import MultiSourceManager

# Configure multiple sources
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

# Create multi-source manager
multi_source = MultiSourceManager("multi", {
    "sources": sources_config,
    "sync_mode": "latest",  # or "nearest", "wait_all"
    "fps": 30
})

# Read synchronized frames
with multi_source:
    success, frames_dict = multi_source.read()
    if success:
        for source_id, frame in frames_dict.items():
            print(f"Source {source_id}: {frame.shape}")
```

## Detailed Configuration

### ImageSource Configuration

```python
config = {
    "source_path": "path/to/image_or_folder",
    "supported_formats": [".jpg", ".png", ".bmp"],  # Supported formats
    "loop": True,                    # Folder looping
    "sort_files": True,              # File sorting
    "timeout": 10,                   # URL timeout (seconds)
    "resolution": (640, 480),        # Output resolution
}
```

### VideoSource Configuration

```python
config = {
    "file_path": "path/to/video.mp4",
    "loop": False,                   # Video looping
    "start_frame": 0,                # Starting frame
    "fps": 30,                       # Target FPS
    "resolution": (1280, 720),       # Output resolution
    "timeout": 10,                   # Network stream timeout (seconds)
}
```

### WebcamSource Configuration

```python
config = {
    "device_id": 0,                  # Device ID
    "fps": 30,                       # Frame rate
    "resolution": (1920, 1080),      # Resolution
    "api_preference": "AUTO",        # API preference (AUTO, DSHOW, V4L2, etc.)
}
```

### IPCameraSource Configuration

```python
config = {
    "url": "rtsp://username:password@ip:port/stream",
    "buffer_size": 30,               # Buffer size
    "reconnect_attempts": 3,         # Reconnection attempts
    "reconnect_delay": 5,            # Reconnection delay (seconds)
    "timeout": 10,                   # Connection timeout (seconds)
    "resolution": (1280, 720),       # Output resolution
}
```

### MIPICameraSource Configuration

```python
config = {
    "pipeline_type": "gstreamer",    # Pipeline type (gstreamer, jetson, picamera2)
    "sensor_id": 0,                  # Sensor ID
    "fps": 30,                       # Frame rate
    "resolution": (1920, 1080),      # Resolution
    "flip_method": 0,                # Flip method
    "sensor_mode": 0,                # Sensor mode
    "custom_pipeline": None,         # Custom GStreamer pipeline
}
```

### MultiSourceManager Configuration

```python
config = {
    "sources": [...],                # Source configuration list
    "sync_mode": "latest",           # Sync mode (latest, nearest, wait_all)
    "max_queue_size": 30,            # Maximum queue size
    "timeout": 10,                   # Operation timeout (seconds)
    "fps": 30,                       # Target FPS
}
```

## Automatic Source Type Detection

The module automatically detects source type based on input:

```python
from hailo_toolbox.sources import detect_source_type, SourceType

# Image files
detect_source_type("image.jpg")          # -> SourceType.IMAGE
detect_source_type("images_folder/")     # -> SourceType.IMAGE

# Video files  
detect_source_type("video.mp4")          # -> SourceType.FILE

# Cameras
detect_source_type(0)                    # -> SourceType.WEBCAM

# IP cameras
detect_source_type("rtsp://...")         # -> SourceType.IP_CAMERA

# MIPI cameras
detect_source_type("/dev/video0")        # -> SourceType.MIPI_CAMERA
detect_source_type("v4l2://...")         # -> SourceType.MIPI_CAMERA

# Network URLs
detect_source_type("http://...jpg")      # -> SourceType.IMAGE
detect_source_type("http://...mp4")      # -> SourceType.FILE

# Multi-source
detect_source_type([0, 1, "video.mp4"]) # -> SourceType.MULTI
```

## Error Handling

All sources implement comprehensive error handling:

```python
source = create_source("invalid_source")

try:
    if source.open():
        success, frame = source.read()
        if not success:
            print("Read failed")
    else:
        print("Open failed")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    source.close()
```

## Performance Optimization Recommendations

1. **Use Appropriate Resolution**: Avoid unnecessary high resolution
2. **Set Reasonable FPS**: Set frame rate according to actual needs
3. **Buffer Size**: For network sources, appropriately increase buffer size
4. **Multi-threading**: MultiSourceManager automatically handles multi-threading
5. **Resource Release**: Use context managers or manually call close()

## Testing

Run complete test suite:

```bash
# Run all tests
python test_sources.py

# Custom test parameters
python test_sources.py --duration 10 --max-frames 20 --verbose

# Run specific tests
python -m pytest tests/test_sources.py -v
```

## Platform-Specific Instructions

### Jetson Platform (MIPI Cameras)

```bash
# Install Jetson tools
sudo apt-get install nvidia-jetpack

# Use CSI camera
source = create_source("csi://0", config={
    "pipeline_type": "jetson",
    "resolution": (1920, 1080),
    "fps": 30
})
```

### Raspberry Pi (MIPI Cameras)

```bash
# Install picamera2
pip install picamera2

# Use Pi camera
source = create_source("/dev/video0", config={
    "pipeline_type": "picamera2",
    "resolution": (1640, 1232),
    "fps": 30
})
```

## Extension Development

Adding new source types:

```python
from hailo_toolbox.sources.base import BaseSource, SourceType

class CustomSource(BaseSource):
    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(source_id, config)
        self.source_type = SourceType.CUSTOM
        
    def open(self) -> bool:
        # Implement open logic
        pass
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # Implement read logic
        pass
        
    def close(self) -> None:
        # Implement close logic
        pass
```

## Common Questions

### Q: How to handle unstable network camera connections?
A: IPCameraSource has built-in reconnection mechanisms. You can adjust by configuring `reconnect_attempts` and `reconnect_delay` parameters.

### Q: What synchronization modes are available for multi-source?
A: Three modes are supported:
- `latest`: Get the latest frame from each source
- `nearest`: Get frames with closest timestamps
- `wait_all`: Wait for all sources to have available frames

### Q: How to optimize memory usage?
A: You can control memory usage by setting smaller `max_queue_size` and appropriate resolution.

### Q: What video formats are supported?
A: Supports all formats that OpenCV can handle, including MP4, AVI, MOV, MKV, WebM, etc.

## License

This project uses the MIT license. See LICENSE file for details. 