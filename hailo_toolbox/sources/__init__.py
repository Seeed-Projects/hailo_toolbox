"""
Module for different video source implementations.
"""

import os
import os.path as osp
from typing import Any, Dict, Union, List, Optional
from pathlib import Path
from urllib.parse import urlparse
from .base import BaseSource, SourceType
from .file import ImageSource, VideoSource, FileSource, FolderSource
from .webcam import WebcamSource
from .ip_camera import IPCameraSource
from .mipi import MIPICameraSource
from .multi import MultiSourceManager
from hailo_toolbox.inference.core import CALLBACK_REGISTRY


def is_url(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is a URL.
    """
    if not isinstance(source, str):
        return False
    try:
        result = urlparse(source)
        return all([result.scheme, result.netloc])
    except:
        return False


def is_image_file(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is an image file.
    """
    if not isinstance(source, (str, Path)):
        return False

    path = Path(source)
    if not path.exists() or not path.is_file():
        return False

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
    ]
    return path.suffix.lower() in image_extensions


def is_image_folder(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is a folder containing images.
    """
    if not isinstance(source, (str, Path)):
        return False

    path = Path(source)
    if not path.exists() or not path.is_dir():
        return False

    # Check if folder contains any image files
    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
    ]
    for ext in image_extensions:
        if list(path.glob(f"*{ext}")) or list(path.glob(f"*{ext.upper()}")):
            return True
    return False


def is_media_folder(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is a folder containing mixed media files (images and videos).
    """
    if not isinstance(source, (str, Path)):
        return False

    path = Path(source)
    if not path.exists() or not path.is_dir():
        return False

    # Check if folder contains any media files (images or videos)
    media_extensions = [
        # Image formats
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
        # Video formats
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".flv",
        ".wmv",
        ".m4v",
    ]

    for ext in media_extensions:
        if list(path.glob(f"*{ext}")) or list(path.glob(f"*{ext.upper()}")):
            return True
    return False


def is_image_url(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is an image URL.
    """
    if not is_url(source):
        return False

    # Check common image file extensions in URL
    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
    ]
    source_lower = source.lower()

    # Check if URL ends with image extension
    for ext in image_extensions:
        if source_lower.endswith(ext):
            return True

    # Check if URL contains image-related keywords
    image_keywords = ["image", "img", "photo", "picture", "pic"]
    for keyword in image_keywords:
        if keyword in source_lower:
            return True

    return False


def is_video_file(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is a video file.
    """
    if not isinstance(source, (str, Path)):
        return False

    path = Path(source)
    if not path.exists() or not path.is_file():
        return False

    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]
    return path.suffix.lower() in video_extensions


def is_video_url(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is a video URL (network stream).
    """
    if not isinstance(source, str):
        return False

    # Check for streaming protocols
    streaming_protocols = ["http://", "https://", "rtmp://", "udp://", "tcp://"]
    if any(source.startswith(protocol) for protocol in streaming_protocols):
        # Check if it's not an image URL
        if not is_image_url(source):
            return True

    return False


def is_webcam(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is a webcam (integer device ID).
    """
    if isinstance(source, str):
        if str(source).isdigit():
            return True
        else:
            return False
    elif isinstance(source, int):
        return source >= 0
    return False


def is_ip_camera(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is an IP camera (RTSP stream).
    """
    return isinstance(source, str) and source.startswith("rtsp://")


def is_mipi_camera(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """
    Check if the source is a MIPI camera.
    """
    if isinstance(source, str):
        # Check for v4l2 device path or CSI camera identifier
        return (
            source.startswith("/dev/video")
            or source.startswith("v4l2://")
            or source.startswith("csi://")
        )
    return False


def is_multi_source(source: Union[str, int, Dict[str, Any], Path, List]) -> bool:
    """
    Check if the source is a multi-source configuration.
    """
    return isinstance(source, (list, dict)) and (
        isinstance(source, list) or (isinstance(source, dict) and "sources" in source)
    )


def detect_source_type(
    source: Union[str, int, Dict[str, Any], Path, List],
) -> SourceType:
    """
    Automatically detect the source type based on the input.

    Args:
        source: Source input (path, URL, device ID, etc.)

    Returns:
        Detected SourceType.
    """
    # Multi-source check first
    if is_multi_source(source):
        return SourceType.MULTI

    # Webcam check (integer device ID)
    if is_webcam(source):
        return SourceType.WEBCAM

    # IP camera check (RTSP)
    if is_ip_camera(source):
        return SourceType.IP_CAMERA

    # MIPI camera check
    if is_mipi_camera(source):
        return SourceType.MIPI_CAMERA

    # URL checks
    if is_url(source):
        if is_image_url(source):
            return SourceType.IMAGE
        elif is_video_url(source):
            return SourceType.FILE  # Network video stream
        else:
            # Default to video for unknown URLs
            return SourceType.FILE

    # Local file/folder checks
    if is_media_folder(source):
        # Check if it's a mixed media folder or just images
        if is_image_folder(source):
            # Check if it contains videos too
            path = Path(source)
            video_extensions = [
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".webm",
                ".flv",
                ".wmv",
                ".m4v",
            ]
            has_videos = any(
                list(path.glob(f"*{ext}")) or list(path.glob(f"*{ext.upper()}"))
                for ext in video_extensions
            )
            if has_videos:
                return SourceType.FOLDER  # Mixed media folder
            else:
                return SourceType.IMAGE  # Images only folder
        else:
            return SourceType.FOLDER  # Mixed media folder
    elif is_image_file(source):
        return SourceType.IMAGE
    elif is_video_file(source):
        return SourceType.FILE

    # Default fallback
    raise ValueError(f"Cannot detect source type for: {source}")


@CALLBACK_REGISTRY.registrySource("base")
def create_source(
    source: Union[str, int, Dict[str, Any], Path, List],
    source_id: Optional[str] = None,
    config: Dict[str, Any] = None,
) -> BaseSource:
    """
    Factory method to create a source based on automatic type detection.

    Args:
        source: Source input (path, URL, device ID, etc.)
        source_id: Optional source ID (auto-generated if not provided)
        config: Optional configuration dictionary

    Returns:
        Instance of appropriate BaseSource subclass.
    """
    # Auto-generate source ID if not provided
    if source_id is None:
        if isinstance(source, (str, Path)):
            source_id = f"source_{hash(str(source)) % 10000}"
        elif isinstance(source, int):
            source_id = f"webcam_{source}"
        else:
            source_id = f"source_{hash(str(source)) % 10000}"

    # Initialize config if not provided
    if config is None:
        config = {}

    # Detect source type
    source_type = detect_source_type(source)

    # Create appropriate source based on type
    if source_type == SourceType.IMAGE:
        # Set source_path in config
        config["source_path"] = source
        return ImageSource(source_id, config)

    elif source_type == SourceType.FILE:
        # Set file_path in config
        config["file_path"] = source
        return VideoSource(source_id, config)

    elif source_type == SourceType.WEBCAM:
        # Set device_id in config
        config["device_id"] = source
        return WebcamSource(source_id, config)

    elif source_type == SourceType.IP_CAMERA:
        # Set url in config
        config["url"] = source
        return IPCameraSource(source_id, config)

    elif source_type == SourceType.MIPI_CAMERA:
        # Parse MIPI camera configuration
        if isinstance(source, str):
            if source.startswith("/dev/video"):
                # Extract device number
                device_num = int(source.split("video")[-1])
                config["sensor_id"] = device_num
            elif source.startswith("v4l2://"):
                config["pipeline_type"] = "v4l2"
                config["custom_pipeline"] = source
            elif source.startswith("csi://"):
                config["pipeline_type"] = "jetson"
                config["custom_pipeline"] = source
        return MIPICameraSource(source_id, config)

    elif source_type == SourceType.FOLDER:
        # Set folder_path in config
        config["folder_path"] = source
        return FolderSource(source_id, config)

    elif source_type == SourceType.MULTI:
        # Handle multi-source configuration
        if isinstance(source, list):
            config["sources"] = source
        elif isinstance(source, dict) and "sources" in source:
            config.update(source)
        return MultiSourceManager(source_id, config)

    else:
        raise ValueError(f"Unsupported source type: {source_type}")


# Legacy functions for backward compatibility
def is_file(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """Check if the source is a file (image or video)."""
    return is_image_file(source) or is_video_file(source)


def is_link(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """Check if the source is a network link."""
    return is_url(source)


def is_multi(source: Union[str, int, Dict[str, Any], Path]) -> bool:
    """Check if the source is a multi source."""
    return is_multi_source(source)


__all__ = [
    "BaseSource",
    "SourceType",
    "ImageSource",
    "VideoSource",
    "FileSource",  # Backward compatibility
    "FolderSource",
    "WebcamSource",
    "IPCameraSource",
    "MIPICameraSource",
    "MultiSourceManager",
    "create_source",
    "detect_source_type",
    "is_image_file",
    "is_image_folder",
    "is_media_folder",
    "is_image_url",
    "is_video_file",
    "is_video_url",
    "is_webcam",
    "is_ip_camera",
    "is_mipi_camera",
    "is_multi_source",
    "is_url",
    # Legacy functions
    "is_file",
    "is_link",
    "is_multi",
]


if __name__ == "__main__":
    from hailo_toolbox.cli.config import parse_args

    args = parse_args()
    source = create_source(args.source)
    print(f"Created source: {source.get_info()}")

    for frame in source:
        print(f"Frame shape: {frame.shape}")
        break  # Just test one frame
