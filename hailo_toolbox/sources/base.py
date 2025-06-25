"""
Base class for video sources.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, Iterator, List, Generator


class SourceType(Enum):
    """Enum for different types of video sources."""

    FILE = auto()
    WEBCAM = auto()
    IP_CAMERA = auto()
    MIPI_CAMERA = auto()
    MULTI = auto()
    CUSTOM = auto()
    IMAGE = auto()
    FOLDER = auto()


class BaseSource(ABC):
    """
    Abstract base class for all video sources.

    This class defines the common interface that all video sources must implement.
    """

    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base source.

        Args:
            source_id: Unique identifier for this source.
            config: Optional configuration dictionary.
        """
        self.source_id = source_id
        self.config = config or {}
        self.is_opened = False
        self.source_type = None
        self.fps = self.config.get("fps", 30)
        self.resolution = self.config.get("resolution", (640, 480))
        self.format = self.config.get("format", "BGR")

    @abstractmethod
    def open(self) -> bool:
        """
        Open the video source.

        Returns:
            True if opened successfully, False otherwise.
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video source.

        Returns:
            Tuple of (success, frame) where:
            - success: True if read was successful, False otherwise.
            - frame: The frame as a numpy array, or None if unsuccessful.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the video source."""
        pass

    def is_open(self) -> bool:
        """
        Check if the source is currently open.

        Returns:
            True if the source is open, False otherwise.
        """
        return self.is_opened

    def get_source_type(self) -> SourceType:
        """
        Get the type of video source.

        Returns:
            SourceType enum indicating the type of source.
        """
        return self.source_type

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the video source.

        Returns:
            Dictionary with information about the video source.
        """
        return {
            "source_id": self.source_id,
            "type": self.source_type,
            "is_open": self.is_opened,
            "fps": self.fps,
            "resolution": self.resolution,
            "format": self.format,
        }

    def __next__(self):
        """
        Stream frames from the source.

        Yields:
            Frames from the source as numpy arrays.
        """
        if not self.is_opened:
            self.open()

        success, frame = self.read()
        if not success:
            self.close()
            raise StopIteration
        return frame

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __iter__(self):
        """Make the source iterable."""
        return self

    @staticmethod
    def create_source(
        source_type: SourceType, source_id: str, config: Optional[Dict[str, Any]] = None
    ) -> "BaseSource":
        """
        Factory method to create a source based on type.

        Args:
            source_type: Type of source to create.
            source_id: Unique identifier for the source.
            config: Optional configuration dictionary.

        Returns:
            Instance of a BaseSource subclass.
        """
        from .file import FileSource, VideoSource, ImageSource
        from .webcam import WebcamSource
        from .ip_camera import IPCameraSource
        from .mipi import MIPICameraSource

        if source_type == SourceType.FILE:
            return VideoSource(source_id, config)
        elif source_type == SourceType.IMAGE:
            return ImageSource(source_id, config)
        elif source_type == SourceType.WEBCAM:
            return WebcamSource(source_id, config)
        elif source_type == SourceType.IP_CAMERA:
            return IPCameraSource(source_id, config)
        elif source_type == SourceType.MIPI_CAMERA:
            return MIPICameraSource(source_id, config)
        else:
            raise ValueError(f"Unknown source type: {source_type}")


if __name__ == "__main__":
    source = BaseSource.create_source(SourceType.FILE, "test")
    print(source.get_info())
    for frame in source:
        print(frame.shape)
