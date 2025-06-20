"""
Deep Learning Model Input Preprocessing Module

This module provides comprehensive preprocessing capabilities for deep learning model inputs,
including image resizing, normalization, data type conversion, and various augmentation options.

The module is designed with modularity and extensibility in mind, allowing easy customization
and extension for different model requirements.
"""

from .preprocessor.preprocessor import ImagePreprocessor, PreprocessConfig
from .preprocessor.transforms import (
    ResizeTransform,
    NormalizationTransform,
    DataTypeTransform,
    PaddingTransform,
    CropTransform,
    InterpolationMethod,
    PaddingMode,
)
from .preprocessor.pipeline import PreprocessPipeline
from .base import (
    BasePreprocessor,
    BasePostprocessor,
    PostprocessConfig,
    DetectionResult,
    SegmentationResult,
    KeypointResult,
    non_max_suppression,
    scale_boxes,
    scale_keypoints,
)
from .exceptions import (
    PreprocessError,
    InvalidConfigError,
    ImageProcessingError,
    UnsupportedFormatError,
)
from .postprocessor import (
    YOLOv8DetPostprocessor,
    YOLOv8SegPostprocessor,
    YOLOv8PosePostprocessor,
)
from .callback import *
from .visualization import *

# from .postprocessor_kp import YOLOv8KpPostprocessor


def create_preprocessor(config: PreprocessConfig) -> BasePreprocessor:
    """
    Create a preprocessor instance based on configuration.

    Args:
        config: Preprocessing configuration

    Returns:
        Preprocessor instance
    """
    return ImagePreprocessor(config)


def create_postprocessor(
    task_type: str, config: PostprocessConfig = None
) -> BasePostprocessor:
    """
    Create a postprocessor instance based on task type and configuration.

    Args:
        task_type: Type of task ('det', 'seg', 'kp')
        config: Postprocessing configuration

    Returns:
        Postprocessor instance

    Raises:
        ValueError: If task_type is not supported
    """
    if config is None:
        config = PostprocessConfig()

    if task_type.lower() in ["det", "detection"]:
        config.det = True
        return YOLOv8DetPostprocessor(config)
    elif task_type.lower() in ["seg", "segmentation", "instance_segmentation"]:
        config.seg = True
        return YOLOv8SegPostprocessor(config)
    elif task_type.lower() in ["kp", "keypoint", "pose", "pose_estimation"]:
        config.kp = True
        return YOLOv8PosePostprocessor(config)
    else:
        raise ValueError(
            f"Unsupported task type: {task_type}. "
            f"Supported types: 'det', 'seg', 'kp'"
        )


__all__ = [
    # Preprocessing
    "ImagePreprocessor",
    "PreprocessConfig",
    "ResizeTransform",
    "NormalizationTransform",
    "DataTypeTransform",
    "PaddingTransform",
    "CropTransform",
    "InterpolationMethod",
    "PaddingMode",
    "PreprocessPipeline",
    "create_preprocessor",
    # Postprocessing
    "BasePostprocessor",
    "PostprocessConfig",
    "DetectionResult",
    "SegmentationResult",
    "KeypointResult",
    "YOLOv8DetPostprocessor",
    "YOLOv8SegPostprocessor",
    "YOLOv8PosePostprocessor",
    "create_postprocessor",
    # Utility functions
    "non_max_suppression",
    "scale_boxes",
    "scale_keypoints",
    # Base classes
    "BasePreprocessor",
    # Exceptions
    "PreprocessError",
    "InvalidConfigError",
    "ImageProcessingError",
    "UnsupportedFormatError",
]
