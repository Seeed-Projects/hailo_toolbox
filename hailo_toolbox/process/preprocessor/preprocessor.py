"""
Main image preprocessor class and configuration management.

This module provides the main ImagePreprocessor class that serves as a
high-level interface for image preprocessing operations, along with
configuration management utilities.
"""

import cv2
import numpy as np
from typing import Union, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from .transforms import (
    ResizeTransform,
    NormalizationTransform,
    DataTypeTransform,
    PaddingTransform,
    CropTransform,
    InterpolationMethod,
)
from ..base import BasePreprocessor
from .pipeline import PreprocessPipeline
from ..exceptions import PreprocessError, InvalidConfigError, UnsupportedFormatError
from hailo_toolbox.inference.core import CALLBACK_REGISTRY


@dataclass
class PreprocessConfig:
    """
    Configuration class for image preprocessing operations.

    This dataclass provides a structured way to define preprocessing
    parameters with sensible defaults and validation.
    """

    # Resize parameters
    target_size: Optional[Tuple[int, int]] = (640, 640)
    interpolation: str = "LINEAR"
    preserve_aspect_ratio: bool = False

    # Normalization parameters
    normalize: bool = False
    mean: Union[float, List[float]] = 0.0
    std: Union[float, List[float]] = 1.0
    scale: float = 1.0

    # Data type parameters
    target_dtype: Optional[str] = None
    scale_values: bool = True
    clip_values: bool = True

    # Padding parameters
    padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]] = None
    padding_mode: str = "CONSTANT"
    padding_value: Union[int, float] = 0

    # Cropping parameters
    crop_size: Optional[Tuple[int, int]] = None
    crop_region: Optional[Tuple[int, int, int, int]] = None
    center_crop: bool = True

    # Pipeline parameters
    enable_timing: bool = False
    pipeline_name: str = "ImagePreprocessor"

    # Channel order and format
    input_format: str = "BGR"  # BGR, RGB, GRAY
    output_format: str = "RGB"  # BGR, RGB, GRAY

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            InvalidConfigError: If any configuration parameter is invalid
        """
        # Validate target_size
        if self.target_size is not None:
            if (
                not isinstance(self.target_size, (tuple, list))
                or len(self.target_size) != 2
            ):
                raise InvalidConfigError(
                    "target_size must be a tuple of (width, height)",
                    config_field="target_size",
                    provided_value=self.target_size,
                )
            if self.target_size[0] <= 0 or self.target_size[1] <= 0:
                raise InvalidConfigError(
                    "target_size dimensions must be positive",
                    config_field="target_size",
                    provided_value=self.target_size,
                )

        # Validate interpolation method
        valid_interpolations = ["NEAREST", "LINEAR", "CUBIC", "AREA", "LANCZOS4"]
        if self.interpolation.upper() not in valid_interpolations:
            raise InvalidConfigError(
                f"Invalid interpolation method. Valid options: {valid_interpolations}",
                config_field="interpolation",
                provided_value=self.interpolation,
            )

        # Validate normalization parameters
        if isinstance(self.std, (list, tuple)):
            if any(s == 0 for s in self.std):
                raise InvalidConfigError(
                    "Standard deviation values cannot be zero",
                    config_field="std",
                    provided_value=self.std,
                )
        elif self.std == 0:
            raise InvalidConfigError(
                "Standard deviation cannot be zero",
                config_field="std",
                provided_value=self.std,
            )

        # Validate format parameters
        valid_formats = ["BGR", "RGB", "GRAY"]
        if self.input_format.upper() not in valid_formats:
            raise InvalidConfigError(
                f"Invalid input format. Valid options: {valid_formats}",
                config_field="input_format",
                provided_value=self.input_format,
            )
        if self.output_format.upper() not in valid_formats:
            raise InvalidConfigError(
                f"Invalid output format. Valid options: {valid_formats}",
                config_field="output_format",
                provided_value=self.output_format,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "target_size": self.target_size,
            "interpolation": self.interpolation,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "normalize": self.normalize,
            "mean": self.mean,
            "std": self.std,
            "scale": self.scale,
            "target_dtype": self.target_dtype,
            "scale_values": self.scale_values,
            "clip_values": self.clip_values,
            "padding": self.padding,
            "padding_mode": self.padding_mode,
            "padding_value": self.padding_value,
            "crop_size": self.crop_size,
            "crop_region": self.crop_region,
            "center_crop": self.center_crop,
            "enable_timing": self.enable_timing,
            "pipeline_name": self.pipeline_name,
            "input_format": self.input_format,
            "output_format": self.output_format,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PreprocessConfig":
        """Create configuration from dictionary."""
        # Convert list fields back to tuples where needed
        config_dict = config_dict.copy()

        # Convert target_size from list to tuple if needed
        if config_dict.get("target_size") is not None:
            config_dict["target_size"] = tuple(config_dict["target_size"])

        # Convert crop_size from list to tuple if needed
        if config_dict.get("crop_size") is not None:
            config_dict["crop_size"] = tuple(config_dict["crop_size"])

        # Convert crop_region from list to tuple if needed
        if config_dict.get("crop_region") is not None:
            config_dict["crop_region"] = tuple(config_dict["crop_region"])

        # Convert padding from list to tuple if needed and it's not a single int
        if config_dict.get("padding") is not None:
            padding = config_dict["padding"]
            if isinstance(padding, list):
                config_dict["padding"] = tuple(padding)

        return cls(**config_dict)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath (Union[str, Path]): Path to save the configuration
        """
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PreprocessConfig":
        """
        Load configuration from JSON file.

        Args:
            filepath (Union[str, Path]): Path to load the configuration from

        Returns:
            PreprocessConfig: Loaded configuration
        """
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@CALLBACK_REGISTRY.registryPreProcessor("base")
class ImagePreprocessor(BasePreprocessor):
    """
    High-level image preprocessor with configurable pipeline.

    This class provides a convenient interface for image preprocessing
    operations commonly used in deep learning workflows. It automatically
    builds a preprocessing pipeline based on the provided configuration.
    """

    def __init__(
        self,
        config: Optional[PreprocessConfig] = None,
        target_size: Optional[Tuple[int, int]] = None,
        output_format: Optional[str] = None,
    ):
        """
        Initialize the image preprocessor.

        Args:
            config (Optional[PreprocessConfig]): Preprocessing configuration
        """
        self.config = config or PreprocessConfig()
        if target_size is not None:
            self.config.target_size = target_size
        if output_format is not None:
            self.config.output_format = output_format
        self.pipeline = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the preprocessing pipeline based on configuration."""
        transforms = []

        # Add format conversion (input format handling)
        if self.config.input_format.upper() != "BGR":
            # OpenCV loads images in BGR by default, so we need conversion
            # This will be handled in the __call__ method
            pass

        # Add cropping transform if specified
        if self.config.crop_size is not None or self.config.crop_region is not None:
            crop_transform = CropTransform(
                crop_size=self.config.crop_size,
                crop_region=self.config.crop_region,
                center_crop=self.config.center_crop,
                name="CropTransform",
            )
            transforms.append(crop_transform)

        # Add resize transform if specified
        if self.config.target_size is not None:
            resize_transform = ResizeTransform(
                target_size=self.config.target_size,
                interpolation=self.config.interpolation,
                preserve_aspect_ratio=self.config.preserve_aspect_ratio,
                name="ResizeTransform",
            )
            transforms.append(resize_transform)

        # Add padding transform if specified
        if self.config.padding is not None:
            padding_transform = PaddingTransform(
                padding=self.config.padding,
                mode=self.config.padding_mode,
                constant_value=self.config.padding_value,
                name="PaddingTransform",
            )
            transforms.append(padding_transform)

        # Add normalization transform if enabled
        if self.config.normalize:
            normalization_transform = NormalizationTransform(
                mean=self.config.mean,
                std=self.config.std,
                scale=self.config.scale,
                dtype=(
                    np.dtype(self.config.target_dtype)
                    if self.config.target_dtype
                    else None
                ),
                name="NormalizationTransform",
            )
            transforms.append(normalization_transform)

        # Add data type conversion if specified and not handled by normalization
        elif self.config.target_dtype is not None:
            dtype_transform = DataTypeTransform(
                target_dtype=self.config.target_dtype,
                scale_values=self.config.scale_values,
                clip_values=self.config.clip_values,
                name="DataTypeTransform",
            )
            transforms.append(dtype_transform)

        # Create pipeline
        self.pipeline = PreprocessPipeline(
            transforms=transforms,
            name=self.config.pipeline_name,
            enable_timing=self.config.enable_timing,
        )

    def _convert_color_format(
        self, image: np.ndarray, from_format: str, to_format: str
    ) -> np.ndarray:
        """
        Convert image color format.

        Args:
            image (np.ndarray): Input image
            from_format (str): Source color format
            to_format (str): Target color format

        Returns:
            np.ndarray: Converted image

        Raises:
            UnsupportedFormatError: If format conversion is not supported
        """
        if from_format == to_format:
            return image

        # Handle grayscale conversions
        if from_format == "GRAY":
            if to_format == "RGB":
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif to_format == "BGR":
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif to_format == "GRAY":
            if from_format == "RGB":
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif from_format == "BGR":
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Handle RGB/BGR conversions
        elif from_format == "BGR" and to_format == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif from_format == "RGB" and to_format == "BGR":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        raise UnsupportedFormatError(
            f"Conversion from {from_format} to {to_format} is not supported",
            format_type=f"{from_format} -> {to_format}",
            supported_formats=["BGR", "RGB", "GRAY"],
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to an image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Preprocessed image

        Raises:
            PreprocessError: If preprocessing fails
        """
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                raise PreprocessError(
                    f"Expected numpy array, got {type(image)}",
                    details={"input_type": type(image)},
                )

            if image.size == 0:
                raise PreprocessError(
                    "Empty image provided", details={"image_shape": image.shape}
                )

            processed_image = image.copy()

            # Handle input format conversion
            if len(processed_image.shape) == 3:
                processed_image = self._convert_color_format(
                    processed_image,
                    self.config.input_format.upper(),
                    "BGR",  # Convert to BGR for internal processing
                )

            # Apply preprocessing pipeline
            processed_image = self.pipeline(processed_image)

            # Handle output format conversion
            if len(processed_image.shape) == 3:
                processed_image = self._convert_color_format(
                    processed_image,
                    "BGR",  # From internal BGR format
                    self.config.output_format.upper(),
                )

            return processed_image

        except Exception as e:
            if isinstance(e, (PreprocessError, UnsupportedFormatError)):
                raise
            raise PreprocessError(
                f"Preprocessing failed: {str(e)}",
                details={
                    "input_shape": (
                        image.shape if isinstance(image, np.ndarray) else None
                    ),
                    "config": self.config.to_dict(),
                },
            )

    def preprocess(self, data: Any) -> Dict[str, np.ndarray]:
        """
        Preprocess input data for model inference.

        This method implements the abstract method from BasePreprocessor.
        It processes the input data and returns a dictionary with the preprocessed tensor.

        Args:
            data: Input data to preprocess (expected to be np.ndarray image)

        Returns:
            Dictionary mapping input names to preprocessed arrays

        Raises:
            PreprocessError: If preprocessing fails
        """
        try:
            # Process the image using the __call__ method
            processed_image = self(data)

            # Convert to tensor format (add batch dimension and transpose to NCHW if needed)
            if len(processed_image.shape) == 3:
                # Add batch dimension: (H, W, C) -> (1, H, W, C)
                processed_tensor = np.expand_dims(processed_image, axis=0)
                # Transpose to NCHW format: (1, H, W, C) -> (1, C, H, W)
                processed_tensor = np.transpose(processed_tensor, (0, 3, 1, 2))
            elif len(processed_image.shape) == 2:
                # Grayscale image: (H, W) -> (1, 1, H, W)
                processed_tensor = np.expand_dims(processed_image, axis=(0, 1))
            else:
                # Already in correct format
                processed_tensor = processed_image

            # Return as dictionary (common format for model inputs)
            return {"input": processed_tensor}

        except Exception as e:
            if isinstance(e, PreprocessError):
                raise
            raise PreprocessError(
                f"Preprocessing failed in preprocess method: {str(e)}",
                details={
                    "input_type": type(data),
                    "input_shape": data.shape if isinstance(data, np.ndarray) else None,
                },
            )

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of images.

        Args:
            images (List[np.ndarray]): List of input images

        Returns:
            List[np.ndarray]: List of preprocessed images
        """
        return [self(image) for image in images]

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters and rebuild pipeline.

        Args:
            **kwargs: Configuration parameters to update
        """
        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise InvalidConfigError(
                    f"Invalid configuration parameter: {key}",
                    config_field=key,
                    provided_value=value,
                )

        # Validate updated configuration
        self.config._validate_config()

        # Rebuild pipeline
        self._build_pipeline()

    def get_config(self) -> PreprocessConfig:
        """Get current configuration."""
        return self.config

    def get_pipeline(self) -> PreprocessPipeline:
        """Get the preprocessing pipeline."""
        return self.pipeline

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics from the pipeline."""
        return self.pipeline.get_timing_stats()

    def print_timing_stats(self) -> None:
        """Print timing statistics."""
        self.pipeline.print_timing_stats()

    def reset_timing_stats(self) -> None:
        """Reset timing statistics."""
        self.pipeline.reset_timing_stats()

    def save_config(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Args:
            filepath (Union[str, Path]): Path to save the configuration
        """
        self.config.save(filepath)

    @classmethod
    def from_config_file(cls, filepath: Union[str, Path]) -> "ImagePreprocessor":
        """
        Create preprocessor from configuration file.

        Args:
            filepath (Union[str, Path]): Path to configuration file

        Returns:
            ImagePreprocessor: Preprocessor instance
        """
        config = PreprocessConfig.load(filepath)
        return cls(config)

    def __repr__(self) -> str:
        """Return string representation of the preprocessor."""
        return f"ImagePreprocessor(config={self.config.pipeline_name}, transforms={len(self.pipeline)})"
