"""
Preprocessing pipeline for chaining multiple transforms together.

This module provides a pipeline class that allows combining multiple
preprocessing transforms into a single, reusable preprocessing pipeline.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
import time
from pathlib import Path

from .transforms import BaseTransform
from ..exceptions import PreprocessError, InvalidConfigError


class PreprocessPipeline:
    """
    Pipeline for chaining multiple preprocessing transforms.

    This class allows you to combine multiple transforms into a single
    preprocessing pipeline that can be applied to images. The pipeline
    maintains the order of transforms and provides utilities for
    configuration management and performance monitoring.
    """

    def __init__(
        self,
        transforms: Optional[List[BaseTransform]] = None,
        name: Optional[str] = None,
        enable_timing: bool = False,
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            transforms (Optional[List[BaseTransform]]): List of transforms to apply
            name (Optional[str]): Optional name for the pipeline
            enable_timing (bool): Whether to enable timing measurements
        """
        self.transforms = transforms or []
        self.name = name or "PreprocessPipeline"
        self.enable_timing = enable_timing
        self.timing_stats = {}
        self._validate_transforms()

    def _validate_transforms(self) -> None:
        """
        Validate that all transforms are instances of BaseTransform.

        Raises:
            InvalidConfigError: If any transform is invalid
        """
        for i, transform in enumerate(self.transforms):
            if not isinstance(transform, BaseTransform):
                raise InvalidConfigError(
                    f"Transform at index {i} is not a BaseTransform instance",
                    config_field="transforms",
                    provided_value=type(transform),
                )

    def add_transform(self, transform: BaseTransform) -> "PreprocessPipeline":
        """
        Add a transform to the pipeline.

        Args:
            transform (BaseTransform): Transform to add

        Returns:
            PreprocessPipeline: Self for method chaining

        Raises:
            InvalidConfigError: If transform is invalid
        """
        if not isinstance(transform, BaseTransform):
            raise InvalidConfigError(
                "Transform must be a BaseTransform instance",
                config_field="transform",
                provided_value=type(transform),
            )

        self.transforms.append(transform)
        return self

    def remove_transform(self, index: int) -> "PreprocessPipeline":
        """
        Remove a transform from the pipeline by index.

        Args:
            index (int): Index of transform to remove

        Returns:
            PreprocessPipeline: Self for method chaining

        Raises:
            InvalidConfigError: If index is invalid
        """
        if not 0 <= index < len(self.transforms):
            raise InvalidConfigError(
                f"Invalid transform index: {index}",
                config_field="index",
                provided_value=index,
            )

        self.transforms.pop(index)
        return self

    def insert_transform(
        self, index: int, transform: BaseTransform
    ) -> "PreprocessPipeline":
        """
        Insert a transform at a specific position in the pipeline.

        Args:
            index (int): Position to insert the transform
            transform (BaseTransform): Transform to insert

        Returns:
            PreprocessPipeline: Self for method chaining

        Raises:
            InvalidConfigError: If parameters are invalid
        """
        if not isinstance(transform, BaseTransform):
            raise InvalidConfigError(
                "Transform must be a BaseTransform instance",
                config_field="transform",
                provided_value=type(transform),
            )

        if not 0 <= index <= len(self.transforms):
            raise InvalidConfigError(
                f"Invalid insert index: {index}",
                config_field="index",
                provided_value=index,
            )

        self.transforms.insert(index, transform)
        return self

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessing pipeline to an image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Processed image

        Raises:
            PreprocessError: If any transform in the pipeline fails
        """
        if len(self.transforms) == 0:
            return image.copy()

        processed_image = image

        for i, transform in enumerate(self.transforms):
            try:
                if self.enable_timing:
                    start_time = time.time()

                processed_image = transform(processed_image)

                if self.enable_timing:
                    elapsed_time = time.time() - start_time
                    transform_name = transform.name

                    if transform_name not in self.timing_stats:
                        self.timing_stats[transform_name] = {
                            "total_time": 0.0,
                            "call_count": 0,
                            "avg_time": 0.0,
                        }

                    stats = self.timing_stats[transform_name]
                    stats["total_time"] += elapsed_time
                    stats["call_count"] += 1
                    stats["avg_time"] = stats["total_time"] / stats["call_count"]

            except Exception as e:
                if isinstance(e, PreprocessError):
                    raise
                raise PreprocessError(
                    f"Pipeline failed at transform {i} ({transform.name}): {str(e)}",
                    details={
                        "transform_index": i,
                        "transform_name": transform.name,
                        "transform_config": transform.get_config(),
                    },
                )

        return processed_image

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply the preprocessing pipeline to a batch of images.

        Args:
            images (List[np.ndarray]): List of input images

        Returns:
            List[np.ndarray]: List of processed images

        Raises:
            PreprocessError: If processing fails for any image
        """
        processed_images = []

        for i, image in enumerate(images):
            try:
                processed_image = self(image)
                processed_images.append(processed_image)
            except Exception as e:
                raise PreprocessError(
                    f"Batch processing failed at image {i}: {str(e)}",
                    details={"image_index": i, "batch_size": len(images)},
                )

        return processed_images

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the entire pipeline.

        Returns:
            Dict[str, Any]: Pipeline configuration
        """
        return {
            "name": self.name,
            "enable_timing": self.enable_timing,
            "transforms": [transform.get_config() for transform in self.transforms],
        }

    def save_config(self, filepath: Union[str, Path]) -> None:
        """
        Save the pipeline configuration to a JSON file.

        Args:
            filepath (Union[str, Path]): Path to save the configuration

        Raises:
            PreprocessError: If saving fails
        """
        try:
            config = self.get_config()
            filepath = Path(filepath)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise PreprocessError(
                f"Failed to save pipeline configuration: {str(e)}",
                details={"filepath": str(filepath)},
            )

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics for all transforms in the pipeline.

        Returns:
            Dict[str, Dict[str, float]]: Timing statistics by transform name
        """
        return self.timing_stats.copy()

    def reset_timing_stats(self) -> None:
        """Reset all timing statistics."""
        self.timing_stats.clear()

    def print_timing_stats(self) -> None:
        """Print timing statistics in a formatted table."""
        if not self.timing_stats:
            print("No timing statistics available. Enable timing to collect stats.")
            return

        # print(f"\nTiming Statistics for Pipeline: {self.name}")
        # print("-" * 70)
        # print(f"{'Transform':<25} {'Calls':<8} {'Total (s)':<12} {'Avg (s)':<12}")
        # print("-" * 70)

        total_time = 0.0
        total_calls = 0

        for transform_name, stats in self.timing_stats.items():
            print(
                f"{transform_name:<25} {stats['call_count']:<8} "
                f"{stats['total_time']:<12.4f} {stats['avg_time']:<12.4f}"
            )
            total_time += stats["total_time"]
            total_calls += stats["call_count"]

        # print("-" * 70)
        avg_total = total_time / total_calls if total_calls > 0 else 0.0
        # print(f"{'TOTAL':<25} {total_calls:<8} {total_time:<12.4f} {avg_total:<12.4f}")
        # print("-" * 70)

    def __len__(self) -> int:
        """Return the number of transforms in the pipeline."""
        return len(self.transforms)

    def __getitem__(self, index: int) -> BaseTransform:
        """Get a transform by index."""
        return self.transforms[index]

    def __iter__(self):
        """Iterate over transforms in the pipeline."""
        return iter(self.transforms)

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        transform_names = [t.name for t in self.transforms]
        return f"PreprocessPipeline(name='{self.name}', transforms={transform_names})"
