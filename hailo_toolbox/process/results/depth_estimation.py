from dataclasses import dataclass

import numpy as np
from typing import Tuple


@dataclass
class DepthEstimationResult:
    depth: np.ndarray
    original_shape: Tuple[int, int]
    input_shape: Tuple[int, int]

    def get_depth(self) -> np.ndarray:
        return self.depth.astype(np.uint8)

    def get_depth_normalized(self) -> np.ndarray:
        """
        Get normalized depth map suitable for visualization and saving
        Normalize depth values to uint8 format in 0-255 range
        """
        depth = self.depth.copy()

        # Handle invalid values (if any)
        if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to 0-255 range
        depth_min = np.min(depth)
        depth_max = np.max(depth)

        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)

        # Convert to uint8 format
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        return depth_uint8

    def get_original_shape(self) -> Tuple[int, int]:
        return self.original_shape

    def get_input_shape(self) -> Tuple[int, int]:
        return self.input_shape
