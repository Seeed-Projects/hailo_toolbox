from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class FaceDetectionResult:
    """Face detection result class"""

    boxes: np.ndarray  # Shape: (N, 4) - [x1, y1, x2, y2] (normalized coordinates)
    scores: np.ndarray  # Shape: (N,) - confidence scores
    landmarks: Optional[np.ndarray] = (
        None  # Shape: (N, 10) - 5 landmarks [x1, y1, x2, y2, ...] (normalized coordinates)
    )
    original_shape: Optional[Tuple[int, int]] = (
        None  # Original image size (height, width)
    )

    def __len__(self) -> int:
        """Return the number of detected faces"""
        return len(self.boxes)

    def get_boxes(self, pixel_coords: bool = False) -> np.ndarray:
        """Get bounding boxes

        Args:
            pixel_coords: If True, return pixel coordinates; otherwise return normalized coordinates
        """
        if pixel_coords and self.original_shape is not None:
            h, w = self.original_shape
            boxes = self.boxes.copy()
            boxes[:, [0, 2]] *= w  # x coordinates
            boxes[:, [1, 3]] *= h  # y coordinates
            return boxes
        return self.boxes

    def get_scores(self) -> np.ndarray:
        """Get confidence scores"""
        return self.scores

    def get_landmarks(self, pixel_coords: bool = False) -> Optional[np.ndarray]:
        """Get face landmarks

        Args:
            pixel_coords: If True, return pixel coordinates; otherwise return normalized coordinates
        """
        if self.landmarks is None:
            return None

        if pixel_coords and self.original_shape is not None:
            h, w = self.original_shape
            landmarks = self.landmarks.copy()
            landmarks[:, 0::2] *= w  # x coordinates
            landmarks[:, 1::2] *= h  # y coordinates
            return landmarks
        return self.landmarks

    def get_original_shape(self) -> Optional[Tuple[int, int]]:
        """Get original image size"""
        return self.original_shape
