from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional


@dataclass
class FacialLandmarkResult:
    landmarks: np.ndarray
    original_shape: Tuple[int, int]
    input_shape: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        self.h_scale = self.original_shape[0] / self.input_shape[0]
        self.w_scale = self.original_shape[1] / self.input_shape[1]

    def get_landmarks(self) -> np.ndarray:
        return self.landmarks
        # self.landmarks = self.landmarks.reshape(-1, 2)
        # self.landmarks[:, 0] = self.landmarks[:, 0] * self.original_shape[1]
        # self.landmarks[:, 1] = self.landmarks[:, 1] * self.original_shape[0]
        # return self.landmarks

    def get_landmarks_with_original_shape(self) -> np.ndarray:
        return self.landmarks * np.array([self.w_scale, self.h_scale])

    def get_original_shape(self):
        return self.original_shape
