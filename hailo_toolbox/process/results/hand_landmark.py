from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class HandLandmarkResult:
    landmarks: np.ndarray
    original_shape: Tuple[int, int]
    input_shape: Tuple[int, int]

    def __post_init__(self):
        self.h_scale = self.original_shape[0] / self.input_shape[0]
        self.w_scale = self.original_shape[1] / self.input_shape[1]

    def get_landmarks(self) -> np.ndarray:
        return self.landmarks

    def get_landmarks_points(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            tuple([int(x), int(y)]) for x, y, _ in self.landmarks.reshape(-1, 3)
        )

    def get_landmarks_points_with_original_shape(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            tuple([int(x * self.w_scale), int(y * self.h_scale)])
            for x, y, _ in self.landmarks.reshape(-1, 3)
        )

    def get_original_shape(self) -> Tuple[int, int]:
        return self.original_shape
