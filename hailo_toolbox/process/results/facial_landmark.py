from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class FacialLandmarkResult:
    landmarks: np.ndarray
    original_shape: Tuple[int, int]

    def get_landmarks(self) -> np.ndarray:
        return self.landmarks
        # self.landmarks = self.landmarks.reshape(-1, 2)
        # self.landmarks[:, 0] = self.landmarks[:, 0] * self.original_shape[1]
        # self.landmarks[:, 1] = self.landmarks[:, 1] * self.original_shape[0]
        # return self.landmarks

    def get_original_shape(self):
        return self.original_shape
