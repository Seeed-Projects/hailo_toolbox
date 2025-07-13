from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class HandLandmarkResult:
    landmarks: np.ndarray
    original_shape: Tuple[int, int]

    def get_landmarks(self) -> np.ndarray:
        return self.landmarks
