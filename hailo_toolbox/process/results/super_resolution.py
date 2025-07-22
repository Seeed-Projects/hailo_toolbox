from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class SuperResolutionResult:
    image: np.ndarray
    original_shape: Tuple[int, int]
    input_shape: Tuple[int, int]

    def get_image(self):
        return (self.image * 255).astype(np.uint8)[..., ::-1]

    def get_original_shape(self):
        return self.original_shape

    def get_input_shape(self):
        return self.input_shape
