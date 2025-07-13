from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class ImageDenoisingResult:
    denoised_image: np.ndarray
    original_shape: Tuple[int, int]

    def get_denoised_image(self) -> np.ndarray:
        return (self.denoised_image.clip(0, 1) * 255.0).astype(np.uint8)

    def get_original_shape(self) -> Tuple[int, int]:
        return self.original_shape
