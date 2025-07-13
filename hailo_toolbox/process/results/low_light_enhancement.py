from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class LowLightEnhancementResult:
    enhanced_image: np.ndarray
    original_shape: Tuple[int, int]

    def get_enhanced_image(self) -> np.ndarray:
        return (self.enhanced_image.clip(0, 1) * 255.0).astype(np.uint8)

    def get_original_shape(self) -> Tuple[int, int]:
        return self.original_shape
