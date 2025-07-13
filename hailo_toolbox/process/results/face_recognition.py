from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class FaceRecognitionResult:
    embeddings: np.ndarray
    original_shape: Tuple[int, int]

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings
