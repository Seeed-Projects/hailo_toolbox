from dataclasses import dataclass
import numpy as np
from typing import Tuple, List


@dataclass
class VideoClassificationResult:
    class_name_top5: List[str]
    class_index_top5: List[int]
    score_top5: List[float]
    original_shape: Tuple[int, int]

    def get_class_name_top5(self) -> List[str]:
        return self.class_name_top5

    def get_class_index_top5(self) -> List[int]:
        return self.class_index_top5

    def get_score_top5(self) -> List[float]:
        return self.score_top5

    def get_original_shape(self) -> Tuple[int, int]:
        return self.original_shape
