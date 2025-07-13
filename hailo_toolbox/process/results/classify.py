from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ImageNetResult:
    class_id_top5: np.ndarray
    score_top5: np.ndarray
    class_name_top5: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.class_id_top5)

    def get_score(self) -> float:
        return self.score_top5[0]

    def get_class_id(self) -> int:
        return self.class_id_top5[0]

    def get_class_name(self) -> str:
        return self.class_name_top5[0]

    def get_top_5_scores(self) -> Tuple[float]:
        return self.score_top5[:5]

    def get_top_5_class_ids(self) -> Tuple[int]:
        return self.class_id_top5[:5]

    def get_top_5_class_names(self) -> Tuple[str]:
        return self.class_name_top5[:5]
