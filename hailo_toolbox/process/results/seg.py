from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Result:
    """
    Result structure for segmentation.
    """

    boxes: Tuple[int, int, int, int]  # Shape: (4) - [x1, y1, x2, y2]
    score: float  # Shape: (1) - confidence scores
    class_id: int  # Shape: (1) - class indices
    masks: Optional[np.ndarray] = None  # Shape: (H, W) - instance masks

    def get_boxes_xyxy(self) -> Tuple[int, int, int, int]:
        return self.boxes

    def get_score(self) -> float:
        return self.score

    def get_class_id(self) -> int:
        return self.class_id

    def get_mask(self) -> np.ndarray:
        return self.masks

    def __repr__(self):
        return f"Result(boxes={self.boxes}, score={self.score}, class_id={self.class_id}), masks_shape={self.masks.shape if self.masks is not None else None}"


@dataclass
class SegmentationResult:
    """
    Result structure for segmentation.
    """

    masks: np.ndarray  # Shape: (N, H, W) or (H, W) for semantic segmentation
    scores: Optional[np.ndarray] = None  # Shape: (N,) - confidence scores
    class_ids: Optional[np.ndarray] = None  # Shape: (N,) - class indices
    boxes: Optional[np.ndarray] = None  # Shape: (N, 4) - bounding boxes
    index: int = 0
    input_shape: Optional[Tuple[int, int]] = None
    original_shape: Optional[Tuple[int, int]] = None

    def __len__(self) -> int:
        return len(self.masks) if len(self.masks.shape) == 3 else 1

    def __post_init__(self):
        self.boxes[:, [0, 2]] *= self.original_shape[1]
        self.boxes[:, [1, 3]] *= self.original_shape[0]

    def get_boxes_xyxy(self) -> np.ndarray:
        return self.boxes

    def get_scores(self) -> np.ndarray:
        return self.scores

    def get_class_ids(self) -> np.ndarray:
        return self.class_ids

    def get_masks(self) -> np.ndarray:
        return self.masks

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            self.index = 0
            raise StopIteration
        else:
            value = Result(
                self.boxes[self.index],
                self.scores[self.index],
                self.class_ids[self.index],
                self.masks[self.index] if self.masks is not None else None,
            )
            self.index += 1
            return value

    def __getitem__(self, index: int) -> Result:
        return Result(
            self.boxes[index],
            self.scores[index],
            self.class_ids[index],
            self.masks[index] if self.masks is not None else None,
        )
