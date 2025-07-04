from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pydoc


@dataclass
class Result:
    """
    Result structure for object detection.
    """

    boxes: np.ndarray  # Shape: (4) - [x1, y1, x2, y2]
    scores: np.ndarray  # Shape: (1) - confidence scores
    class_ids: np.ndarray  # Shape: (1) - class indices
    masks: Optional[np.ndarray] = None  # Shape: (H, W) - instance masks

    def get_boxes_xyxy(self) -> np.ndarray:
        return self.boxes

    def get_scores(self) -> np.ndarray:
        return self.scores

    def get_class_ids(self) -> np.ndarray:
        return self.class_ids

    def __repr__(self):
        return f"Result(boxes={self.boxes}, scores={self.scores}, class_ids={self.class_ids})"


@dataclass
class DetectionResult:
    """
    Result structure for object detection.
    """

    boxes: np.ndarray  # Shape: (N, 4) - [x1, y1, x2, y2]
    scores: np.ndarray  # Shape: (N,) - confidence scores
    class_ids: np.ndarray  # Shape: (N,) - class indices
    masks: Optional[np.ndarray] = None  # Shape: (N, H, W) - instance masks
    index: int = 0

    def __len__(self) -> int:
        return len(self.boxes)

    def get_boxes(self) -> np.ndarray:
        return self.boxes

    def get_scores(self) -> np.ndarray:
        return self.scores

    def get_class_ids(self) -> np.ndarray:
        return self.class_ids

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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return Result(
            self.boxes[index],
            self.scores[index],
            self.class_ids[index],
            self.masks[index] if self.masks is not None else None,
        )


if __name__ == "__main__":
    result = DetectionResult(
        boxes=np.array([[100, 100, 200, 200], [0, 0, 100, 100]]),
        scores=np.array([0.9, 1]),
        class_ids=np.array([0, 1]),
    )
    for r in result:
        print(r.get_boxes())
        print(r.get_scores())
        print(r.get_class_ids())
