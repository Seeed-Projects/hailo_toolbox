from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Result:
    """
    Result structure for keypoint detection.
    """

    keypoints: Tuple[int, int]  # Shape: (K, 2) - [x, y]
    score: float  # Shape: (1) - person confidence score
    boxes: Optional[Tuple[int, int, int, int]] = (
        None  # Shape: (4) - person bounding boxes
    )
    joint_scores: Optional[Tuple[float]] = (
        None  # Shape: (K, 1) - joint confidence scores
    )

    def get_keypoints(self) -> Tuple[int, int, int]:
        return self.keypoints

    def get_score(self) -> float:
        return self.score

    def get_boxes(self) -> Tuple[int, int, int, int]:
        return self.boxes

    def get_joint_scores(self) -> Tuple[float]:
        return self.joint_scores

    def __repr__(self):
        return f"Result(keypoints_shape={self.keypoints.shape if self.keypoints is not None else None}, score_shape={self.score.shape if self.score is not None else None}, boxes_shape={self.boxes.shape if self.boxes is not None else None}, joint_scores_shape={self.joint_scores.shape if self.joint_scores is not None else None})"


@dataclass
class KeypointResult:
    """
    Result structure for keypoint detection.
    """

    keypoints: np.ndarray  # Shape: (N, K, 2) - [x, y]
    scores: np.ndarray  # Shape: (N, 1) - person confidence scores
    boxes: Optional[np.ndarray] = None  # Shape: (N, 4) - person bounding boxes
    joint_scores: Optional[np.ndarray] = (
        None  # Shape: (N, K, 1) - joint confidence scores
    )
    index: int = 0

    def __len__(self) -> int:
        return len(self.keypoints)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            self.index = 0
            raise StopIteration
        else:
            print(
                self.keypoints.shape,
                self.scores.shape,
                self.boxes.shape,
                self.joint_scores.shape,
            )
            value = Result(
                self.keypoints[self.index],
                self.scores[self.index],
                self.boxes[self.index],
                self.joint_scores[self.index],
            )
            self.index += 1
            return value

    def __getitem__(self, index: int) -> Result:
        return Result(
            self.keypoints[index],
            self.scores[index],
            self.boxes[index],
            self.joint_scores[index],
        )
