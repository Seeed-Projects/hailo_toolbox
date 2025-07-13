from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional


@dataclass
class FaceDetectionResult:
    """人脸检测结果类"""

    boxes: np.ndarray  # Shape: (N, 4) - [x1, y1, x2, y2] (归一化坐标)
    scores: np.ndarray  # Shape: (N,) - confidence scores
    landmarks: Optional[np.ndarray] = (
        None  # Shape: (N, 10) - 5个关键点 [x1, y1, x2, y2, ...] (归一化坐标)
    )
    original_shape: Optional[Tuple[int, int]] = None  # 原始图像尺寸 (height, width)

    def __len__(self) -> int:
        """返回检测到的人脸数量"""
        return len(self.boxes)

    def get_boxes(self, pixel_coords: bool = False) -> np.ndarray:
        """获取边界框

        Args:
            pixel_coords: 如果为True，返回像素坐标；否则返回归一化坐标
        """
        if pixel_coords and self.original_shape is not None:
            return self._to_pixel_coords(self.boxes, is_landmarks=False)
        return self.boxes

    def get_scores(self) -> np.ndarray:
        """获取置信度分数"""
        return self.scores

    def get_landmarks(self, pixel_coords: bool = False) -> Optional[np.ndarray]:
        """获取人脸关键点

        Args:
            pixel_coords: 如果为True，返回像素坐标；否则返回归一化坐标
        """
        if self.landmarks is None:
            return None

        if pixel_coords and self.original_shape is not None:
            return self._to_pixel_coords(self.landmarks, is_landmarks=True)
        return self.landmarks

    def get_original_shape(self) -> Optional[Tuple[int, int]]:
        """获取原始图像尺寸"""
        return self.original_shape

    def _to_pixel_coords(
        self, coords: np.ndarray, is_landmarks: bool = False
    ) -> np.ndarray:
        """将归一化坐标转换为像素坐标"""
        if self.original_shape is None:
            return coords

        orig_h, orig_w = self.original_shape
        pixel_coords = coords.copy()

        if is_landmarks:
            # 关键点坐标：[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
            pixel_coords[:, 0::2] *= orig_w  # x坐标
            pixel_coords[:, 1::2] *= orig_h  # y坐标
        else:
            # 边界框坐标：[x1, y1, x2, y2]
            pixel_coords[:, [0, 2]] *= orig_w  # x坐标
            pixel_coords[:, [1, 3]] *= orig_h  # y坐标

        return pixel_coords

    def filter_by_confidence(self, threshold: float = 0.5) -> "FaceDetectionResult":
        """根据置信度阈值过滤结果"""
        if len(self.boxes) == 0:
            return self

        valid_mask = self.scores >= threshold
        if not np.any(valid_mask):
            return FaceDetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                landmarks=(
                    np.empty((0, 10), dtype=np.float32)
                    if self.landmarks is not None
                    else None
                ),
                original_shape=self.original_shape,
            )

        filtered_boxes = self.boxes[valid_mask]
        filtered_scores = self.scores[valid_mask]
        filtered_landmarks = (
            self.landmarks[valid_mask] if self.landmarks is not None else None
        )

        return FaceDetectionResult(
            boxes=filtered_boxes,
            scores=filtered_scores,
            landmarks=filtered_landmarks,
            original_shape=self.original_shape,
        )
