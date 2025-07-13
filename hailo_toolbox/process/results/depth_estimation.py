from dataclasses import dataclass

import numpy as np
from typing import Tuple


@dataclass
class DepthEstimationResult:
    depth: np.ndarray
    original_shape: Tuple[int, int]

    def get_depth(self) -> np.ndarray:
        return self.depth.astype(np.uint8)

    def get_depth_normalized(self) -> np.ndarray:
        """
        获取归一化的深度图，适合可视化和保存
        将深度值归一化到0-255范围内的uint8格式
        """
        depth = self.depth.copy()

        # 处理无效值（如果有的话）
        if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # 归一化到0-255范围
        depth_min = np.min(depth)
        depth_max = np.max(depth)

        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)

        # 转换为uint8格式
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        return depth_uint8
