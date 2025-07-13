from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.inference import CALLBACK_REGISTRY
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
import yaml

from hailo_toolbox.process.results.depth_estimation import DepthEstimationResult


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


@CALLBACK_REGISTRY.registryPostProcessor("fast_depth")
class EstimationPostprocessor(BasePostprocessor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def postprocess(
        self,
        preds: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> List[DepthEstimationResult]:
        results = []

        for output_name, pred in preds.items():

            # 确保预测输出的shape符合预期 [B, H, W, 1]
            if len(pred.shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor with shape [B, H, W, 1], got shape {pred.shape}"
                )

            batch_size, height, width, channels = pred.shape

            if channels != 1:
                raise ValueError(f"Expected depth channel to be 1, got {channels}")

            # 为每个batch创建一个DepthEstimationResult
            for batch_idx in range(batch_size):
                # 提取单个batch的深度数据，去掉最后一个维度 [H, W]
                depth_map = pred[batch_idx, :, :, 0]

                # 如果没有提供original_shape，使用深度图的shape
                if original_shape is None:
                    shape = (height, width)
                else:
                    shape = original_shape
                # 创建DepthEstimationResult对象
                result = DepthEstimationResult(depth=depth_map, original_shape=shape)
                results.append(result)

        return results


@CALLBACK_REGISTRY.registryPostProcessor("scdepthv3")
class EstimationPostprocessor(BasePostprocessor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def postprocess(
        self,
        preds: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> List[DepthEstimationResult]:
        results = []

        for output_name, pred in preds.items():

            # 确保预测输出的shape符合预期 [B, H, W, 1]
            if len(pred.shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor with shape [B, H, W, 1], got shape {pred.shape}"
                )
            batch_size, height, width, channels = pred.shape

            if channels != 1:
                raise ValueError(f"Expected depth channel to be 1, got {channels}")

            # 为每个batch创建一个DepthEstimationResult
            for batch_idx in range(batch_size):
                # 提取单个batch的深度数据，去掉最后一个维度 [H, W]
                depth_map = pred[batch_idx, :, :, 0]
                depth_map = np.reciprocal(_sigmoid(depth_map) * 10 + 0.009)

                # 如果没有提供original_shape，使用深度图的shape
                if original_shape is None:
                    shape = (height, width)
                else:
                    shape = original_shape

                # 创建DepthEstimationResult对象
                result = DepthEstimationResult(depth=depth_map, original_shape=shape)
                results.append(result)

        return results
