from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.inference import CALLBACK_REGISTRY
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
import yaml
import cv2
import io
from hailo_toolbox.process.results.depth_estimation import DepthEstimationResult
import matplotlib.pyplot as plt


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
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> List[DepthEstimationResult]:
        results = []

        for output_name, pred in preds.items():

            # Ensure prediction output shape meets expectation [B, H, W, 1]
            if len(pred.shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor with shape [B, H, W, 1], got shape {pred.shape}"
                )

            batch_size, height, width, channels = pred.shape

            if channels != 1:
                raise ValueError(f"Expected depth channel to be 1, got {channels}")

            # Create a DepthEstimationResult for each batch
            for batch_idx in range(batch_size):
                # Extract depth data for single batch, remove last dimension [H, W]
                depth_map = pred[batch_idx, :, :, 0]
                # If original_shape is not provided, use depth map shape
                if original_shape is None:
                    shape = (height, width)
                else:
                    shape = original_shape
                # Create DepthEstimationResult object
                result = DepthEstimationResult(
                    depth=depth_map,
                    original_shape=shape,
                    input_shape=input_shape,
                )
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
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> List[DepthEstimationResult]:
        results = []

        for output_name, pred in preds.items():

            # Ensure prediction output shape meets expectation [B, H, W, 1]
            if len(pred.shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor with shape [B, H, W, 1], got shape {pred.shape}"
                )
            batch_size, height, width, channels = pred.shape

            if channels != 1:
                raise ValueError(f"Expected depth channel to be 1, got {channels}")

            # Create a DepthEstimationResult for each batch
            for batch_idx in range(batch_size):
                # Extract depth data for single batch, remove last dimension [H, W]
                depth_map = pred[batch_idx, :, :, 0]
                depth_map = np.reciprocal(_sigmoid(depth_map) * 10 + 0.009)

                # If original_shape is not provided, use depth map shape
                if original_shape is None:
                    shape = (height, width)
                else:
                    shape = original_shape

                # Create DepthEstimationResult object
                result = DepthEstimationResult(
                    depth=depth_map, original_shape=shape, input_shape=input_shape
                )
                results.append(result)

        return results
