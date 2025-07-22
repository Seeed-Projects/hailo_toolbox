from typing import Optional, Tuple, Dict
from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.process.results.super_resolution import SuperResolutionResult
from hailo_toolbox.inference.core import CALLBACK_REGISTRY
import numpy as np


@CALLBACK_REGISTRY.registryPostProcessor("real_esrgan")
class RealESRGANPostprocessor(BasePostprocessor):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

    def postprocess(
        self,
        preds: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        results = []
        for output_name, pred in preds.items():
            # print(f"Processing output '{output_name}' with shape: {pred.shape}")
            for p in pred:
                results.append(SuperResolutionResult(p, original_shape, input_shape))
            break
        return results
