from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.inference import CALLBACK_REGISTRY
from typing import Dict, Optional, Any, List, Tuple
from hailo_toolbox.process.results.hand_landmark import HandLandmarkResult
import numpy as np
import yaml


@CALLBACK_REGISTRY.registryPostProcessor("hand_landmark")
class HandLandmarkPostprocessor(BasePostprocessor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def postprocess(
        self,
        preds: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> List[HandLandmarkResult]:
        results = []
        for output_name, pred in preds.items():
            print(f"Processing output '{output_name}' with shape: {pred.shape}")
            if pred.shape[-1] != 63:
                continue
            for p in pred:
                results.append(HandLandmarkResult(p, original_shape))
            break
        return results
