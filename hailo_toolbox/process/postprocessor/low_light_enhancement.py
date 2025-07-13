from typing import Dict, Tuple, List
import numpy as np
from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.process.results.low_light_enhancement import (
    LowLightEnhancementResult,
)
from hailo_toolbox.inference import CALLBACK_REGISTRY


@CALLBACK_REGISTRY.registryPostProcessor("zero_dce", "zero_dce_pp")
class LowLightEnhancementPostprocessor(BasePostprocessor):
    def __init__(self):
        super().__init__()

    def postprocess(
        self, predictions: Dict[str, np.ndarray], original_shape: Tuple[int, int]
    ) -> List[LowLightEnhancementResult]:
        results = []
        for key, value in predictions.items():
            print(key, value.shape)
            for v in value:
                results.append(LowLightEnhancementResult(v, original_shape))
        return results
