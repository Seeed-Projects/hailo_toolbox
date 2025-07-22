from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.process.results.person_reid import PersonReIDResult
from hailo_toolbox.inference import CALLBACK_REGISTRY

from typing import Dict, Tuple, List
import numpy as np


@CALLBACK_REGISTRY.registryPostProcessor("osnet_x1", "repvgg_a0")
class PersonReIDPostprocessor(BasePostprocessor):
    def __init__(self):
        super().__init__()

    def postprocess(
        self,
        predictions: Dict[str, np.ndarray],
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
    ) -> List[PersonReIDResult]:
        results = []
        for key, value in predictions.items():
            # print(key, value.shape)
            for v in value:
                results.append(PersonReIDResult(v, original_shape))
        return results
