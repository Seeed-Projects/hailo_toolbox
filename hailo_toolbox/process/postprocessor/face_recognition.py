from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.process.results.face_recognition import FaceRecognitionResult
from hailo_toolbox.inference import CALLBACK_REGISTRY
from typing import Dict, Tuple, Optional
import numpy as np


@CALLBACK_REGISTRY.registryPostProcessor("arcface_mbnet", "arcface_r50", "lprnet")
class FaceRecognitionPostProcessor(BasePostprocessor):
    def __init__(self):
        super().__init__()

    def postprocess(
        self,
        predictions: Dict[str, np.ndarray],
        original_shape: Tuple[int, int],
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> FaceRecognitionResult:
        results = []
        for key, value in predictions.items():
            # print(key, value.shape)
            for i in range(value.shape[0]):
                results.append(FaceRecognitionResult(value[i], original_shape))
        return results
