from typing import Dict, Tuple, List
import numpy as np
from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.process.results.image_denoising import ImageDenoisingResult
from hailo_toolbox.inference import CALLBACK_REGISTRY


@CALLBACK_REGISTRY.registryPostProcessor("dncnn3", "dncnn_color_blind")
class ImageDenoisingPostprocessor(BasePostprocessor):
    def __init__(self):
        super().__init__()

    def postprocess(
        self, predictions: Dict[str, np.ndarray], original_shape: Tuple[int, int]
    ) -> List[ImageDenoisingResult]:
        results = []
        for key, value in predictions.items():
            print(key, value.shape)
            for v in value:
                results.append(ImageDenoisingResult(v, original_shape))
        return results
