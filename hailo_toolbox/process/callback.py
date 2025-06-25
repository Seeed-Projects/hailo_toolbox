from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
from hailo_toolbox.inference.core import CALLBACK_REGISTRY


class CallbackType(Enum):
    AFTER_INFER = "after_infer"
    AFTER_POSTPROCESS = "after_postprocess"
    AFTER_PREPROCESS = "after_preprocess"
    BEFORE_INFER = "before_infer"
    BEFORE_POSTPROCESS = "before_postprocess"
    BEFORE_PREPROCESS = "before_preprocess"


class BaseCallback(ABC):
    def __init__(self, callback_type: CallbackType):
        self.callback_type = callback_type

    @abstractmethod
    def __call__(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        pass


class YOLOv8DetCallback(BaseCallback):
    def __init__(self, callback_type: CallbackType):
        super().__init__(callback_type)

    def __call__(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        pass


@CALLBACK_REGISTRY.registryCollateInfer("yolov8det", "yolo11det")
def yolov8_det_callback(results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    results_new = {}
    for key, value in results.items():
        data_new = []
        for batch in value:
            batch_new = []
            for c, box in enumerate(batch):
                if box.shape[0] > 0:
                    box = np.hstack((box, np.full((box.shape[0], 1), c)))
                    batch_new.append(box)
            batch_new = (
                np.concatenate(batch_new, axis=0) if batch_new else np.empty((0, 6))
            )
        data_new.append(batch_new)
        max_dim = max([box.shape[0] for box in data_new])
        padded_array = []
        for data in data_new:
            padded_array.append(
                np.vstack(
                    (
                        data,
                        np.zeros((max_dim - data.shape[0], data.shape[1])),
                    )
                )
            )
        results_new[key] = np.stack(padded_array, axis=0)
    return results_new
