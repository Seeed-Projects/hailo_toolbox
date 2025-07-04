"""
Module for inference engines that run deep learning models.
"""

from typing import Optional
from .base import BaseInferenceEngine, InferenceResult, InferenceCallback
from .onnx_engine import ONNXInference
from .pipeline import InferencePipeline
from .core import CALLBACK_REGISTRY, InferenceEngine, InferencePostProcess
from hailo_toolbox.inference.hailo_engine import HailoInference
from hailo_toolbox.inference.onnx_engine import ONNXInference
import numpy as np


class Inference:

    def __init__(self, model_path: str, model_name: Optional[str] = None):
        self.model_path = model_path
        self.model_name = model_name
        self.inference_engine = None
        self.load_model()

    def load_model(self):
        if self.model_name is not None:
            pass
        else:
            if self.model_path.endswith(".hef"):
                self.inference_engine = HailoInference(self.model_path)
            elif self.model_path.endswith(".onnx"):
                self.inference_engine = ONNXInference(self.model_path)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.inference_engine.predict(input_data)

    def unload_model(self):
        self.inference_engine.stop_process()


def load_model(model_path, model_name: Optional[str] = None):
    return InferencePostProcess(model_path, model_name)


__all__ = [
    "BaseInferenceEngine",
    "InferenceResult",
    "InferenceCallback",
    "ONNXInference",
    "InferencePipeline",
    "CALLBACK_REGISTRY",
    "InferenceEngine",
]
