from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.inference import CALLBACK_REGISTRY
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
import yaml

from hailo_toolbox.process.results.classify import ImageNetResult


def _softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    return np.exp(x - np.max(x, axis=dim, keepdims=True)) / np.sum(
        np.exp(x - np.max(x, axis=dim, keepdims=True)), axis=dim, keepdims=True
    )


def load_imagenet_classes() -> Dict[int, str]:
    with open("examples/ImageNet.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


@CALLBACK_REGISTRY.registryPostProcessor("mobilenetv1", "resnet18")
class ClassifyPostprocessor(BasePostprocessor):
    with_softmax = False
    top_k = 5
    imagenet_classes = load_imagenet_classes()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def postprocess(
        self,
        preds: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> List[ImageNetResult]:
        """
        Process classification model outputs

        Args:
            preds: Dictionary with key as output layer name, value as numpy array with shape [B, number_class]
            original_shape: Original image size (optional)

        Returns:
            List[ImageNetResult]: Each batch sample corresponds to one ImageNetResult object
        """
        results = []

        for k, multibatch_results in preds.items():
            # print(f"Processing output '{k}' with shape: {multibatch_results.shape}")

            # Process each sample in the batch
            for batch_idx, v in enumerate(multibatch_results):
                # Apply softmax if needed
                if self.with_softmax:
                    v = _softmax(v)

                # Get top k result indices (sorted from small to large, take the last k)
                top_k_indices = np.argsort(v)[-self.top_k :]
                # Reverse order so highest score comes first
                top_k_indices = top_k_indices[::-1]

                # Get corresponding scores and class IDs
                top_k_scores = v[top_k_indices]
                top_k_classes = top_k_indices

                # Get class names
                top_k_names = []
                for class_id in top_k_classes:
                    class_name = self.imagenet_classes.get(
                        class_id, f"Unknown_{class_id}"
                    )
                    top_k_names.append(class_name)

                # Create ImageNetResult object
                result = ImageNetResult(
                    class_id_top5=top_k_classes,
                    score_top5=top_k_scores,
                    class_name_top5=top_k_names,
                )

                results.append(result)

        return results
