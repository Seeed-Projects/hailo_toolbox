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


class BaseClassifyPostprocessor(BasePostprocessor):
    imagenet_classes = load_imagenet_classes()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def postprocess(self, results: Dict[str, np.ndarray]) -> List[ImageNetResult]:
        """
        基础后处理方法，子类应该重写此方法

        Args:
            results: 模型输出结果字典

        Returns:
            List[ImageNetResult]: 处理后的结果列表
        """
        # 这是一个基础实现，子类应该重写
        # 这里只是一个占位符实现
        raise NotImplementedError("子类必须实现 postprocess 方法")


@CALLBACK_REGISTRY.registryPostProcessor("mobilenetv1", "resnet18")
class ClassifyPostprocessor(BaseClassifyPostprocessor):
    with_softmax = False
    top_k = 5

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def postprocess(
        self,
        preds: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> List[ImageNetResult]:
        """
        处理分类模型的输出

        Args:
            preds: 字典，key为输出层名称，value为形状[B, number_class]的numpy数组
            original_shape: 原始图像尺寸（可选）

        Returns:
            List[ImageNetResult]: 每个batch样本对应一个ImageNetResult对象
        """
        results = []

        for k, multibatch_results in preds.items():
            print(f"Processing output '{k}' with shape: {multibatch_results.shape}")

            # 处理每个batch中的样本
            for batch_idx, v in enumerate(multibatch_results):
                # 如果需要，应用softmax
                if self.with_softmax:
                    v = _softmax(v)

                # 获取top k结果的索引（从小到大排序，取最后k个）
                top_k_indices = np.argsort(v)[-self.top_k :]
                # 反转顺序，使得最高分在前
                top_k_indices = top_k_indices[::-1]

                # 获取对应的分数和类别ID
                top_k_scores = v[top_k_indices]
                top_k_classes = top_k_indices

                # 获取类别名称
                top_k_names = []
                for class_id in top_k_classes:
                    class_name = self.imagenet_classes.get(
                        class_id, f"Unknown_{class_id}"
                    )
                    top_k_names.append(class_name)

                # 创建ImageNetResult对象
                result = ImageNetResult(
                    class_id_top5=top_k_classes,
                    score_top5=top_k_scores,
                    class_name_top5=top_k_names,
                )

                results.append(result)

        return results
