from typing import Dict, Tuple, List, Optional
import numpy as np
import yaml
from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.process.results.video_classification import VideoClassificationResult
from hailo_toolbox.inference import CALLBACK_REGISTRY


def softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    return np.exp(x - np.max(x, axis=dim, keepdims=True)) / np.sum(
        np.exp(x - np.max(x, axis=dim, keepdims=True)), axis=dim, keepdims=True
    )


def load_label_map(label_map_path: str) -> Dict[int, str]:
    with open(label_map_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


@CALLBACK_REGISTRY.registryPostProcessor("r3d_18")
class VideoClassificationPostprocessor(BasePostprocessor):
    top_k = 5

    def __init__(self, label_map_path: Optional[str] = None):
        super().__init__()
        if label_map_path is not None:
            self.label_map = load_label_map(label_map_path)
        else:
            self.label_map = load_label_map("examples/kinetics400.yaml")

    def postprocess(
        self, preds: Dict[str, np.ndarray], original_shape: Tuple[int, int]
    ) -> List[VideoClassificationResult]:
        results = []
        print(self.label_map.keys())
        for k, multibatch_results in preds.items():
            print(f"Processing output '{k}' with shape: {multibatch_results.shape}")

            # 处理每个batch中的样本
            for batch_idx, v in enumerate(multibatch_results):
                # 如果需要，应用softmax
                v = v.reshape(-1)
                v = softmax(v)

                # 获取top k结果的索引（从小到大排序，取最后k个）
                top_k_indices = np.argsort(v)[-self.top_k :]
                # 反转顺序，使得最高分在前
                top_k_indices = top_k_indices[::-1]

                # 获取对应的分数和类别ID
                print(top_k_indices)
                top_k_scores = v[top_k_indices]
                top_k_classes = top_k_indices

                # 获取类别名称
                top_k_names = []
                for class_id in top_k_classes:
                    class_name = self.label_map.get(class_id, f"Unknown_{class_id}")
                    top_k_names.append(class_name)

                # 创建ImageNetResult对象
                result = VideoClassificationResult(
                    class_index_top5=top_k_classes,
                    score_top5=top_k_scores,
                    class_name_top5=top_k_names,
                    original_shape=original_shape,
                )

                results.append(result)

        return results
