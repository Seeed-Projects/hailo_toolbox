from typing import Dict, Any, List, Tuple, Optional, Union
from itertools import product

from hailo_toolbox.process.base import BasePostprocessor, non_max_suppression
from hailo_toolbox.process.results.face_detection import FaceDetectionResult
from hailo_toolbox.inference.core import CALLBACK_REGISTRY
import numpy as np


@CALLBACK_REGISTRY.registryPostProcessor("scrfd_10g", "scrfd_2_5g", "scrfd_500m")
class SCRFDPostprocessor(BasePostprocessor):
    """SCRFD 人脸检测后处理器 - 纯NumPy实现"""

    # 模型相关参数
    NUM_CLASSES = 1
    NUM_LANDMARKS = 10
    LABEL_OFFSET = 1

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # 默认参数
        self.image_dims = (640, 640)  # 默认输入尺寸
        self.nms_iou_thresh = 0.4
        self.score_threshold = 0.5

        # 从config中获取参数
        if config:
            self.image_dims = config.get("image_dims", self.image_dims)
            self.nms_iou_thresh = config.get("nms_iou_thresh", self.nms_iou_thresh)
            self.score_threshold = config.get("score_threshold", self.score_threshold)

        # SCRFD锚点配置
        self.anchors_config = {
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
        }

        # 生成锚点
        self._anchors = self._generate_anchors()

    def _generate_anchors(self) -> np.ndarray:
        """生成SCRFD模型的锚点"""
        anchors = []
        min_sizes = self.anchors_config["min_sizes"]
        steps = self.anchors_config["steps"]

        for stride, min_size in zip(steps, min_sizes):
            height = self.image_dims[0] // stride
            width = self.image_dims[1] // stride
            num_anchors = len(min_size)

            # 生成锚点中心
            y_coords, x_coords = np.mgrid[:height, :width]
            anchor_centers = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            # 归一化坐标
            anchor_centers[:, 0] /= self.image_dims[1]  # x坐标归一化
            anchor_centers[:, 1] /= self.image_dims[0]  # y坐标归一化

            # 如果有多个锚点尺寸，复制锚点中心
            if num_anchors > 1:
                anchor_centers = np.tile(
                    anchor_centers[:, None, :], (1, num_anchors, 1)
                ).reshape((-1, 2))

            # 生成锚点尺寸
            anchor_scales = np.ones_like(anchor_centers, dtype=np.float32) * stride
            anchor_scales[:, 0] /= self.image_dims[1]  # x方向尺寸归一化
            anchor_scales[:, 1] /= self.image_dims[0]  # y方向尺寸归一化

            # 合并锚点中心和尺寸
            anchor = np.concatenate([anchor_centers, anchor_scales], axis=1)
            anchors.append(anchor)

        return np.concatenate(anchors, axis=0)

    def _decode_boxes(
        self, box_predictions: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """解码边界框预测"""
        # box_predictions: [N, 4] - [dx, dy, dw, dh]
        # anchors: [N, 4] - [cx, cy, sx, sy]

        x1 = anchors[:, 0] - box_predictions[:, 0] * anchors[:, 2]
        y1 = anchors[:, 1] - box_predictions[:, 1] * anchors[:, 3]
        x2 = anchors[:, 0] + box_predictions[:, 2] * anchors[:, 2]
        y2 = anchors[:, 1] + box_predictions[:, 3] * anchors[:, 3]

        return np.stack([x1, y1, x2, y2], axis=-1)

    def _decode_landmarks(
        self, landmarks_predictions: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """解码关键点预测"""
        # landmarks_predictions: [N, 10] - [dx1, dy1, dx2, dy2, ..., dx5, dy5]
        # anchors: [N, 4] - [cx, cy, sx, sy]

        decoded_landmarks = []
        for i in range(0, self.NUM_LANDMARKS, 2):
            px = anchors[:, 0] + landmarks_predictions[:, i] * anchors[:, 2]
            py = anchors[:, 1] + landmarks_predictions[:, i + 1] * anchors[:, 3]
            decoded_landmarks.extend([px, py])

        return np.stack(decoded_landmarks, axis=-1)

    def _apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """应用非最大值抑制"""
        if len(boxes) == 0:
            return boxes, scores, landmarks

        # 使用框架提供的NMS函数
        keep_indices = non_max_suppression(
            boxes=boxes,
            scores=scores,
            iou_threshold=self.nms_iou_thresh,
            score_threshold=self.score_threshold,
            max_detections=1000,
        )

        if len(keep_indices) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 10), dtype=np.float32) if landmarks is not None else None,
            )

        nms_boxes = boxes[keep_indices]
        nms_scores = scores[keep_indices]
        nms_landmarks = landmarks[keep_indices] if landmarks is not None else None

        return nms_boxes, nms_scores, nms_landmarks

    def postprocess(
        self,
        preds: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> List[FaceDetectionResult]:
        """后处理函数"""
        results = []

        # 如果没有输出，返回空结果
        if not preds:
            return [
                FaceDetectionResult(
                    boxes=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    landmarks=None,
                    original_shape=original_shape,
                )
            ]

        # 根据输出形状组织数据
        # 从实际输出可以看到：
        # - (3, 80, 80, 2) -> stride 8 分类
        # - (3, 80, 80, 8) -> stride 8 边界框 (4个坐标 * 2个锚点)
        # - (3, 80, 80, 20) -> stride 8 关键点 (10个坐标 * 2个锚点)
        # - (3, 40, 40, 2) -> stride 16 分类
        # - (3, 40, 40, 8) -> stride 16 边界框
        # - (3, 40, 40, 20) -> stride 16 关键点
        # - (3, 20, 20, 2) -> stride 32 分类
        # - (3, 20, 20, 8) -> stride 32 边界框
        # - (3, 20, 20, 20) -> stride 32 关键点

        stride_outputs = {}

        # 按照输出形状组织数据
        for key, value in preds.items():
            # 根据特征图尺寸确定stride
            if len(value.shape) == 4:
                _, h, w, c = value.shape

                # 根据特征图尺寸确定stride
                if h == self.image_dims[0] // 8 and w == self.image_dims[1] // 8:
                    stride = 8
                elif h == self.image_dims[0] // 16 and w == self.image_dims[1] // 16:
                    stride = 16
                elif h == self.image_dims[0] // 32 and w == self.image_dims[1] // 32:
                    stride = 32
                else:
                    print(f"Warning: Unknown feature map size {h}x{w}, skipping")
                    continue

                if stride not in stride_outputs:
                    stride_outputs[stride] = {}

                # 根据通道数确定输出类型
                if c == 2 or c == 4:  # 分类输出 (1个类别 * 2个锚点)
                    stride_outputs[stride]["classes"] = value
                elif c == 8:  # 边界框输出 (4个坐标 * 2个锚点)
                    stride_outputs[stride]["boxes"] = value
                elif c == 20:  # 关键点输出 (10个坐标 * 2个锚点)
                    stride_outputs[stride]["landmarks"] = value
                else:
                    print(
                        f"Warning: Unknown channel size {c} for stride {stride}, skipping"
                    )

        # 收集所有分支的预测
        box_predictors_list = []
        class_predictors_list = []
        landmarks_predictors_list = []

        # 按照stride顺序处理
        for stride in [8, 16, 32]:
            if stride in stride_outputs:
                stride_data = stride_outputs[stride]

                # 获取batch size
                batch_size = list(stride_data.values())[0].shape[0]

                # 处理边界框预测
                if "boxes" in stride_data:
                    box_pred = stride_data["boxes"]  # (batch, h, w, 8)
                    # 重新整形为 (batch, h*w*2, 4)
                    box_pred = box_pred.reshape(batch_size, -1, 4)
                    box_predictors_list.append(box_pred)

                # 处理类别预测
                if "classes" in stride_data:
                    class_pred = stride_data["classes"]  # (batch, h, w, 2)
                    # 重新整形为 (batch, h*w*2, 1)
                    class_pred = class_pred.reshape(batch_size, -1, 1)
                    class_predictors_list.append(class_pred)

                # 处理关键点预测
                if "landmarks" in stride_data:
                    landmarks_pred = stride_data["landmarks"]  # (batch, h, w, 20)
                    # 重新整形为 (batch, h*w*2, 10)
                    landmarks_pred = landmarks_pred.reshape(batch_size, -1, 10)
                    landmarks_predictors_list.append(landmarks_pred)

        # 合并所有分支的预测
        if not box_predictors_list or not class_predictors_list:
            return [
                FaceDetectionResult(
                    boxes=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    landmarks=None,
                    original_shape=original_shape,
                )
            ]

        box_predictions = np.concatenate(box_predictors_list, axis=1)
        class_predictions = np.concatenate(class_predictors_list, axis=1)
        landmarks_predictions = (
            np.concatenate(landmarks_predictors_list, axis=1)
            if landmarks_predictors_list
            else None
        )

        # 处理每个batch
        batch_size = box_predictions.shape[0]
        for batch_idx in range(batch_size):
            # 获取当前batch的预测
            batch_box_pred = box_predictions[batch_idx]  # [N, 4]
            batch_class_pred = class_predictions[batch_idx]  # [N, 1]
            batch_landmarks_pred = (
                landmarks_predictions[batch_idx]
                if landmarks_predictions is not None
                else None
            )

            # 获取置信度分数
            detection_scores = batch_class_pred[:, 0]  # 假设只有一个类别（人脸）

            # 过滤低置信度预测
            valid_mask = detection_scores >= self.score_threshold
            if not np.any(valid_mask):
                results.append(
                    FaceDetectionResult(
                        boxes=np.empty((0, 4), dtype=np.float32),
                        scores=np.empty((0,), dtype=np.float32),
                        landmarks=None,
                        original_shape=original_shape,
                    )
                )
                continue

            # 应用有效性掩码
            valid_boxes = batch_box_pred[valid_mask]
            valid_scores = detection_scores[valid_mask]
            valid_landmarks = (
                batch_landmarks_pred[valid_mask]
                if batch_landmarks_pred is not None
                else None
            )
            valid_anchors = self._anchors[valid_mask]

            # 解码边界框
            decoded_boxes = self._decode_boxes(valid_boxes, valid_anchors)

            # 解码关键点
            decoded_landmarks = None
            if valid_landmarks is not None:
                decoded_landmarks = self._decode_landmarks(
                    valid_landmarks, valid_anchors
                )

            # 应用NMS
            nms_boxes, nms_scores, nms_landmarks = self._apply_nms(
                decoded_boxes, valid_scores, decoded_landmarks
            )

            # 创建结果 (保持归一化坐标)
            result = FaceDetectionResult(
                boxes=nms_boxes,
                scores=nms_scores,
                landmarks=nms_landmarks,
                original_shape=original_shape,
            )

            results.append(result)

        return results


@CALLBACK_REGISTRY.registryPostProcessor("retinaface_mbnet")
class RetinafaceMBNetPostprocessor(BasePostprocessor):
    """RetinaFace MobileNet 人脸检测后处理器 - 纯NumPy实现"""

    # 模型相关参数
    LABEL_OFFSET = 1
    NUM_CLASSES = 1
    SCALE_FACTORS = (10.0, 5.0)

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # 默认参数 - 根据模型输出推断的输入尺寸
        self.image_dims = (736, 1280)  # 根据输出特征图尺寸推断
        self.nms_iou_thresh = 0.6
        self.score_threshold = 0.005  # 设置更合理的阈值

        # 从config中获取参数
        if config:
            self.image_dims = config.get("image_dims", self.image_dims)
            self.nms_iou_thresh = config.get("nms_iou_thresh", self.nms_iou_thresh)
            self.score_threshold = config.get("score_threshold", self.score_threshold)

        # RetinaFace锚点配置
        self.anchors_config = {
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
        }

        # 生成锚点
        self._anchors = self._extract_anchors()

    def _extract_anchors(self) -> np.ndarray:
        """生成RetinaFace模型的锚点，参考retain_face.py的逻辑"""
        min_sizes = self.anchors_config["min_sizes"]
        steps = self.anchors_config["steps"]

        # 计算特征图尺寸
        feature_maps = [
            [
                int(np.ceil(self.image_dims[0] / step)),
                int(np.ceil(self.image_dims[1] / step)),
            ]
            for step in steps
        ]

        anchors = []
        for feature_map_index, feature_map in enumerate(feature_maps):
            current_min_sizes = min_sizes[feature_map_index]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in current_min_sizes:
                    s_kx = min_size / self.image_dims[1]
                    s_ky = min_size / self.image_dims[0]
                    cx = (j + 0.5) / feature_map[1]
                    cy = (i + 0.5) / feature_map[0]
                    anchor = np.clip(
                        np.array([cx, cy, s_kx, s_ky], dtype=np.float32), 0.0, 1.0
                    )
                    anchors.append(anchor)

        return np.array(anchors, dtype=np.float32)

    def _collect_box_class_predictions(
        self, preds: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """收集并组织边界框、分类和关键点预测，根据输出形状确定类型"""

        # 根据输出形状组织数据
        # RetinaFace MobileNet模型的输出形状：
        # - (3, 92, 160, 8) -> stride 8 边界框 (4个坐标 * 2个锚点)
        # - (3, 92, 160, 4) -> stride 8 分类 (2个类别 * 2个锚点)
        # - (3, 92, 160, 20) -> stride 8 关键点 (10个坐标 * 2个锚点)
        # - (3, 46, 80, 8) -> stride 16 边界框
        # - (3, 46, 80, 4) -> stride 16 分类
        # - (3, 46, 80, 20) -> stride 16 关键点
        # - (3, 23, 40, 8) -> stride 32 边界框
        # - (3, 23, 40, 4) -> stride 32 分类
        # - (3, 23, 40, 20) -> stride 32 关键点

        stride_outputs = {}

        # 按照输出形状组织数据
        for key, value in preds.items():
            if len(value.shape) == 4:
                _, h, w, c = value.shape

                # 根据特征图尺寸确定stride
                if h == self.image_dims[0] // 8 and w == self.image_dims[1] // 8:
                    stride = 8
                elif h == self.image_dims[0] // 16 and w == self.image_dims[1] // 16:
                    stride = 16
                elif h == self.image_dims[0] // 32 and w == self.image_dims[1] // 32:
                    stride = 32
                else:
                    print(f"Warning: Unknown feature map size {h}x{w}, skipping")
                    continue

                if stride not in stride_outputs:
                    stride_outputs[stride] = {}

                # 根据通道数确定输出类型
                if c == 4:  # 分类输出 (2个类别 * 2个锚点)
                    stride_outputs[stride]["classes"] = value
                elif c == 8:  # 边界框输出 (4个坐标 * 2个锚点)
                    stride_outputs[stride]["boxes"] = value
                elif c == 20:  # 关键点输出 (10个坐标 * 2个锚点)
                    stride_outputs[stride]["landmarks"] = value
                else:
                    print(
                        f"Warning: Unknown channel size {c} for stride {stride}, skipping"
                    )

        # 收集所有分支的预测
        box_predictors_list = []
        class_predictors_list = []
        landmarks_predictors_list = []

        # 按照stride顺序处理
        for stride in [8, 16, 32]:
            if stride in stride_outputs:
                stride_data = stride_outputs[stride]

                # 获取batch size
                batch_size = list(stride_data.values())[0].shape[0]

                # 处理边界框预测
                if "boxes" in stride_data:
                    box_pred = stride_data["boxes"]  # (batch, h, w, 8)
                    # 重新整形为 (batch, h*w*2, 4)
                    box_pred = box_pred.reshape(batch_size, -1, 4)
                    box_predictors_list.append(box_pred)

                # 处理类别预测
                if "classes" in stride_data:
                    class_pred = stride_data["classes"]  # (batch, h, w, 4)
                    # 重新整形为 (batch, h*w*2, 2)
                    class_pred = class_pred.reshape(
                        batch_size, -1, self.NUM_CLASSES + 1
                    )
                    class_predictors_list.append(class_pred)

                # 处理关键点预测
                if "landmarks" in stride_data:
                    landmarks_pred = stride_data["landmarks"]  # (batch, h, w, 20)
                    # 重新整形为 (batch, h*w*2, 10)
                    landmarks_pred = landmarks_pred.reshape(batch_size, -1, 10)
                    landmarks_predictors_list.append(landmarks_pred)

        # 合并所有分支的预测
        if not box_predictors_list or not class_predictors_list:
            batch_size = 1
            return (
                np.empty((batch_size, 0, 4), dtype=np.float32),
                np.empty((batch_size, 0, self.NUM_CLASSES + 1), dtype=np.float32),
                None,
            )

        box_predictions = np.concatenate(box_predictors_list, axis=1)
        class_predictions = np.concatenate(class_predictors_list, axis=1)
        landmarks_predictions = (
            np.concatenate(landmarks_predictors_list, axis=1)
            if landmarks_predictors_list
            else None
        )

        return box_predictions, class_predictions, landmarks_predictions

    def _decode_boxes(
        self, box_detections: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """解码边界框预测，参考retain_face.py的逻辑"""
        # box_detections: [N, 4] - [dx, dy, dw, dh]
        # anchors: [N, 4] - [cx, cy, sx, sy]

        # 解码中心点和尺寸
        boxes = np.concatenate(
            [
                anchors[:, :2]
                + box_detections[:, :2] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                anchors[:, 2:] * np.exp(box_detections[:, 2:] / self.SCALE_FACTORS[1]),
            ],
            axis=1,
        )

        # 转换为[x1, y1, x2, y2]格式
        boxes_low_dims = boxes[:, :2] - boxes[:, 2:] / 2
        boxes_high_dims = boxes[:, 2:] + boxes_low_dims
        new_boxes = np.concatenate([boxes_low_dims, boxes_high_dims], axis=1)

        return new_boxes

    def _decode_landmarks(
        self, landmarks_detections: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """解码关键点预测，参考retain_face.py的逻辑"""
        # landmarks_detections: [N, 10] - [dx1, dy1, dx2, dy2, ..., dx5, dy5]
        # anchors: [N, 4] - [cx, cy, sx, sy]

        decoded_landmarks = []
        for i in range(0, 10, 2):
            px = (
                anchors[:, 0]
                + landmarks_detections[:, i] / self.SCALE_FACTORS[0] * anchors[:, 2]
            )
            py = (
                anchors[:, 1]
                + landmarks_detections[:, i + 1] / self.SCALE_FACTORS[0] * anchors[:, 3]
            )
            decoded_landmarks.extend([px, py])

        return np.stack(decoded_landmarks, axis=1)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """计算softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """应用非最大值抑制"""
        if len(boxes) == 0:
            return boxes, scores, landmarks

        # 使用框架提供的NMS函数
        keep_indices = non_max_suppression(
            boxes=boxes,
            scores=scores,
            iou_threshold=self.nms_iou_thresh,
            score_threshold=self.score_threshold,
            max_detections=1000,
        )

        if len(keep_indices) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 10), dtype=np.float32) if landmarks is not None else None,
            )

        nms_boxes = boxes[keep_indices]
        nms_scores = scores[keep_indices]
        nms_landmarks = landmarks[keep_indices] if landmarks is not None else None

        return nms_boxes, nms_scores, nms_landmarks

    def postprocess(
        self,
        preds: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> List[FaceDetectionResult]:
        """后处理函数，参考retain_face.py的逻辑"""
        results = []

        # 如果没有输出，返回空结果
        if not preds:
            return [
                FaceDetectionResult(
                    boxes=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    landmarks=None,
                    original_shape=original_shape,
                )
            ]

        # 收集边界框、分类和关键点预测
        box_predictions, class_predictions, landmarks_predictions = (
            self._collect_box_class_predictions(preds)
        )

        if box_predictions.shape[1] == 0:
            return [
                FaceDetectionResult(
                    boxes=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    landmarks=None,
                    original_shape=original_shape,
                )
            ]

        # 处理每个batch
        batch_size = box_predictions.shape[0]
        for batch_idx in range(batch_size):
            # 获取当前batch的预测
            batch_box_pred = box_predictions[batch_idx]  # [N, 4]
            batch_class_pred = class_predictions[batch_idx]  # [N, num_classes+1]
            batch_landmarks_pred = (
                landmarks_predictions[batch_idx]
                if landmarks_predictions is not None
                else None
            )

            # 计算softmax分数 - 注意：标准流程中axis=2，但我们这里是2D数组，所以用axis=1
            class_scores_softmax = self._softmax(batch_class_pred, axis=1)

            # 获取人脸类别的分数 (跳过背景类别)
            detection_scores = class_scores_softmax[:, 1]  # 人脸类别的分数

            # 过滤低置信度预测
            valid_mask = detection_scores >= self.score_threshold
            if not np.any(valid_mask):
                results.append(
                    FaceDetectionResult(
                        boxes=np.empty((0, 4), dtype=np.float32),
                        scores=np.empty((0,), dtype=np.float32),
                        landmarks=None,
                        original_shape=original_shape,
                    )
                )
                continue

            # 应用有效性掩码
            valid_boxes = batch_box_pred[valid_mask]
            valid_scores = detection_scores[valid_mask]
            valid_landmarks = (
                batch_landmarks_pred[valid_mask]
                if batch_landmarks_pred is not None
                else None
            )
            valid_anchors = self._anchors[valid_mask]

            # 解码边界框
            decoded_boxes = self._decode_boxes(valid_boxes, valid_anchors)

            # 解码关键点
            decoded_landmarks = None
            if valid_landmarks is not None:
                decoded_landmarks = self._decode_landmarks(
                    valid_landmarks, valid_anchors
                )

            # 应用NMS
            nms_boxes, nms_scores, nms_landmarks = self._apply_nms(
                decoded_boxes, valid_scores, decoded_landmarks
            )

            # 创建结果 (保持归一化坐标)
            result = FaceDetectionResult(
                boxes=nms_boxes,
                scores=nms_scores,
                landmarks=nms_landmarks,
                original_shape=original_shape,
            )

            results.append(result)

        return results
