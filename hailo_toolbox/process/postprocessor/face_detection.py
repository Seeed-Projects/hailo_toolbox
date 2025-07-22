from typing import Dict, Any, List, Tuple, Optional, Union
from itertools import product

from hailo_toolbox.process.base import BasePostprocessor, non_max_suppression
from hailo_toolbox.process.results.face_detection import FaceDetectionResult
from hailo_toolbox.inference.core import CALLBACK_REGISTRY
import numpy as np


@CALLBACK_REGISTRY.registryPostProcessor("scrfd_10g", "scrfd_2_5g", "scrfd_500m")
class SCRFDPostprocessor(BasePostprocessor):
    """SCRFD Face Detection Postprocessor - Pure NumPy Implementation"""

    # Model related parameters
    NUM_CLASSES = 1
    NUM_LANDMARKS = 10
    LABEL_OFFSET = 1

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # Default parameters
        self.image_dims = (640, 640)  # Default input size
        self.nms_iou_thresh = 0.4
        self.score_threshold = 0.5

        # Get parameters from config
        if config:
            self.image_dims = config.get("image_dims", self.image_dims)
            self.nms_iou_thresh = config.get("nms_iou_thresh", self.nms_iou_thresh)
            self.score_threshold = config.get("score_threshold", self.score_threshold)

        # SCRFD anchor configuration
        self.anchors_config = {
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
        }

        # Generate anchors
        self._anchors = self._generate_anchors()

    def _generate_anchors(self) -> np.ndarray:
        """Generate anchors for SCRFD model"""
        anchors = []
        min_sizes = self.anchors_config["min_sizes"]
        steps = self.anchors_config["steps"]

        for stride, min_size in zip(steps, min_sizes):
            height = self.image_dims[0] // stride
            width = self.image_dims[1] // stride
            num_anchors = len(min_size)

            # Generate anchor centers
            y_coords, x_coords = np.mgrid[:height, :width]
            anchor_centers = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))

            # Normalize coordinates
            anchor_centers[:, 0] /= self.image_dims[1]  # Normalize x coordinates
            anchor_centers[:, 1] /= self.image_dims[0]  # Normalize y coordinates

            # If there are multiple anchor sizes, replicate anchor centers
            if num_anchors > 1:
                anchor_centers = np.tile(
                    anchor_centers[:, None, :], (1, num_anchors, 1)
                ).reshape((-1, 2))

            # Generate anchor scales
            anchor_scales = np.ones_like(anchor_centers, dtype=np.float32) * stride
            anchor_scales[:, 0] /= self.image_dims[1]  # Normalize x direction scale
            anchor_scales[:, 1] /= self.image_dims[0]  # Normalize y direction scale

            # Combine anchor centers and scales
            anchor = np.concatenate([anchor_centers, anchor_scales], axis=1)
            anchors.append(anchor)

        return np.concatenate(anchors, axis=0)

    def _decode_boxes(
        self, box_predictions: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """Decode bounding box predictions"""
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
        """Decode landmark predictions"""
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
        """Apply non-maximum suppression"""
        if len(boxes) == 0:
            return boxes, scores, landmarks

        # Use framework-provided NMS function
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
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> List[FaceDetectionResult]:
        """Postprocessing function"""
        results = []

        # If no output, return empty result
        if not preds:
            return [
                FaceDetectionResult(
                    boxes=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    landmarks=None,
                    original_shape=original_shape,
                )
            ]

        # Organize data based on output shape
        # From actual output we can see:
        # - (3, 80, 80, 2) -> stride 8 classification
        # - (3, 80, 80, 8) -> stride 8 bounding boxes (4 coordinates * 2 anchors)
        # - (3, 80, 80, 20) -> stride 8 landmarks (10 coordinates * 2 anchors)
        # - (3, 40, 40, 2) -> stride 16 classification
        # - (3, 40, 40, 8) -> stride 16 bounding boxes
        # - (3, 40, 40, 20) -> stride 16 landmarks
        # - (3, 20, 20, 2) -> stride 32 classification
        # - (3, 20, 20, 8) -> stride 32 bounding boxes
        # - (3, 20, 20, 20) -> stride 32 landmarks

        stride_outputs = {}

        # Organize data by output shape
        for key, value in preds.items():
            # Determine stride based on feature map size
            if len(value.shape) == 4:
                _, h, w, c = value.shape

                # Determine stride based on feature map size
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

                # Determine output type based on channel count
                if c == 2 or c == 4:  # Classification output (1 class * 2 anchors)
                    stride_outputs[stride]["classes"] = value
                elif c == 8:  # Bounding box output (4 coordinates * 2 anchors)
                    stride_outputs[stride]["boxes"] = value
                elif c == 20:  # Landmark output (10 coordinates * 2 anchors)
                    stride_outputs[stride]["landmarks"] = value
                else:
                    print(
                        f"Warning: Unknown channel size {c} for stride {stride}, skipping"
                    )

        # Collect predictions from all branches
        box_predictors_list = []
        class_predictors_list = []
        landmarks_predictors_list = []

        # Process in stride order
        for stride in [8, 16, 32]:
            if stride in stride_outputs:
                stride_data = stride_outputs[stride]

                # Get batch size
                batch_size = list(stride_data.values())[0].shape[0]

                # Process bounding box predictions
                if "boxes" in stride_data:
                    box_pred = stride_data["boxes"]  # (batch, h, w, 8)
                    # Reshape to (batch, h*w*2, 4)
                    box_pred = box_pred.reshape(batch_size, -1, 4)
                    box_predictors_list.append(box_pred)

                # Process class predictions
                if "classes" in stride_data:
                    class_pred = stride_data["classes"]  # (batch, h, w, 2)
                    # Reshape to (batch, h*w*2, 1)
                    class_pred = class_pred.reshape(batch_size, -1, 1)
                    class_predictors_list.append(class_pred)

                # Process landmark predictions
                if "landmarks" in stride_data:
                    landmarks_pred = stride_data["landmarks"]  # (batch, h, w, 20)
                    # Reshape to (batch, h*w*2, 10)
                    landmarks_pred = landmarks_pred.reshape(batch_size, -1, 10)
                    landmarks_predictors_list.append(landmarks_pred)

        # Merge predictions from all branches
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

        # Process each batch
        batch_size = box_predictions.shape[0]
        for batch_idx in range(batch_size):
            # Get predictions for current batch
            batch_box_pred = box_predictions[batch_idx]  # [N, 4]
            batch_class_pred = class_predictions[batch_idx]  # [N, 1]
            batch_landmarks_pred = (
                landmarks_predictions[batch_idx]
                if landmarks_predictions is not None
                else None
            )

            # Get confidence scores
            detection_scores = batch_class_pred[:, 0]  # Assume only one class (face)

            # Filter low confidence predictions
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

            # Apply validity mask
            valid_boxes = batch_box_pred[valid_mask]
            valid_scores = detection_scores[valid_mask]
            valid_landmarks = (
                batch_landmarks_pred[valid_mask]
                if batch_landmarks_pred is not None
                else None
            )
            valid_anchors = self._anchors[valid_mask]

            # Decode bounding boxes
            decoded_boxes = self._decode_boxes(valid_boxes, valid_anchors)

            # Decode landmarks
            decoded_landmarks = None
            if valid_landmarks is not None:
                decoded_landmarks = self._decode_landmarks(
                    valid_landmarks, valid_anchors
                )

            # Apply NMS
            nms_boxes, nms_scores, nms_landmarks = self._apply_nms(
                decoded_boxes, valid_scores, decoded_landmarks
            )

            # Create result (keep normalized coordinates)
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
    """RetinaFace MobileNet Face Detection Postprocessor - Pure NumPy Implementation"""

    # Model related parameters
    LABEL_OFFSET = 1
    NUM_CLASSES = 1
    SCALE_FACTORS = (10.0, 5.0)

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # Default parameters - inferred input size based on model output
        self.image_dims = (736, 1280)  # Inferred from output feature map sizes
        self.nms_iou_thresh = 0.6
        self.score_threshold = 0.005  # Set more reasonable threshold

        # Get parameters from config
        if config:
            self.image_dims = config.get("image_dims", self.image_dims)
            self.nms_iou_thresh = config.get("nms_iou_thresh", self.nms_iou_thresh)
            self.score_threshold = config.get("score_threshold", self.score_threshold)

        # RetinaFace anchor configuration
        self.anchors_config = {
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
        }

        # Generate anchors
        self._anchors = self._extract_anchors()

    def _extract_anchors(self) -> np.ndarray:
        """Generate anchors for RetinaFace model, referencing retain_face.py logic"""
        min_sizes = self.anchors_config["min_sizes"]
        steps = self.anchors_config["steps"]

        # Calculate feature map sizes
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
        """Collect and organize bounding box, classification and landmark predictions based on output shapes"""

        # Organize data based on output shapes
        # RetinaFace MobileNet model output shapes:
        # - (3, 92, 160, 8) -> stride 8 bounding boxes (4 coordinates * 2 anchors)
        # - (3, 92, 160, 4) -> stride 8 classification (2 classes * 2 anchors)
        # - (3, 92, 160, 20) -> stride 8 landmarks (10 coordinates * 2 anchors)
        # - (3, 46, 80, 8) -> stride 16 bounding boxes
        # - (3, 46, 80, 4) -> stride 16 classification
        # - (3, 46, 80, 20) -> stride 16 landmarks
        # - (3, 23, 40, 8) -> stride 32 bounding boxes
        # - (3, 23, 40, 4) -> stride 32 classification
        # - (3, 23, 40, 20) -> stride 32 landmarks

        stride_outputs = {}

        # Organize data by output shape
        for key, value in preds.items():
            if len(value.shape) == 4:
                _, h, w, c = value.shape

                # Determine stride based on feature map size
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

                # Determine output type based on channel count
                if c == 4:  # Classification output (2 classes * 2 anchors)
                    stride_outputs[stride]["classes"] = value
                elif c == 8:  # Bounding box output (4 coordinates * 2 anchors)
                    stride_outputs[stride]["boxes"] = value
                elif c == 20:  # Landmark output (10 coordinates * 2 anchors)
                    stride_outputs[stride]["landmarks"] = value
                else:
                    print(
                        f"Warning: Unknown channel size {c} for stride {stride}, skipping"
                    )

        # Collect predictions from all branches
        box_predictors_list = []
        class_predictors_list = []
        landmarks_predictors_list = []

        # Process in stride order
        for stride in [8, 16, 32]:
            if stride in stride_outputs:
                stride_data = stride_outputs[stride]

                # Get batch size
                batch_size = list(stride_data.values())[0].shape[0]

                # Process bounding box predictions
                if "boxes" in stride_data:
                    box_pred = stride_data["boxes"]  # (batch, h, w, 8)
                    # Reshape to (batch, h*w*2, 4)
                    box_pred = box_pred.reshape(batch_size, -1, 4)
                    box_predictors_list.append(box_pred)

                # Process class predictions
                if "classes" in stride_data:
                    class_pred = stride_data["classes"]  # (batch, h, w, 4)
                    # Reshape to (batch, h*w*2, 2)
                    class_pred = class_pred.reshape(
                        batch_size, -1, self.NUM_CLASSES + 1
                    )
                    class_predictors_list.append(class_pred)

                # Process landmark predictions
                if "landmarks" in stride_data:
                    landmarks_pred = stride_data["landmarks"]  # (batch, h, w, 20)
                    # Reshape to (batch, h*w*2, 10)
                    landmarks_pred = landmarks_pred.reshape(batch_size, -1, 10)
                    landmarks_predictors_list.append(landmarks_pred)

        # Merge predictions from all branches
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
        """Decode bounding box predictions, referencing retain_face.py logic"""
        # box_detections: [N, 4] - [dx, dy, dw, dh]
        # anchors: [N, 4] - [cx, cy, sx, sy]

        # Decode center points and sizes
        boxes = np.concatenate(
            [
                anchors[:, :2]
                + box_detections[:, :2] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                anchors[:, 2:] * np.exp(box_detections[:, 2:] / self.SCALE_FACTORS[1]),
            ],
            axis=1,
        )

        # Convert to [x1, y1, x2, y2] format
        boxes_low_dims = boxes[:, :2] - boxes[:, 2:] / 2
        boxes_high_dims = boxes[:, 2:] + boxes_low_dims
        new_boxes = np.concatenate([boxes_low_dims, boxes_high_dims], axis=1)

        return new_boxes

    def _decode_landmarks(
        self, landmarks_detections: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        """Decode landmark predictions, referencing retain_face.py logic"""
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
        """Calculate softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply non-maximum suppression"""
        if len(boxes) == 0:
            return boxes, scores, landmarks

        # Use framework-provided NMS function
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
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> List[FaceDetectionResult]:
        """Postprocessing function, referencing retain_face.py logic"""
        results = []

        # If no output, return empty result
        if not preds:
            return [
                FaceDetectionResult(
                    boxes=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    landmarks=None,
                    original_shape=original_shape,
                )
            ]

        # Collect bounding box, classification and landmark predictions
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

        # Process each batch
        batch_size = box_predictions.shape[0]
        for batch_idx in range(batch_size):
            # Get predictions for current batch
            batch_box_pred = box_predictions[batch_idx]  # [N, 4]
            batch_class_pred = class_predictions[batch_idx]  # [N, num_classes+1]
            batch_landmarks_pred = (
                landmarks_predictions[batch_idx]
                if landmarks_predictions is not None
                else None
            )

            # Calculate softmax scores - Note: in standard process axis=2, but here we have 2D array, so use axis=1
            class_scores_softmax = self._softmax(batch_class_pred, axis=1)

            # Get face class scores (skip background class)
            detection_scores = class_scores_softmax[:, 1]  # Face class scores

            # Filter low confidence predictions
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

            # Apply validity mask
            valid_boxes = batch_box_pred[valid_mask]
            valid_scores = detection_scores[valid_mask]
            valid_landmarks = (
                batch_landmarks_pred[valid_mask]
                if batch_landmarks_pred is not None
                else None
            )
            valid_anchors = self._anchors[valid_mask]

            # Decode bounding boxes
            decoded_boxes = self._decode_boxes(valid_boxes, valid_anchors)

            # Decode landmarks
            decoded_landmarks = None
            if valid_landmarks is not None:
                decoded_landmarks = self._decode_landmarks(
                    valid_landmarks, valid_anchors
                )

            # Apply NMS
            nms_boxes, nms_scores, nms_landmarks = self._apply_nms(
                decoded_boxes, valid_scores, decoded_landmarks
            )

            # Create result (keep normalized coordinates)
            result = FaceDetectionResult(
                boxes=nms_boxes,
                scores=nms_scores,
                landmarks=nms_landmarks,
                original_shape=original_shape,
            )

            results.append(result)

        return results
