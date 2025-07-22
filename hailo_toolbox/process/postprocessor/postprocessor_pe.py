"""
Pose Estimation Postprocessor

This module implements postprocessing for pose estimation models including
YOLOv8 pose estimation and CenterPose models. It handles various output formats
and provides comprehensive keypoint detection results.
"""

import math
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from operator import itemgetter
import logging

from ..base import (
    BasePostprocessor,
    PostprocessConfig,
)
from hailo_toolbox.process.results.pose import KeypointResult
from hailo_toolbox.inference.core import CALLBACK_REGISTRY
from hailo_toolbox.process.postprocessor.centerpose_postprocessing import (
    centerpose_postprocessing,
)


logger = logging.getLogger(__name__)


# COCO pose estimation constants
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Joint pairs for skeleton visualization (COCO format)
COCO_JOINT_PAIRS = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],  # Head
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],  # Arms
    [5, 11],
    [6, 12],
    [11, 12],  # Torso
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],  # Legs
]

# OpenPose body parts keypoint IDs for traditional pose estimation
BODY_PARTS_KPT_IDS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 16],
    [5, 17],
]

# OpenPose Part Affinity Fields IDs
BODY_PARTS_PAF_IDS = (
    [12, 13],
    [20, 21],
    [14, 15],
    [16, 17],
    [22, 23],
    [24, 25],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [28, 29],
    [30, 31],
    [34, 35],
    [32, 33],
    [36, 37],
    [18, 19],
    [26, 27],
)

# Default stride for heatmap processing
HEATMAP_STRIDE = 8


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from xywh to xyxy format.

    Args:
        boxes: Boxes in xywh format (center_x, center_y, width, height)

    Returns:
        Boxes in xyxy format (x1, y1, x2, y2)
    """
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return boxes_xyxy


def linspace2d(
    start: np.ndarray, stop: np.ndarray, n: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate linearly spaced points between start and stop points in 2D.

    Args:
        start: Starting points with shape (2,)
        stop: Ending points with shape (2,)
        n: Number of points to generate

    Returns:
        Tuple of (x_coords, y_coords) arrays
    """
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


@CALLBACK_REGISTRY.registryPostProcessor("yolov8spose", "yolov8mpose")
class YOLOv8PosePostprocessor(BasePostprocessor):
    """
    Postprocessor for YOLOv8 pose estimation models.

    This class handles the conversion of raw YOLOv8 pose estimation outputs into
    structured keypoint results with proper NMS and coordinate scaling.

    Input format (endnodes list):
    - endnodes[0]: bbox output with shapes (BS, 20, 20, 64)
    - endnodes[1]: scores output with shapes (BS, 20, 20, 80)
    - endnodes[2]: keypoints output with shapes (BS, 20, 20, 51)
    - endnodes[3]: bbox output with shapes (BS, 40, 40, 64)
    - endnodes[4]: scores output with shapes (BS, 40, 40, 80)
    - endnodes[5]: keypoints output with shapes (BS, 40, 40, 51)
    - endnodes[6]: bbox output with shapes (BS, 80, 80, 64)
    - endnodes[7]: scores output with shapes (BS, 80, 80, 80)
    - endnodes[8]: keypoints output with shapes (BS, 80, 80, 51)
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the YOLOv8 pose estimation postprocessor.

        Args:
            config: Postprocessing configuration. If None, default config is used.
        """
        super().__init__(config)

        # Set pose estimation specific defaults
        if not self.config.kp:
            self.config.kp = True

        # Validate configuration
        self._validate_config()

        # Set up keypoint names if not provided
        if self.config.keypoint_names is None:
            self.config.keypoint_names = COCO_KEYPOINT_NAMES

        if self.config.keypoint_connections is None:
            self.config.keypoint_connections = COCO_JOINT_PAIRS

        logger.info(
            f"Initialized YOLOv8PosePostprocessor with {self.config.num_keypoints} keypoints, "
            f"NMS enabled: {self.config.nms}"
        )

    def _validate_config(self) -> None:
        """
        Validate the postprocessing configuration.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.config.kp_conf_threshold < 0 or self.config.kp_conf_threshold > 1:
            raise ValueError(
                f"Keypoint confidence threshold must be between 0 and 1, "
                f"got {self.config.kp_conf_threshold}"
            )

        if (
            self.config.kp_visibility_threshold < 0
            or self.config.kp_visibility_threshold > 1
        ):
            raise ValueError(
                f"Keypoint visibility threshold must be between 0 and 1, "
                f"got {self.config.kp_visibility_threshold}"
            )

        if self.config.nms_iou_threshold < 0 or self.config.nms_iou_threshold > 1:
            raise ValueError(
                f"NMS IoU threshold must be between 0 and 1, "
                f"got {self.config.nms_iou_threshold}"
            )

        if self.config.num_keypoints <= 0:
            raise ValueError(
                f"Number of keypoints must be positive, got {self.config.num_keypoints}"
            )

        if self.config.kp_max_persons <= 0:
            raise ValueError(
                f"Maximum persons must be positive, got {self.config.kp_max_persons}"
            )

    NODE_LOCAL_DICT = {
        (20, 64): 0,
        (20, 1): 1,  # Changed from (20,1) to (20,80) for scores
        (20, 51): 2,
        (40, 64): 3,
        (40, 1): 4,  # Changed from (40,1) to (40,80) for scores
        (40, 51): 5,
        (80, 64): 6,
        (80, 1): 7,  # Changed from (80,1) to (80,80) for scores
        (80, 51): 8,
    }

    def postprocess(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
        input_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Postprocess raw YOLOv8 pose estimation outputs.

        Args:
            raw_outputs: Dictionary of raw model outputs where keys are output names
            original_shape: Original image shape (height, width) for coordinate scaling
            **kwargs: Additional parameters including:
                - classes: Number of classes (should be 1 for pose estimation)
                - nms_max_output_per_class: Maximum detections per class
                - anchors: Dictionary containing strides and regression_length
                - img_dims: Input image dimensions
                - score_threshold: Score threshold for filtering
                - nms_iou_thresh: IoU threshold for NMS

        Returns:
            Dictionary with the following structure:
            {
                'bboxes': np.ndarray with shape (batch_size, max_detections, 4),
                'keypoints': np.ndarray with shape (batch_size, max_detections, 17, 2),
                'joint_scores': np.ndarray with shape (batch_size, max_detections, 17, 1),
                'scores': np.ndarray with shape (batch_size, max_detections, 1)
            }

        Raises:
            ValueError: If input format is not supported
            RuntimeError: If postprocessing fails
        """
        # try:
        # Validate input and convert raw_outputs to endnodes
        endnodes = [None for _ in range(len(self.NODE_LOCAL_DICT))]
        for key, value in raw_outputs.items():
            shape_key = value.shape[-2:]
            if shape_key in self.NODE_LOCAL_DICT:
                endnodes[self.NODE_LOCAL_DICT[shape_key]] = value
            else:
                logger.warning(f"Unknown output shape {shape_key} for key {key}")

        # Check that all endnodes are filled
        for i, node in enumerate(endnodes):
            if node is None:
                raise ValueError(f"Missing endnode at index {i}")

        if len(endnodes) != 9:
            raise ValueError(
                f"Expected 9 endnodes for YOLOv8 pose estimation, got {len(endnodes)}"
            )

        # Extract parameters from kwargs with defaults
        batch_size = endnodes[0].shape[0]
        num_classes = kwargs.get("classes", 1)  # Always 1 for pose estimation
        max_detections = kwargs.get(
            "nms_max_output_per_class", self.config.kp_max_persons
        )

        # Extract anchor configuration
        anchors_config = kwargs.get("anchors", {})
        strides = anchors_config.get("strides", [8, 16, 32])[::-1]  # Reverse order
        reg_max = anchors_config.get("regression_length", 15)

        # Extract image dimensions
        image_dims = kwargs.get("img_dims", self.config.input_shape)
        if isinstance(image_dims, (list, tuple)):
            image_dims = tuple(image_dims)
        else:
            image_dims = (image_dims, image_dims)

        # Extract thresholds
        score_thres = kwargs.get("score_threshold", self.config.kp_conf_threshold)
        iou_thres = kwargs.get("nms_iou_thresh", self.config.nms_iou_threshold)

        # Extract outputs by type
        raw_boxes = endnodes[0::3]  # bbox outputs: indices 0, 3, 6
        raw_scores = endnodes[1::3]  # score outputs: indices 1, 4, 7
        raw_kpts = endnodes[2::3]  # keypoint outputs: indices 2, 5, 8

        # Process scores - reshape and concatenate
        scores = [
            np.reshape(s, (s.shape[0], s.shape[1] * s.shape[2], s.shape[3]))
            for s in raw_scores
        ]
        scores = np.concatenate(scores, axis=1)

        # Take only the first class if num_classes is 1
        if num_classes == 1 and scores.shape[2] > 1:
            scores = scores[:, :, :1]  # Take only first class channel

        # Process keypoints - reshape
        kpts = [np.reshape(c, (-1, c.shape[1] * c.shape[2], 17, 3)) for c in raw_kpts]

        # Decode boxes and keypoints
        decoded_boxes, decoded_kpts = self._yolov8_decoding(
            raw_boxes, kpts, strides, image_dims, reg_max
        )

        # Reshape keypoints for NMS processing
        decoded_kpts = np.reshape(
            decoded_kpts, (batch_size, -1, 51)
        )  # 17 keypoints × 3

        # Ensure all arrays have the same number of proposals
        min_proposals = min(
            decoded_boxes.shape[1], scores.shape[1], decoded_kpts.shape[1]
        )
        decoded_boxes = decoded_boxes[:, :min_proposals, :]
        scores = scores[:, :min_proposals, :]
        decoded_kpts = decoded_kpts[:, :min_proposals, :]

        # Concatenate predictions for NMS
        predictions = np.concatenate([decoded_boxes, scores, decoded_kpts], axis=2)

        # Apply NMS
        nms_results = self._apply_pose_nms(
            predictions,
            conf_thres=score_thres,
            iou_thres=iou_thres,
            max_det=max_detections,
        )

        # Format output to match original implementation
        output = {
            "bboxes": np.zeros((batch_size, max_detections, 4)),
            "keypoints": np.zeros((batch_size, max_detections, 17, 2)),
            "joint_scores": np.zeros((batch_size, max_detections, 17, 1)),
            "scores": np.zeros((batch_size, max_detections, 1)),
        }

        for b in range(batch_size):
            num_detections = nms_results[b]["num_detections"]
            if num_detections > 0:
                output["bboxes"][b, :num_detections] = nms_results[b]["bboxes"]
                output["keypoints"][b, :num_detections] = nms_results[b]["keypoints"][
                    ..., :2
                ]
                output["joint_scores"][b, :num_detections, ..., 0] = _sigmoid(
                    nms_results[b]["keypoints"][..., 2]
                )
                output["scores"][b, :num_detections, ..., 0] = nms_results[b]["scores"]

        # Scale coordinates to original image size if needed
        if original_shape is not None:
            output = self._scale_output_to_original(output, original_shape, image_dims)
        # print(output["bboxes"].shape)

        keypoint_result = [
            KeypointResult(
                keypoints=output["keypoints"][i],
                scores=output["scores"][i],
                boxes=output["bboxes"][i],
                joint_scores=output["joint_scores"][i],
            )
            for i in range(batch_size)
        ]
        return keypoint_result

        # except Exception as e:
        #     logger.error(f"Error in YOLOv8 pose estimation postprocessing: {str(e)}")
        #     raise RuntimeError(f"YOLOv8 pose estimation postprocessing failed: {str(e)}") from e

    def _scale_output_to_original(
        self,
        output: Dict[str, np.ndarray],
        original_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        """
        Scale output coordinates to original image size.

        Args:
            output: Dictionary containing bboxes, keypoints, etc.
            original_shape: Original image shape (height, width)
            input_shape: Input image shape (height, width)

        Returns:
            Scaled output dictionary
        """
        # Calculate scaling factors
        scale_y = original_shape[0] / input_shape[0]
        scale_x = original_shape[1] / input_shape[1]

        # Scale bounding boxes
        if "bboxes" in output:
            bboxes = output["bboxes"].copy()
            bboxes[..., [0, 2]] *= scale_x  # x coordinates
            bboxes[..., [1, 3]] *= scale_y  # y coordinates
            output["bboxes"] = bboxes

        # Scale keypoints
        if "keypoints" in output:
            keypoints = output["keypoints"].copy()
            keypoints[..., 0] *= scale_x  # x coordinates
            keypoints[..., 1] *= scale_y  # y coordinates
            output["keypoints"] = keypoints

        return output

    def _yolov8_decoding(
        self,
        raw_boxes: List[np.ndarray],
        raw_kpts: List[np.ndarray],
        strides: List[int],
        image_dims: Tuple[int, int],
        reg_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode YOLOv8 pose estimation outputs.

        Args:
            raw_boxes: List of bbox outputs from different scales
            raw_kpts: List of keypoint outputs from different scales
            strides: List of stride values for each scale
            image_dims: Input image dimensions (height, width)
            reg_max: Maximum regression value

        Returns:
            Tuple of (decoded_boxes, decoded_keypoints)
        """
        boxes = None
        decoded_kpts = None

        for box_distribute, kpts, stride in zip(raw_boxes, raw_kpts, strides):
            # Create grid
            shape = [int(x / stride) for x in image_dims]
            grid_x = np.arange(shape[1]) + 0.5
            grid_y = np.arange(shape[0]) + 0.5
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            ct_row = grid_y.flatten() * stride
            ct_col = grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # Box distribution to distance
            reg_range = np.arange(reg_max + 1)
            box_distribute = np.reshape(
                box_distribute,
                (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1),
            )
            box_distance = _softmax(box_distribute)
            box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
            box_distance = np.sum(box_distance, axis=-1)
            box_distance = box_distance * stride

            # Decode box
            box_distance = np.concatenate(
                [box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1
            )
            decode_box = np.expand_dims(center, axis=0) + box_distance

            xmin = decode_box[:, :, 0]
            ymin = decode_box[:, :, 1]
            xmax = decode_box[:, :, 2]
            ymax = decode_box[:, :, 3]

            xywh_box = np.transpose(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin],
                [1, 2, 0],
            )
            boxes = (
                xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
            )

            # Keypoints decoding
            kpts_decoded = kpts.copy()
            kpts_decoded[..., :2] *= 2
            kpts_decoded[..., :2] = stride * (
                kpts_decoded[..., :2] - 0.5
            ) + np.expand_dims(center[..., :2], axis=1)
            decoded_kpts = (
                kpts_decoded
                if decoded_kpts is None
                else np.concatenate([decoded_kpts, kpts_decoded], axis=1)
            )

        return boxes, decoded_kpts

    def _apply_pose_nms(
        self,
        predictions: np.ndarray,
        conf_thres: float = 0.1,
        iou_thres: float = 0.45,
        max_det: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression to pose estimation predictions.

        Args:
            predictions: Predictions array with shape (batch_size, num_proposals, features)
                        where features = [x, y, w, h, conf, kpt1_x, kpt1_y, kpt1_v, ...]
            conf_thres: Confidence threshold for NMS
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections to keep after NMS

        Returns:
            List of per-batch detection dictionaries
        """
        assert 0 <= conf_thres <= 1, f"Invalid confidence threshold {conf_thres}"
        assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}"

        n_kpts = 17  # Fixed for COCO pose estimation
        nc = predictions.shape[2] - n_kpts * 3 - 4  # number of classes
        xc = predictions[..., 4] > conf_thres  # candidates

        ki = 4 + nc  # keypoints start index
        output = []

        for xi, x in enumerate(predictions):  # image index, image inference
            x = x[xc[xi]]  # Filter by confidence

            # If none remain, process next image
            if not x.shape[0]:
                output.append(
                    {
                        "bboxes": np.zeros((0, 4)),
                        "keypoints": np.zeros((0, n_kpts, 3)),
                        "scores": np.zeros((0)),
                        "num_detections": 0,
                    }
                )
                continue

            # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
            boxes = xywh_to_xyxy(x[:, :4])
            kpts = x[:, ki:]

            conf = np.expand_dims(x[:, 4:ki].max(1), 1)
            j = np.expand_dims(x[:, 4:ki].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, kpts), 1)[keep]

            # Sort by confidence
            x = x[x[:, 4].argsort()[::-1]]

            boxes = x[:, :4]
            conf = x[:, 4:5]
            preds = np.hstack([boxes.astype(np.float32), conf.astype(np.float32)])

            # Apply NMS using external library or fallback
            try:
                from hailo_model_zoo.core.postprocessing.cython_utils.cython_nms import (
                    nms as cnms,
                )

                keep = cnms(preds, iou_thres)
            except ImportError:
                # Fallback to custom NMS implementation
                keep = self._custom_nms(boxes, conf.squeeze(), iou_thres)

            if keep.shape[0] > max_det:
                keep = keep[:max_det]

            out = x[keep]
            scores = out[:, 4]
            boxes = out[:, :4]
            kpts = out[:, 6:]
            kpts = np.reshape(kpts, (-1, n_kpts, 3))

            output.append(
                {
                    "bboxes": boxes,
                    "keypoints": kpts,
                    "scores": scores,
                    "num_detections": int(scores.shape[0]),
                }
            )

        return output

    def _custom_nms(
        self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> np.ndarray:
        """
        Custom NMS implementation as fallback.

        Args:
            boxes: Bounding boxes in xyxy format
            scores: Confidence scores
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        # Sort by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]

        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        keep = []
        while len(sorted_indices) > 0:
            # Pick the box with highest score
            current = sorted_indices[0]
            keep.append(current)

            if len(sorted_indices) == 1:
                break

            # Calculate IoU with remaining boxes
            remaining = sorted_indices[1:]

            # Calculate intersection coordinates
            x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
            y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
            x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
            y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])

            # Calculate intersection area
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

            # Calculate union area
            union = areas[current] + areas[remaining] - intersection

            # Calculate IoU
            iou = intersection / (union + 1e-6)

            # Keep boxes with IoU below threshold
            sorted_indices = remaining[iou <= iou_threshold]

        return np.array(keep, dtype=np.int32)

    def _create_empty_keypoint_result(self) -> KeypointResult:
        """
        Create an empty keypoint result.

        Returns:
            Empty KeypointResult
        """
        return KeypointResult(
            keypoints=np.empty((0, self.config.num_keypoints, 3), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            boxes=np.empty((0, 4), dtype=np.float32),
        )


@CALLBACK_REGISTRY.registryPostProcessor("openpose")
class OpenPosePosePostprocessor(BasePostprocessor):
    """
    Postprocessor for OpenPose-style pose estimation models.

    This class handles the conversion of heatmaps and Part Affinity Fields (PAFs)
    into structured keypoint results using traditional pose estimation algorithms.

    Expected input format:
    - Heatmaps: (batch, height, width, num_keypoints+1) - last channel is background
    - PAFs: (batch, height, width, num_pafs) - Part Affinity Fields
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the OpenPose pose estimation postprocessor.

        Args:
            config: Postprocessing configuration. If None, default config is used.
        """
        super().__init__(config)

        # Set pose estimation specific defaults
        if not self.config.kp:
            self.config.kp = True

        # Validate configuration
        self._validate_config()

        # Set up keypoint names if not provided
        if self.config.keypoint_names is None:
            self.config.keypoint_names = COCO_KEYPOINT_NAMES

        if self.config.keypoint_connections is None:
            self.config.keypoint_connections = COCO_JOINT_PAIRS

        logger.info(
            f"Initialized OpenPosePosePostprocessor with {self.config.num_keypoints} keypoints"
        )

    def _validate_config(self) -> None:
        """
        Validate the postprocessing configuration.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.config.kp_conf_threshold < 0 or self.config.kp_conf_threshold > 1:
            raise ValueError(
                f"Keypoint confidence threshold must be between 0 and 1, "
                f"got {self.config.kp_conf_threshold}"
            )

        if (
            self.config.kp_visibility_threshold < 0
            or self.config.kp_visibility_threshold > 1
        ):
            raise ValueError(
                f"Keypoint visibility threshold must be between 0 and 1, "
                f"got {self.config.kp_visibility_threshold}"
            )

        if self.config.num_keypoints <= 0:
            raise ValueError(
                f"Number of keypoints must be positive, got {self.config.num_keypoints}"
            )

        if self.config.kp_max_persons <= 0:
            raise ValueError(
                f"Maximum persons must be positive, got {self.config.kp_max_persons}"
            )

    NODE_LOCAL_DICT = {
        (160, 1): 0,
        (160, 2): 1,
        (160, 34): 2,
        (160, 2): 3,
        (160, 17): 4,
        (160, 2): 5,
    }

    def postprocess(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> KeypointResult:
        """
        Postprocess raw OpenPose outputs (heatmaps and PAFs).

        Args:
            raw_outputs: Dictionary containing heatmaps and PAFs
            original_shape: Original image shape (height, width) for coordinate scaling

        Returns:
            KeypointResult containing processed keypoint detections

        Raises:
            ValueError: If output format is not supported
            RuntimeError: If postprocessing fails
        """
        try:
            results = centerpose_postprocessing(raw_outputs)

            return KeypointResult(
                keypoints=results["keypoints"],
                scores=results["scores"],
                boxes=results["bboxes"],
                joint_scores=results["joint_scores"],
            )

        except Exception as e:
            logger.error(f"Error in OpenPose postprocessing: {str(e)}")
            raise RuntimeError(f"OpenPose postprocessing failed: {str(e)}") from e

    def _extract_heatmaps_and_pafs(
        self, raw_outputs: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract heatmaps and PAFs from raw outputs.

        Args:
            raw_outputs: Dictionary of raw model outputs

        Returns:
            Tuple of (heatmaps, pafs)
        """
        # Try common naming patterns
        heatmap_names = ["heatmaps", "heatmap", "keypoints", "kpts"]
        paf_names = ["pafs", "paf", "limbs", "connections"]

        heatmaps = None
        pafs = None

        # Look for heatmaps
        for name in heatmap_names:
            if name in raw_outputs:
                heatmaps = raw_outputs[name]
                break

        # Look for PAFs
        for name in paf_names:
            if name in raw_outputs:
                pafs = raw_outputs[name]
                break

        # If not found by name, try to infer from output order and shapes
        if heatmaps is None or pafs is None:
            outputs = list(raw_outputs.values())
            if len(outputs) >= 2:
                # Assume first output is heatmaps, second is PAFs
                heatmaps = outputs[0]
                pafs = outputs[1]

        if heatmaps is None or pafs is None:
            raise ValueError("Could not identify heatmaps and PAFs in raw outputs")

        return heatmaps, pafs

    def _scale_heatmaps_and_pafs(
        self,
        heatmaps: np.ndarray,
        pafs: np.ndarray,
        original_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale heatmaps and PAFs to original image size.

        Args:
            heatmaps: Heatmaps array
            pafs: PAFs array
            original_shape: Original image shape (height, width)

        Returns:
            Tuple of (scaled_heatmaps, scaled_pafs)
        """
        height, width = original_shape

        # Resize heatmaps with stride
        heatmaps = cv2.resize(
            heatmaps,
            (0, 0),
            fx=HEATMAP_STRIDE,
            fy=HEATMAP_STRIDE,
            interpolation=cv2.INTER_CUBIC,
        )

        # Remove padding if present (assuming symmetric padding)
        pad_h = max(0, (heatmaps.shape[0] - height) // 2)
        pad_w = max(0, (heatmaps.shape[1] - width) // 2)

        if pad_h > 0 or pad_w > 0:
            heatmaps = heatmaps[
                pad_h : heatmaps.shape[0] - pad_h, pad_w : heatmaps.shape[1] - pad_w, :
            ]

        # Resize to exact original size
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)

        # Same for PAFs
        pafs = cv2.resize(
            pafs,
            (0, 0),
            fx=HEATMAP_STRIDE,
            fy=HEATMAP_STRIDE,
            interpolation=cv2.INTER_CUBIC,
        )

        if pad_h > 0 or pad_w > 0:
            pafs = pafs[pad_h : pafs.shape[0] - pad_h, pad_w : pafs.shape[1] - pad_w, :]

        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs

    def _extract_keypoints(
        self, heatmap: np.ndarray, all_keypoints: List, total_keypoint_num: int
    ) -> int:
        """
        Extract keypoints from a single heatmap channel.

        Args:
            heatmap: Single channel heatmap
            all_keypoints: List to append extracted keypoints
            total_keypoint_num: Current total number of keypoints

        Returns:
            Number of keypoints extracted from this heatmap
        """
        # Apply threshold to heatmap
        heatmap[heatmap < 0.1] = 0

        # Add borders for peak detection
        heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")

        # Extract different regions for peak detection
        heatmap_center = heatmap_with_borders[1:-1, 1:-1]
        heatmap_left = heatmap_with_borders[1:-1, 2:]
        heatmap_right = heatmap_with_borders[1:-1, :-2]
        heatmap_up = heatmap_with_borders[2:, 1:-1]
        heatmap_down = heatmap_with_borders[:-2, 1:-1]

        # Find peaks (local maxima)
        heatmap_peaks = (
            (heatmap_center > heatmap_left)
            & (heatmap_center > heatmap_right)
            & (heatmap_center > heatmap_up)
            & (heatmap_center > heatmap_down)
        )

        # Remove border effects
        heatmap_peaks = heatmap_peaks[1:-1, 1:-1]

        # Get keypoint coordinates
        keypoints = list(
            zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0])
        )  # (w, h)
        keypoints = sorted(keypoints, key=itemgetter(0))

        # Non-maximum suppression
        suppressed = np.zeros(len(keypoints), np.uint8)
        keypoints_with_score_and_id = []
        keypoint_num = 0

        for i in range(len(keypoints)):
            if suppressed[i]:
                continue

            # Suppress nearby keypoints
            for j in range(i + 1, len(keypoints)):
                distance = math.sqrt(
                    (keypoints[i][0] - keypoints[j][0]) ** 2
                    + (keypoints[i][1] - keypoints[j][1]) ** 2
                )
                if distance < 6:
                    suppressed[j] = 1

            # Add keypoint with score and ID
            keypoint_with_score_and_id = (
                keypoints[i][0],  # x coordinate
                keypoints[i][1],  # y coordinate
                heatmap[keypoints[i][1], keypoints[i][0]],  # confidence score
                total_keypoint_num + keypoint_num,  # unique ID
            )
            keypoints_with_score_and_id.append(keypoint_with_score_and_id)
            keypoint_num += 1

        all_keypoints.append(keypoints_with_score_and_id)
        return keypoint_num

    def _group_keypoints(
        self,
        all_keypoints_by_type: List,
        pafs: np.ndarray,
        pose_entry_size: int = 20,
        min_paf_score: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Group keypoints into poses using Part Affinity Fields.

        Args:
            all_keypoints_by_type: List of keypoints for each type
            pafs: Part Affinity Fields
            pose_entry_size: Size of pose entry array
            min_paf_score: Minimum PAF score threshold

        Returns:
            Tuple of (pose_entries, all_keypoints)
        """
        pose_entries = []
        all_keypoints = np.array(
            [item for sublist in all_keypoints_by_type for item in sublist]
        )

        for part_id in range(len(BODY_PARTS_PAF_IDS)):
            part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
            kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
            kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
            num_kpts_a = len(kpts_a)
            num_kpts_b = len(kpts_b)
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            if num_kpts_a == 0 and num_kpts_b == 0:
                continue
            elif num_kpts_a == 0:
                # Only 'b' keypoints available
                for i in range(num_kpts_b):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                            num += 1
                            continue
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_b_id] = kpts_b[i][3]
                        pose_entry[-1] = 1  # num keypoints in pose
                        pose_entry[-2] = kpts_b[i][2]  # pose score
                        pose_entries.append(pose_entry)
                continue
            elif num_kpts_b == 0:
                # Only 'a' keypoints available
                for i in range(num_kpts_a):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                            num += 1
                            continue
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = kpts_a[i][3]
                        pose_entry[-1] = 1
                        pose_entry[-2] = kpts_a[i][2]
                        pose_entries.append(pose_entry)
                continue

            # Find connections between keypoints using PAFs
            connections = []
            for i in range(num_kpts_a):
                kpt_a = np.array(kpts_a[i][0:2])
                for j in range(num_kpts_b):
                    kpt_b = np.array(kpts_b[j][0:2])

                    # Calculate connection score using PAF
                    connection_score = self._calculate_paf_score(
                        kpt_a, kpt_b, part_pafs, min_paf_score
                    )

                    if connection_score > 0:
                        score_all = connection_score + kpts_a[i][2] + kpts_b[j][2]
                        connections.append([i, j, connection_score, score_all])

            if len(connections) > 0:
                connections = sorted(connections, key=itemgetter(2), reverse=True)

            # Filter connections to avoid duplicates
            num_connections = min(num_kpts_a, num_kpts_b)
            has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
            has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
            filtered_connections = []

            for row in range(len(connections)):
                if len(filtered_connections) == num_connections:
                    break
                i, j, cur_point_score = connections[row][0:3]
                if not has_kpt_a[i] and not has_kpt_b[j]:
                    filtered_connections.append(
                        [kpts_a[i][3], kpts_b[j][3], cur_point_score]
                    )
                    has_kpt_a[i] = 1
                    has_kpt_b[j] = 1

            connections = filtered_connections
            if len(connections) == 0:
                continue

            # Update pose entries based on connections
            if part_id == 0:
                # First body part - create new pose entries
                pose_entries = [
                    np.ones(pose_entry_size) * -1 for _ in range(len(connections))
                ]
                for i in range(len(connections)):
                    pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                    pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                    pose_entries[i][-1] = 2
                    pose_entries[i][-2] = (
                        np.sum(all_keypoints[connections[i][0:2], 2])
                        + connections[i][2]
                    )
            elif part_id == 17 or part_id == 18:
                # Special handling for ear connections
                for i in range(len(connections)):
                    for j in range(len(pose_entries)):
                        if (
                            pose_entries[j][kpt_a_id] == connections[i][0]
                            and pose_entries[j][kpt_b_id] == -1
                        ):
                            pose_entries[j][kpt_b_id] = connections[i][1]
                        elif (
                            pose_entries[j][kpt_b_id] == connections[i][1]
                            and pose_entries[j][kpt_a_id] == -1
                        ):
                            pose_entries[j][kpt_a_id] = connections[i][0]
                continue
            else:
                # Regular body parts
                for i in range(len(connections)):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0]:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                            num += 1
                            pose_entries[j][-1] += 1
                            pose_entries[j][-2] += (
                                all_keypoints[connections[i][1], 2] + connections[i][2]
                            )

                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = connections[i][0]
                        pose_entry[kpt_b_id] = connections[i][1]
                        pose_entry[-1] = 2
                        pose_entry[-2] = (
                            np.sum(all_keypoints[connections[i][0:2], 2])
                            + connections[i][2]
                        )
                        pose_entries.append(pose_entry)

        # Filter poses by minimum keypoints and score
        filtered_entries = []
        for i in range(len(pose_entries)):
            if pose_entries[i][-1] < 3 or (
                pose_entries[i][-2] / pose_entries[i][-1] < 0.2
            ):
                continue
            filtered_entries.append(pose_entries[i])

        pose_entries = np.asarray(filtered_entries)
        return pose_entries, all_keypoints

    def _calculate_paf_score(
        self,
        kpt_a: np.ndarray,
        kpt_b: np.ndarray,
        part_pafs: np.ndarray,
        min_paf_score: float,
    ) -> float:
        """
        Calculate PAF score between two keypoints.

        Args:
            kpt_a: First keypoint coordinates
            kpt_b: Second keypoint coordinates
            part_pafs: Part Affinity Fields for this connection
            min_paf_score: Minimum PAF score threshold

        Returns:
            Connection score
        """
        # Calculate vector between keypoints
        vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
        vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)

        if vec_norm == 0:
            return 0

        # Normalize vector
        vec[0] /= vec_norm
        vec[1] /= vec_norm

        # Sample points along the connection
        point_num = 10
        x, y = linspace2d(kpt_a, kpt_b, point_num)

        passed_point_score = 0
        passed_point_num = 0

        for point_idx in range(point_num):
            px = int(round(x[point_idx]))
            py = int(round(y[point_idx]))

            # Ensure coordinates are within bounds
            if 0 <= py < part_pafs.shape[0] and 0 <= px < part_pafs.shape[1]:
                paf = part_pafs[py, px, 0:2]
                cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]

                if cur_point_score > min_paf_score:
                    passed_point_score += cur_point_score
                    passed_point_num += 1

        success_ratio = passed_point_num / point_num

        if passed_point_num > 0 and success_ratio > 0.8:
            ratio = passed_point_score / passed_point_num
            height_n = part_pafs.shape[0] // 2
            ratio += min(height_n / vec_norm - 1, 0)
            return ratio

        return 0

    def _convert_to_coco_format(
        self, pose_entries: np.ndarray, all_keypoints: np.ndarray
    ) -> Tuple[List, List]:
        """
        Convert pose entries to COCO keypoint format.

        Args:
            pose_entries: Array of pose entries
            all_keypoints: Array of all detected keypoints

        Returns:
            Tuple of (coco_keypoints, scores)
        """
        coco_keypoints = []
        scores = []

        # Mapping from OpenPose to COCO keypoint order
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue

            keypoints = [0] * 17 * 3  # 17 keypoints × 3 (x, y, visibility)
            person_score = pose_entries[n][-2]
            position_id = -1

            for keypoint_id in pose_entries[n][:-2]:
                position_id += 1
                if position_id == 1:  # Skip 'neck' in COCO format
                    continue

                cx, cy, score, visibility = 0, 0, 0, 0  # Default: keypoint not found

                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                    cx = cx + 0.5  # Sub-pixel adjustment
                    cy = cy + 0.5
                    visibility = 1 if score > self.config.kp_visibility_threshold else 0

                # Map to COCO format
                coco_idx = to_coco_map[position_id]
                if coco_idx >= 0:  # Valid COCO keypoint
                    keypoints[coco_idx * 3 + 0] = cx
                    keypoints[coco_idx * 3 + 1] = cy
                    keypoints[coco_idx * 3 + 2] = visibility

            coco_keypoints.append(keypoints)
            # Calculate final score (person score × number of visible keypoints - 1 for neck)
            scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))

        return coco_keypoints, scores

    def _generate_bounding_boxes(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Generate bounding boxes from keypoints.

        Args:
            keypoints: Keypoints array with shape (N, 17, 3)

        Returns:
            Bounding boxes array with shape (N, 4) in format [x1, y1, x2, y2]
        """
        if len(keypoints) == 0:
            return np.empty((0, 4), dtype=np.float32)

        boxes = []

        for person_kpts in keypoints:
            # Get visible keypoints
            visible_mask = person_kpts[:, 2] > 0

            if not np.any(visible_mask):
                # No visible keypoints, create a default small box
                boxes.append([0, 0, 1, 1])
                continue

            visible_kpts = person_kpts[visible_mask]

            # Calculate bounding box from visible keypoints
            x_coords = visible_kpts[:, 0]
            y_coords = visible_kpts[:, 1]

            x_min = np.min(x_coords)
            y_min = np.min(y_coords)
            x_max = np.max(x_coords)
            y_max = np.max(y_coords)

            # Add some padding
            padding = 0.1
            width = x_max - x_min
            height = y_max - y_min

            x_min = max(0, x_min - padding * width)
            y_min = max(0, y_min - padding * height)
            x_max = x_max + padding * width
            y_max = y_max + padding * height

            boxes.append([x_min, y_min, x_max, y_max])

        return np.array(boxes, dtype=np.float32)

    def _create_empty_keypoint_result(self) -> KeypointResult:
        """
        Create an empty keypoint result.

        Returns:
            Empty KeypointResult
        """
        return KeypointResult(
            keypoints=np.empty((0, self.config.num_keypoints, 3), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            boxes=np.empty((0, 4), dtype=np.float32),
        )


# Factory function for creating pose estimation postprocessors
def create_pose_postprocessor(
    model_type: str, config: Optional[PostprocessConfig] = None
) -> BasePostprocessor:
    """
    Factory function to create pose estimation postprocessors.

    Args:
        model_type: Type of pose estimation model ("yolov8", "openpose", "centerpose")
        config: Postprocessing configuration

    Returns:
        Appropriate postprocessor instance

    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()

    if model_type in ["yolov8", "yolov8_pose", "nanodet_v8"]:
        return YOLOv8PosePostprocessor(config)
    elif model_type in ["openpose", "pose_estimation"]:
        return OpenPosePosePostprocessor(config)
    elif model_type == "centerpose":
        # CenterPose would need its own implementation
        # For now, we'll use OpenPose as fallback
        logger.warning("CenterPose postprocessor not implemented, using OpenPose")
        return OpenPosePosePostprocessor(config)
    else:
        raise ValueError(
            f"Unsupported pose estimation model type: {model_type}. "
            f"Supported types: yolov8, openpose, centerpose"
        )


# Compatibility wrapper function that matches the original API
def pose_estimation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    """
    Compatibility wrapper for pose estimation postprocessing.

    This function provides the same interface as the original pose_estimation_postprocessing
    function for backward compatibility.

    Args:
        endnodes: List of raw model outputs or dict for different model types
        device_pre_post_layers: Unused parameter for compatibility
        **kwargs: Additional parameters including meta_arch to determine model type

    Returns:
        Processed pose estimation results
    """
    meta_arch = kwargs.get("meta_arch", "")

    if meta_arch == "centerpose":
        # CenterPose handling - would need specific implementation
        logger.warning("CenterPose processing not fully implemented")
        # For now, return empty result
        return {"predictions": []}

    elif meta_arch == "nanodet_v8" or "yolov8" in meta_arch.lower():
        # YOLOv8 pose estimation
        postprocessor = YOLOv8PosePostprocessor()
        result = postprocessor.postprocess(endnodes, **kwargs)
        return result

    else:
        # OpenPose style processing (original implementation)
        if not isinstance(endnodes, (list, tuple)) or len(endnodes) < 2:
            raise ValueError("OpenPose requires at least heatmaps and PAFs as input")

        # Use original OpenPose processing logic
        postprocessor = OpenPosePosePostprocessor()

        # Convert endnodes to dictionary format for OpenPose
        raw_outputs = {"heatmaps": endnodes[0], "pafs": endnodes[1]}

        # Extract image info if available
        image_info = kwargs.get("gt_images", {})
        original_shape = None
        if "orig_shape" in image_info and len(image_info["orig_shape"]) > 0:
            original_shape = image_info["orig_shape"][0][:2]  # (height, width)

        result = postprocessor.postprocess(raw_outputs, original_shape)

        # Convert to COCO format for compatibility
        coco_result_list = []
        if len(result.keypoints) > 0:
            for idx in range(len(result.keypoints)):
                # Convert keypoints to COCO format (flattened)
                kpts_flat = result.keypoints[idx].flatten().tolist()

                coco_result_list.append(
                    {
                        "image_id": (
                            image_info.get("image_id", [0])[0]
                            if "image_id" in image_info
                            else 0
                        ),
                        "category_id": 1,  # person
                        "keypoints": kpts_flat,
                        "score": float(result.scores[idx]),
                    }
                )

        return {"predictions": coco_result_list}
