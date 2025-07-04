"""
YOLOv8 Detection Postprocessor

This module implements postprocessing for YOLOv8 object detection models.
It handles various output formats and provides comprehensive detection results.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

from ..base import (
    BasePostprocessor,
    PostprocessConfig,
    # DetectionResult,
    non_max_suppression,
    scale_boxes,
)
from ..results.det import DetectionResult
from hailo_toolbox.inference.core import CALLBACK_REGISTRY

logger = logging.getLogger(__name__)


@CALLBACK_REGISTRY.registryPostProcessor("yolov8det", "yolo11det")
class YOLOv8DetPostprocessor(BasePostprocessor):
    """
    Postprocessor for YOLOv8 object detection models.

    This class handles the conversion of raw YOLOv8 detection outputs into
    structured detection results with proper NMS and coordinate scaling.

    Supported output formats:
    1. Non-NMS format: (batch_size, num_detections, 6) - [x1, y1, x2, y2, conf, label]
    2. NMS format: Multiple outputs that need to be concatenated before NMS processing
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the YOLOv8 detection postprocessor.

        Args:
            config: Postprocessing configuration. If None, default config is used.
        """
        super().__init__(config)

        # Validate configuration
        self._validate_config()

        # Set up class names if not provided
        if self.config.class_names is None:
            self.config.class_names = [
                f"class_{i}" for i in range(self.config.num_classes)
            ]

        logger.info(
            f"Initialized YOLOv8DetPostprocessor with {self.config.num_classes} classes, "
            f"NMS enabled: {self.config.nms}"
        )

    def _validate_config(self) -> None:
        """
        Validate the postprocessing configuration.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.config.det_conf_threshold < 0 or self.config.det_conf_threshold > 1:
            raise ValueError(
                f"Detection confidence threshold must be between 0 and 1, "
                f"got {self.config.det_conf_threshold}"
            )

        if self.config.nms_iou_threshold < 0 or self.config.nms_iou_threshold > 1:
            raise ValueError(
                f"NMS IoU threshold must be between 0 and 1, "
                f"got {self.config.nms_iou_threshold}"
            )

        if self.config.num_classes <= 0:
            raise ValueError(
                f"Number of classes must be positive, got {self.config.num_classes}"
            )

        if self.config.det_max_detections <= 0:
            raise ValueError(
                f"Maximum detections must be positive, got {self.config.det_max_detections}"
            )

    def postprocess(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> DetectionResult:
        """
        Postprocess raw YOLOv8 detection outputs.

        Args:
            raw_outputs: Dictionary containing raw model outputs
            original_shape: Original image shape (height, width) for coordinate scaling

        Returns:
            DetectionResult containing processed detections

        Raises:
            ValueError: If output format is not supported
            RuntimeError: If postprocessing fails
        """
        # try:
        if self.config.nms:
            # NMS format: Multiple outputs need to be concatenated
            return self._postprocess_with_nms(raw_outputs, original_shape)
        else:
            # Non-NMS format: Direct output format [x1, y1, x2, y2, conf, label]
            return self._postprocess_without_nms(raw_outputs, original_shape)

        # except Exception as e:
        #     logger.error(f"Error in detection postprocessing: {str(e)}")
        #     raise RuntimeError(f"Detection postprocessing failed: {str(e)}") from e

    def _postprocess_without_nms(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> DetectionResult:
        """
        Postprocess detection outputs without NMS.
        Expected format: [batch, number, 6] where 6 = [x1, y1, x2, y2, conf, label]

        Args:
            raw_outputs: Dictionary containing raw model outputs
            original_shape: Original image shape (height, width) for coordinate scaling

        Returns:
            DetectionResult containing processed detections
        """
        # Extract the main detection output
        detection_output = self._extract_detection_output(raw_outputs)

        # Validate output format
        if len(detection_output.shape) != 3:
            raise ValueError(
                f"Expected 3D detection output for non-NMS format, got shape {detection_output.shape}"
            )

        batch_size, num_detections, num_features = detection_output.shape
        if num_features != 6:
            raise ValueError(
                f"Expected 6 features for non-NMS format [x1, y1, x2, y2, conf, label], "
                f"got {num_features}"
            )

        # Process first batch only
        detection_output = detection_output[0]  # Shape: (num_detections, 6)

        # Extract components
        boxes = detection_output[:, :4].astype(np.float32)  # [x1, y1, x2, y2]
        boxes = self.yxyx_to_xyxy(boxes)
        scores = detection_output[:, 4].astype(np.float32)  # confidence
        class_ids = detection_output[:, 5].astype(np.int32)  # class labels

        # Apply confidence filtering
        valid_detections = scores >= self.config.det_conf_threshold
        if not np.any(valid_detections):
            logger.debug("No detections above confidence threshold")
            return self._create_empty_result()

        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        class_ids = class_ids[valid_detections]

        # Limit number of detections by confidence ranking
        if len(boxes) > self.config.det_max_detections:
            # Sort by confidence and keep top detections
            top_indices = np.argsort(scores)[::-1][: self.config.det_max_detections]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            class_ids = class_ids[top_indices]

        # Scale boxes to original image size if needed
        if original_shape is not None:
            boxes = scale_boxes(boxes, original_shape, self.config.input_shape)
            # Clip boxes to image boundaries
            boxes = self._clip_boxes(boxes, original_shape)

        logger.debug(f"Postprocessed {len(boxes)} detections (without NMS)")

        return DetectionResult(boxes=boxes, scores=scores, class_ids=class_ids)

    def yxyx_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from yxyx format to xyxy format.
        """
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 1]
        boxes_xyxy[:, 1] = boxes[:, 0]
        boxes_xyxy[:, 2] = boxes[:, 3]
        boxes_xyxy[:, 3] = boxes[:, 2]
        return boxes_xyxy

    def _postprocess_with_nms(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> DetectionResult:
        """
        Postprocess detection outputs with NMS.
        Multiple outputs are concatenated before NMS processing.

        Args:
            raw_outputs: Dictionary containing multiple raw model outputs
            original_shape: Original image shape (height, width) for coordinate scaling

        Returns:
            DetectionResult containing processed detections
        """
        # Concatenate all outputs
        concatenated_output = self._concatenate_outputs(raw_outputs)

        # Handle batch dimension
        if len(concatenated_output.shape) == 3:
            # Process first batch only
            concatenated_output = concatenated_output[0]

        # Parse detections based on output format
        boxes, scores, class_ids = self._parse_detections_for_nms(concatenated_output)

        # Apply confidence filtering
        valid_detections = scores >= self.config.det_conf_threshold
        if not np.any(valid_detections):
            logger.debug("No detections above confidence threshold")
            return self._create_empty_result()

        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        class_ids = class_ids[valid_detections]

        # Apply Non-Maximum Suppression
        keep_indices = self._apply_nms(boxes, scores, class_ids)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]

        # Limit number of detections
        if len(boxes) > self.config.det_max_detections:
            # Sort by confidence and keep top detections
            top_indices = np.argsort(scores)[::-1][: self.config.det_max_detections]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            class_ids = class_ids[top_indices]

        # Scale boxes to original image size if needed
        if original_shape is not None:
            boxes = scale_boxes(boxes, original_shape, self.config.input_shape)
            # Clip boxes to image boundaries
            boxes = self._clip_boxes(boxes, original_shape)

        logger.debug(f"Postprocessed {len(boxes)} detections (with NMS)")

        return DetectionResult(boxes=boxes, scores=scores, class_ids=class_ids)

    def _concatenate_outputs(self, raw_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate multiple model outputs for NMS processing.

        Args:
            raw_outputs: Dictionary of raw model outputs from different stages

        Returns:
            Concatenated output array

        Raises:
            ValueError: If outputs cannot be concatenated
        """
        if len(raw_outputs) == 0:
            raise ValueError("No outputs provided for concatenation")

        # Convert to list and sort by key for consistent ordering
        output_items = sorted(raw_outputs.items())
        outputs = [output for _, output in output_items]

        # Validate that all outputs have compatible shapes for concatenation
        first_output = outputs[0]
        if len(first_output.shape) < 2:
            raise ValueError(
                f"Expected at least 2D outputs, got shape {first_output.shape}"
            )

        # Check if all outputs have the same batch size and feature dimensions
        batch_size = first_output.shape[0] if len(first_output.shape) >= 3 else 1
        feature_dim = first_output.shape[-1]

        for i, output in enumerate(outputs[1:], 1):
            if len(output.shape) != len(first_output.shape):
                raise ValueError(
                    f"Output {i} has different number of dimensions: "
                    f"{len(output.shape)} vs {len(first_output.shape)}"
                )

            if len(output.shape) >= 3 and output.shape[0] != batch_size:
                raise ValueError(
                    f"Output {i} has different batch size: "
                    f"{output.shape[0]} vs {batch_size}"
                )

            if output.shape[-1] != feature_dim:
                raise ValueError(
                    f"Output {i} has different feature dimension: "
                    f"{output.shape[-1]} vs {feature_dim}"
                )

        try:
            # Concatenate along the detection dimension (axis=1 for 3D, axis=0 for 2D)
            concat_axis = -2 if len(first_output.shape) >= 3 else 0
            concatenated = np.concatenate(outputs, axis=concat_axis)

            logger.debug(
                f"Concatenated {len(outputs)} outputs into shape {concatenated.shape}"
            )

            return concatenated

        except Exception as e:
            raise ValueError(f"Failed to concatenate outputs: {str(e)}") from e

    def _extract_detection_output(
        self, raw_outputs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Extract the main detection output from raw outputs.

        Args:
            raw_outputs: Dictionary of raw model outputs

        Returns:
            Main detection output array

        Raises:
            ValueError: If no suitable output is found
        """
        # Try common output names
        common_names = ["output", "output0", "detections", "predictions", "boxes"]

        for name in common_names:
            if name in raw_outputs:
                return raw_outputs[name]

        # If no common name found, use the first output
        if len(raw_outputs) == 1:
            return list(raw_outputs.values())[0]

        # If multiple outputs, try to find the largest one (likely detections)
        largest_output = None
        largest_size = 0

        for name, output in raw_outputs.items():
            if output.size > largest_size:
                largest_size = output.size
                largest_output = output

        if largest_output is not None:
            logger.warning("Using largest output for detection postprocessing")
            return largest_output

        raise ValueError("Could not identify detection output in raw_outputs")

    def _parse_detections_for_nms(
        self, detection_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse detection output for NMS processing.
        Supports traditional YOLOv8 format with class predictions.

        Args:
            detection_output: Raw detection output array

        Returns:
            Tuple of (boxes, scores, class_ids)

        Raises:
            ValueError: If output format is not supported
        """
        if len(detection_output.shape) != 2:
            raise ValueError(
                f"Expected 2D detection output, got shape {detection_output.shape}"
            )

        num_detections, num_features = detection_output.shape

        # Determine output format
        expected_features_with_obj = (
            5 + self.config.num_classes
        )  # x, y, w, h, objectness, classes
        expected_features_without_obj = (
            4 + self.config.num_classes
        )  # x, y, w, h, classes

        if num_features == expected_features_with_obj:
            # Format: [x, y, w, h, objectness, class_0, class_1, ...]
            boxes_xywh = detection_output[:, :4]
            objectness = detection_output[:, 4]
            class_scores = detection_output[:, 5:]

            # Convert to confidence scores
            class_confidences = class_scores * objectness[:, np.newaxis]

        elif num_features == expected_features_without_obj:
            # Format: [x, y, w, h, class_0, class_1, ...]
            boxes_xywh = detection_output[:, :4]
            class_confidences = detection_output[:, 4:]

        else:
            raise ValueError(
                f"Unsupported detection output format for NMS processing. "
                f"Expected {expected_features_with_obj} or {expected_features_without_obj} features, "
                f"got {num_features}"
            )

        # Get best class for each detection
        class_ids = np.argmax(class_confidences, axis=1)
        scores = np.max(class_confidences, axis=1)

        # Convert boxes from xywh to xyxy format
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        return boxes_xyxy, scores, class_ids

    def _xywh_to_xyxy(self, boxes_xywh: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from xywh to xyxy format.

        Args:
            boxes_xywh: Boxes in xywh format (center_x, center_y, width, height)

        Returns:
            Boxes in xyxy format (x1, y1, x2, y2)
        """
        boxes_xyxy = np.zeros_like(boxes_xywh)

        # Calculate corners
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        return boxes_xyxy

    def _apply_nms(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray
    ) -> np.ndarray:
        """
        Apply Non-Maximum Suppression to detections.

        Args:
            boxes: Bounding boxes in xyxy format
            scores: Confidence scores
            class_ids: Class IDs

        Returns:
            Indices of detections to keep
        """
        if self.config.det_class_agnostic:
            # Apply NMS across all classes
            return non_max_suppression(
                boxes=boxes,
                scores=scores,
                iou_threshold=self.config.nms_iou_threshold,
                score_threshold=self.config.det_conf_threshold,
                max_detections=self.config.det_max_detections,
            )
        else:
            # Apply NMS per class
            keep_indices = []

            for class_id in np.unique(class_ids):
                class_mask = class_ids == class_id
                class_boxes = boxes[class_mask]
                class_scores = scores[class_mask]

                if len(class_boxes) == 0:
                    continue

                class_keep = non_max_suppression(
                    boxes=class_boxes,
                    scores=class_scores,
                    iou_threshold=self.config.nms_iou_threshold,
                    score_threshold=self.config.det_conf_threshold,
                    max_detections=self.config.det_max_detections,
                )

                # Convert back to original indices
                original_indices = np.where(class_mask)[0]
                keep_indices.extend(original_indices[class_keep])

            return np.array(keep_indices, dtype=np.int32)

    def _clip_boxes(
        self, boxes: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Clip bounding boxes to image boundaries.

        Args:
            boxes: Bounding boxes in xyxy format
            image_shape: Image shape (height, width)

        Returns:
            Clipped bounding boxes
        """
        if boxes.size == 0:
            return boxes

        height, width = image_shape

        # Clip coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)  # y2

        # Ensure x2 > x1 and y2 > y1
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)

        return boxes

    def _create_empty_result(self) -> DetectionResult:
        """
        Create an empty detection result.

        Returns:
            Empty DetectionResult
        """
        return DetectionResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int32),
        )

    def get_class_name(self, class_id: int) -> str:
        """
        Get class name for a given class ID.

        Args:
            class_id: Class ID

        Returns:
            Class name
        """
        if 0 <= class_id < len(self.config.class_names):
            return self.config.class_names[class_id]
        else:
            return f"unknown_{class_id}"

    def filter_by_classes(
        self, result: DetectionResult, class_ids: List[int]
    ) -> DetectionResult:
        """
        Filter detection results by specific class IDs.

        Args:
            result: Detection result to filter
            class_ids: List of class IDs to keep

        Returns:
            Filtered detection result
        """
        if len(result) == 0:
            return result

        # Create mask for desired classes
        class_mask = np.isin(result.class_ids, class_ids)

        if not np.any(class_mask):
            return self._create_empty_result()

        return DetectionResult(
            boxes=result.boxes[class_mask],
            scores=result.scores[class_mask],
            class_ids=result.class_ids[class_mask],
            masks=result.masks[class_mask] if result.masks is not None else None,
        )

    def filter_by_confidence(
        self, result: DetectionResult, min_confidence: float
    ) -> DetectionResult:
        """
        Filter detection results by minimum confidence threshold.

        Args:
            result: Detection result to filter
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered detection result
        """
        if len(result) == 0:
            return result

        # Create mask for confident detections
        conf_mask = result.scores >= min_confidence

        if not np.any(conf_mask):
            return self._create_empty_result()

        return DetectionResult(
            boxes=result.boxes[conf_mask],
            scores=result.scores[conf_mask],
            class_ids=result.class_ids[conf_mask],
            masks=result.masks[conf_mask] if result.masks is not None else None,
        )
