"""
YOLOv8 Segmentation Postprocessor

This module implements postprocessing for YOLOv8 instance segmentation models.
It handles detection postprocessing and mask generation from prototype coefficients.
Based on the reference implementation from yolov8segpostprocess.py.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

from ..base import (
    BasePostprocessor,
    PostprocessConfig,
)
from hailo_toolbox.process.results.seg import SegmentationResult
from hailo_toolbox.inference.core import CALLBACK_REGISTRY

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid activation function.

    Args:
        x: Input array

    Returns:
        Sigmoid activated array
    """
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Apply softmax activation function.

    Args:
        x: Input array

    Returns:
        Softmax activated array
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from xywh to xyxy format.

    Args:
        x: Boxes in xywh format (center_x, center_y, width, height)

    Returns:
        Boxes in xyxy format (x1, y1, x2, y2)
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Crop masks to bounding box regions.

    Args:
        masks: Masks array with shape [n, h, w]
        boxes: Bounding boxes with shape [n, 4] in xyxy format

    Returns:
        Cropped masks
    """
    n_masks, h, w = masks.shape
    integer_boxes = np.ceil(boxes).astype(int)
    x1, y1, x2, y2 = np.array_split(
        np.where(integer_boxes > 0, integer_boxes, 0), 4, axis=1
    )

    for k in range(n_masks):
        masks[k, : y1[k, 0], :] = 0
        masks[k, y2[k, 0] :, :] = 0
        masks[k, :, : x1[k, 0]] = 0
        masks[k, :, x2[k, 0] :] = 0

    return masks


def process_mask(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    shape: Tuple[int, int],
    upsample: bool = True,
    downsample: bool = False,
) -> Optional[np.ndarray]:
    """
    Process mask prototypes and coefficients to generate final masks.

    Args:
        protos: Mask prototypes with shape (h, w, c)
        masks_in: Mask coefficients with shape (n, c)
        bboxes: Bounding boxes with shape (n, 4)
        shape: Target shape (height, width)
        upsample: Whether to upsample masks to target shape
        downsample: Whether to downsample bounding boxes

    Returns:
        Processed masks with shape (n, h, w) or None if no masks
    """
    mh, mw, c = protos.shape
    ih, iw = shape

    # Generate masks from prototypes and coefficients
    masks = _sigmoid(masks_in @ protos.reshape((-1, c)).transpose((1, 0))).reshape(
        (-1, mh, mw)
    )

    downsampled_bboxes = bboxes.copy()
    if downsample:
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        if not masks.shape[0]:
            return None
        masks = cv2.resize(
            np.transpose(masks, axes=(1, 2, 0)), shape, interpolation=cv2.INTER_LINEAR
        )
        if len(masks.shape) == 2:
            masks = masks[..., np.newaxis]
        masks = np.transpose(masks, axes=(2, 0, 1))  # CHW

    if not downsample:
        masks = crop_mask(masks, downsampled_bboxes)  # CHW

    return masks


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    nm: int = 32,
    multi_label: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Args:
        prediction: Predictions with shape (batch_size, num_proposals, 4 + num_classes + 1 + nm)
        conf_thres: Confidence threshold for NMS
        iou_thres: IoU threshold for NMS
        max_det: Maximum number of detections to keep after NMS
        nm: Number of masks
        multi_label: Consider only best class per proposal or all conf_thresh passing proposals

    Returns:
        List of per image detections, each containing detection_boxes, mask, detection_classes, detection_scores
    """
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    mi = 5 + nc  # mask start index
    output = []

    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            output.append(
                {
                    "detection_boxes": np.zeros((0, 4)),
                    "mask": np.zeros((0, nm)),
                    "detection_classes": np.zeros((0,)),
                    "detection_scores": np.zeros((0,)),
                }
            )
            continue

        # Confidence = Objectness X Class Score
        x[:, 5:] *= x[:, 4:5]

        # (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        multi_label &= nc > 1
        if not multi_label:
            conf = np.expand_dims(x[:, 5:mi].max(1), 1)
            j = np.expand_dims(x[:, 5:mi].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, mask), 1)[keep]
        else:
            i, j = (x[:, 5:mi] > conf_thres).nonzero()
            x = np.concatenate(
                (boxes[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]), 1
            )

        # sort by confidence
        x = x[x[:, 4].argsort()[::-1]]

        # per-class NMS
        cls_shift = x[:, 5:6] * max_wh
        boxes = x[:, :4] + cls_shift
        conf = x[:, 4:5]
        preds = np.hstack([boxes.astype(np.float32), conf.astype(np.float32)])

        # Use simple NMS implementation
        keep = _nms(preds, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        # Ensure keep is integer array for indexing
        keep = keep.astype(int)
        out = x[keep]
        scores = out[:, 4]
        classes = out[:, 5]
        boxes = out[:, :4]
        masks = out[:, 6:]

        out = {
            "detection_boxes": boxes,
            "mask": masks,
            "detection_classes": classes,
            "detection_scores": scores,
        }
        output.append(out)

    return output


def _nms(dets: np.ndarray, thresh: float) -> np.ndarray:
    """
    Simple NMS implementation.

    Args:
        dets: Detections with shape (n, 5) - [x1, y1, x2, y2, score]
        thresh: IoU threshold

    Returns:
        Indices of detections to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def _yolov8_decoding(
    raw_boxes: List[np.ndarray],
    strides: List[int],
    image_dims: Tuple[int, int],
    reg_max: int,
) -> np.ndarray:
    """
    Decode YOLOv8 box predictions.

    Args:
        raw_boxes: List of raw box predictions for each scale
        strides: List of strides for each scale
        image_dims: Image dimensions (height, width)
        reg_max: Maximum regression range

    Returns:
        Decoded boxes in xywh format
    """
    boxes = None
    for box_distribute, stride in zip(raw_boxes, strides):
        # Get the shape of the feature map
        batch_size, height, width, channels = box_distribute.shape

        # Ensure channels is divisible by 4 * (reg_max + 1)
        expected_channels = 4 * (reg_max + 1)
        if channels != expected_channels:
            # If channels don't match expected, reshape to match the actual data
            # This handles cases where the mock data doesn't match expected format
            total_elements = batch_size * height * width * channels
            num_anchors = height * width
            if total_elements % (4 * (reg_max + 1)) == 0:
                # Reshape to match expected format
                box_distribute = box_distribute.reshape(batch_size, -1, 4, reg_max + 1)
            else:
                # Skip this scale if it doesn't match expected format
                logger.warning(
                    f"Skipping scale with stride {stride} due to shape mismatch"
                )
                continue
        else:
            # Reshape to expected format
            box_distribute = box_distribute.reshape(
                batch_size, height * width, 4, reg_max + 1
            )

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
            [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0]
        )
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)

    return boxes


@CALLBACK_REGISTRY.registryPostProcessor("yolov8mseg", "yolov8nseg")
class YOLOv8SegPostprocessor(BasePostprocessor):
    """
    Postprocessor for YOLOv8 instance segmentation models.

    This class handles the conversion of raw YOLOv8 segmentation outputs into
    structured segmentation results with proper mask generation, NMS, and coordinate scaling.

    Supported output formats:
    - YOLOv8-seg: Detection output + mask prototypes + mask coefficients
    - Combined format: (batch_size, num_detections, 4 + num_classes + num_masks)
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the YOLOv8 segmentation postprocessor.

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

        # Default number of mask prototypes (YOLOv8 typically uses 32)
        self.num_masks = 32

        logger.info(
            f"Initialized YOLOv8SegPostprocessor with {self.config.num_classes} classes"
        )

    def _validate_config(self) -> None:
        """
        Validate postprocessing configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive, got {self.config.num_classes}"
            )

        if not (0.0 <= self.config.seg_conf_threshold <= 1.0):
            raise ValueError(
                f"seg_conf_threshold must be between 0 and 1, got {self.config.seg_conf_threshold}"
            )

        if not (0.0 <= self.config.nms_iou_threshold <= 1.0):
            raise ValueError(
                f"nms_iou_threshold must be between 0 and 1, got {self.config.nms_iou_threshold}"
            )

        if self.config.seg_max_instances <= 0:
            raise ValueError(
                f"seg_max_instances must be positive, got {self.config.seg_max_instances}"
            )

        if len(self.config.input_shape) != 2:
            raise ValueError(
                f"input_shape must have 2 dimensions, got {len(self.config.input_shape)}"
            )

    NODE_LOCAL_DICT = {
        (20, 64): 0,
        (20, 80): 1,
        (20, 32): 2,
        (40, 64): 3,
        (40, 80): 4,
        (40, 32): 5,
        (80, 64): 6,
        (80, 80): 7,
        (80, 32): 8,
        (160, 32): 9,
    }

    def postprocess(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> SegmentationResult:
        """
        Postprocess raw YOLOv8 segmentation outputs using the reference implementation.

        Args:
            raw_outputs: Dictionary containing raw model outputs or list of endnodes
            original_shape: Original image shape (height, width) for coordinate scaling
            **kwargs: Additional keyword arguments including model configuration

        Returns:
            SegmentationResult containing processed segmentation masks and detections

        Raises:
            ValueError: If output format is not supported
            RuntimeError: If postprocessing fails
        """
        # try:
        # Handle both dictionary and list inputs
        if isinstance(raw_outputs, dict):
            # Convert dict to list format expected by reference implementation
            endnodes_tmp = list(raw_outputs.values())
        else:
            endnodes_tmp = raw_outputs

        endnodes = [0 for _ in range(len(endnodes_tmp))]
        for node in endnodes_tmp:
            endnodes[self.NODE_LOCAL_DICT[node.shape[-2:]]] = node

        # Use reference implementation parameters
        num_classes = kwargs.get("classes", self.config.num_classes)
        strides = kwargs.get("anchors", {}).get("strides", [8, 16, 32])[::-1]
        image_dims = tuple(kwargs.get("img_dims", self.config.input_shape))
        reg_max = kwargs.get("anchors", {}).get("regression_length", 15)
        score_thres = kwargs.get("score_threshold", self.config.seg_conf_threshold)
        iou_thres = kwargs.get("nms_iou_thresh", self.config.nms_iou_threshold)

        # Extract components following reference implementation
        raw_boxes = endnodes[:7:3]  # bbox outputs at indices 0, 3, 6
        scores = [
            np.reshape(s, (-1, s.shape[1] * s.shape[2], num_classes))
            for s in endnodes[1:8:3]
        ]
        scores = np.concatenate(scores, axis=1)
        proto_data = endnodes[9]  # mask prototypes
        batch_size, _, _, n_masks = proto_data.shape

        # Decode boxes
        decoded_boxes = _yolov8_decoding(raw_boxes, strides, image_dims, reg_max)

        # Add objectness=1 for working with NMS
        fake_objectness = np.ones((scores.shape[0], scores.shape[1], 1))
        scores_obj = np.concatenate([fake_objectness, scores], axis=-1)

        # Extract mask coefficients
        coeffs = [
            np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks))
            for c in endnodes[2:9:3]
        ]
        coeffs = np.concatenate(coeffs, axis=1)

        # Re-arrange predictions for NMS
        predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)

        # Apply NMS
        nms_res = non_max_suppression(
            predictions,
            conf_thres=score_thres,
            iou_thres=iou_thres,
            multi_label=True,
            nm=n_masks,
        )

        # Process results for each batch
        outputs = []
        for b in range(batch_size):
            protos = proto_data[b]
            nms_result = nms_res[b]

            # Generate masks from prototypes and coefficients
            masks = process_mask(
                protos,
                nms_result["mask"],
                nms_result["detection_boxes"],
                image_dims,
                upsample=True,
            )

            # Normalize bounding boxes
            boxes = np.array(nms_result["detection_boxes"]) / np.tile(image_dims, 2)

            # Scale to original image size if provided
            # if original_shape is not None:
            #     scale_h = original_shape[0] / image_dims[0]
            #     scale_w = original_shape[1] / image_dims[1]
            #     boxes[:, [0, 2]] *= scale_w
            #     boxes[:, [1, 3]] *= scale_h

            #     if masks is not None:
            #         masks = self._scale_masks(masks, original_shape)

            # Create result
            if masks is not None:
                result = SegmentationResult(
                    masks=masks,
                    scores=np.array(nms_result["detection_scores"]),
                    class_ids=np.array(nms_result["detection_classes"]).astype(int),
                    boxes=boxes,
                )
            else:
                result = self._create_empty_result()

            outputs.append(result)

        # Return first result for single batch, or list for multiple batches
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

        # except Exception as e:
        #     logger.error(f"Error in segmentation postprocessing: {str(e)}")
        #     raise RuntimeError(f"Segmentation postprocessing failed: {str(e)}") from e

    def _extract_outputs(
        self, raw_outputs: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract detection and mask prototype outputs from raw model outputs.

        Args:
            raw_outputs: Dictionary containing raw model outputs

        Returns:
            Tuple of (detection_output, mask_prototypes)

        Raises:
            ValueError: If required outputs are not found
        """
        # Try to find detection and prototype outputs
        detection_output = None
        mask_prototypes = None

        # Look for common output names
        for key, value in raw_outputs.items():
            if "detection" in key.lower() or "output" in key.lower():
                if detection_output is None:
                    detection_output = value
            elif "proto" in key.lower() or "mask" in key.lower():
                if mask_prototypes is None:
                    mask_prototypes = value

        # If not found by name, use shape heuristics
        if detection_output is None or mask_prototypes is None:
            outputs_by_size = sorted(
                raw_outputs.items(), key=lambda x: x[1].size, reverse=True
            )

            if detection_output is None:
                detection_output = outputs_by_size[0][1]
            if mask_prototypes is None:
                # Look for output with 3D spatial dimensions (likely prototypes)
                for key, value in outputs_by_size:
                    if len(value.shape) >= 3 and value.shape[-1] == self.num_masks:
                        mask_prototypes = value
                        break

                if mask_prototypes is None:
                    mask_prototypes = outputs_by_size[1][1]

        if detection_output is None:
            raise ValueError("Could not find detection output in raw_outputs")
        if mask_prototypes is None:
            raise ValueError("Could not find mask prototype output in raw_outputs")

        logger.debug(f"Detection output shape: {detection_output.shape}")
        logger.debug(f"Mask prototypes shape: {mask_prototypes.shape}")

        return detection_output, mask_prototypes

    def _parse_detections(
        self, detection_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse detection output into boxes, scores, class IDs, and mask coefficients.

        Args:
            detection_output: Raw detection output

        Returns:
            Tuple of (boxes, scores, class_ids, mask_coefficients)
        """
        # YOLOv8 output format: [x, y, w, h, class_scores..., mask_coeffs...]
        num_detections = detection_output.shape[0]

        # Extract components
        boxes = detection_output[:, :4]  # x, y, w, h
        class_scores = detection_output[:, 4 : 4 + self.config.num_classes]
        mask_coeffs = detection_output[:, 4 + self.config.num_classes :]

        # Convert boxes from xywh to xyxy
        boxes = self._xywh_to_xyxy(boxes)

        # Get best class and score for each detection
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        logger.debug(f"Parsed {num_detections} detections")
        logger.debug(f"Boxes shape: {boxes.shape}")
        logger.debug(f"Scores shape: {scores.shape}")
        logger.debug(f"Class IDs shape: {class_ids.shape}")
        logger.debug(f"Mask coefficients shape: {mask_coeffs.shape}")

        return boxes, scores, class_ids, mask_coeffs

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
        # Apply NMS per class for segmentation
        keep_indices = []

        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            if len(class_boxes) == 0:
                continue

            # Prepare detections for NMS
            dets = np.hstack([class_boxes, class_scores.reshape(-1, 1)])
            class_keep = _nms(dets, self.config.nms_iou_threshold)

            # Convert back to original indices
            original_indices = np.where(class_mask)[0]
            keep_indices.extend(original_indices[class_keep])

        return np.array(keep_indices, dtype=np.int32)

    def _generate_masks(
        self, prototypes: np.ndarray, coefficients: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:
        """
        Generate segmentation masks from prototypes and coefficients.

        Args:
            prototypes: Mask prototypes with shape (H, W, num_masks)
            coefficients: Mask coefficients with shape (num_detections, num_masks)
            boxes: Bounding boxes with shape (num_detections, 4)

        Returns:
            Generated masks with shape (num_detections, H, W)
        """
        if len(coefficients) == 0:
            return np.empty(
                (0, prototypes.shape[0], prototypes.shape[1]), dtype=np.float32
            )

        # Matrix multiplication to generate masks
        # prototypes: (H, W, C) -> (H*W, C)
        # coefficients: (N, C)
        # result: (N, H*W) -> (N, H, W)

        H, W, C = prototypes.shape
        N = coefficients.shape[0]

        # Reshape prototypes for matrix multiplication
        proto_flat = prototypes.reshape(-1, C)  # (H*W, C)

        # Generate masks: (N, C) @ (C, H*W) -> (N, H*W)
        masks_flat = coefficients @ proto_flat.T  # (N, H*W)

        # Apply sigmoid activation
        masks_flat = self._sigmoid(masks_flat)

        # Reshape to spatial dimensions
        masks = masks_flat.reshape(N, H, W)  # (N, H, W)

        # Crop masks to bounding boxes
        masks = self._crop_masks(masks, boxes)

        logger.debug(f"Generated {N} masks with shape {masks.shape}")

        return masks

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid activation function with numerical stability.

        Args:
            x: Input array

        Returns:
            Sigmoid activated array
        """
        # Clip input to prevent overflow
        x_clipped = np.clip(x, -250, 250)
        return 1 / (1 + np.exp(-x_clipped))

    def _resize_masks(
        self, masks: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize masks to target shape.

        Args:
            masks: Input masks with shape (num_masks, H, W)
            target_shape: Target shape (height, width)

        Returns:
            Resized masks
        """
        if masks.size == 0:
            return np.empty((0, target_shape[0], target_shape[1]), dtype=np.float32)

        num_masks = masks.shape[0]
        resized_masks = np.zeros(
            (num_masks, target_shape[0], target_shape[1]), dtype=np.float32
        )

        for i in range(num_masks):
            resized_masks[i] = cv2.resize(
                masks[i],
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return resized_masks

    def _crop_masks(self, masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Crop masks to bounding box regions.

        Args:
            masks: Masks with shape (num_masks, H, W)
            boxes: Bounding boxes with shape (num_masks, 4) in xyxy format

        Returns:
            Cropped masks
        """
        if masks.size == 0 or boxes.size == 0:
            return masks

        num_masks, H, W = masks.shape

        # Convert boxes to integer coordinates
        boxes_int = np.round(boxes).astype(int)

        # Clip boxes to image boundaries
        boxes_int[:, 0] = np.clip(boxes_int[:, 0], 0, W - 1)  # x1
        boxes_int[:, 1] = np.clip(boxes_int[:, 1], 0, H - 1)  # y1
        boxes_int[:, 2] = np.clip(boxes_int[:, 2], 0, W - 1)  # x2
        boxes_int[:, 3] = np.clip(boxes_int[:, 3], 0, H - 1)  # y2

        # Crop each mask
        for i in range(num_masks):
            x1, y1, x2, y2 = boxes_int[i]

            # Zero out regions outside the bounding box
            masks[i, :y1, :] = 0  # Above box
            masks[i, y2 + 1 :, :] = 0  # Below box
            masks[i, :, :x1] = 0  # Left of box
            masks[i, :, x2 + 1 :] = 0  # Right of box

        return masks

    def _scale_masks(
        self, masks: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scale masks to original image size.

        Args:
            masks: Masks with shape (num_masks, H, W)
            original_shape: Original image shape (height, width)

        Returns:
            Scaled masks
        """
        if masks.size == 0:
            return np.empty((0, original_shape[0], original_shape[1]), dtype=np.float32)

        return self._resize_masks(masks, original_shape)

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

    def _create_empty_result(self) -> SegmentationResult:
        """
        Create an empty segmentation result.

        Returns:
            Empty SegmentationResult
        """
        return SegmentationResult(
            masks=np.empty(
                (0, self.config.input_shape[0], self.config.input_shape[1]),
                dtype=np.float32,
            ),
            scores=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int32),
            boxes=np.empty((0, 4), dtype=np.float32),
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
        self, result: SegmentationResult, class_ids: List[int]
    ) -> SegmentationResult:
        """
        Filter segmentation results by specific class IDs.

        Args:
            result: Input segmentation result
            class_ids: List of class IDs to keep

        Returns:
            Filtered segmentation result
        """
        if len(result) == 0:
            return result

        # Create mask for desired classes
        class_mask = np.isin(result.class_ids, class_ids)

        if not np.any(class_mask):
            return self._create_empty_result()

        # Filter all components
        return SegmentationResult(
            masks=result.masks[class_mask],
            scores=result.scores[class_mask],
            class_ids=result.class_ids[class_mask],
            boxes=result.boxes[class_mask] if result.boxes is not None else None,
        )

    def filter_by_confidence(
        self, result: SegmentationResult, min_confidence: float
    ) -> SegmentationResult:
        """
        Filter segmentation results by minimum confidence threshold.

        Args:
            result: Input segmentation result
            min_confidence: Minimum confidence threshold (0.0 to 1.0)

        Returns:
            Filtered segmentation result

        Raises:
            ValueError: If min_confidence is not in valid range
        """
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be between 0 and 1, got {min_confidence}"
            )

        if result.scores is None or len(result.scores) == 0:
            return result

        # Find indices of detections above threshold
        keep_indices = result.scores >= min_confidence

        # Filter all components
        filtered_masks = (
            result.masks[keep_indices] if result.masks is not None else None
        )
        filtered_scores = result.scores[keep_indices]
        filtered_class_ids = (
            result.class_ids[keep_indices] if result.class_ids is not None else None
        )
        filtered_boxes = (
            result.boxes[keep_indices] if result.boxes is not None else None
        )

        return SegmentationResult(
            masks=filtered_masks,
            scores=filtered_scores,
            class_ids=filtered_class_ids,
            boxes=filtered_boxes,
        )


def _make_grid(
    anchors: np.ndarray, stride: int, bs: int = 8, nx: int = 20, ny: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create grid and anchor grid for YOLOv5 decoding.

    Args:
        anchors: Anchor sizes
        stride: Current stride
        bs: Batch size
        nx: Grid width
        ny: Grid height

    Returns:
        Tuple of (grid, anchor_grid)
    """
    na = len(anchors) // 2
    y, x = np.arange(ny), np.arange(nx)
    yv, xv = np.meshgrid(y, x, indexing="ij")

    grid = np.stack((xv, yv), 2)
    grid = np.stack([grid for _ in range(na)], 0) - 0.5
    grid = np.stack([grid for _ in range(bs)], 0)

    anchor_grid = np.reshape(anchors * stride, (na, -1))
    anchor_grid = np.stack([anchor_grid for _ in range(ny)], axis=1)
    anchor_grid = np.stack([anchor_grid for _ in range(nx)], axis=2)
    anchor_grid = np.stack([anchor_grid for _ in range(bs)], 0)

    return grid, anchor_grid


def _yolov5_decoding(
    branch_idx: int,
    output: np.ndarray,
    stride_list: List[int],
    anchor_list: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Decode YOLOv5 segmentation output for a single branch.

    Args:
        branch_idx: Branch index (0, 1, 2 for different scales)
        output: Raw output tensor with shape (BS, H, W, channels)
        stride_list: List of strides for each branch
        anchor_list: Anchor configurations for each branch
        num_classes: Number of classes

    Returns:
        Decoded output with shape (BS, num_proposals, channels)
    """
    BS, H, W = output.shape[0:3]
    stride = stride_list[branch_idx]
    anchors = anchor_list[branch_idx] / stride
    num_anchors = len(anchors) // 2

    grid, anchor_grid = _make_grid(anchors, stride, BS, W, H)

    # Reshape output: (BS, H, W, channels) -> (BS, num_anchors, -1, H, W) -> (BS, num_anchors, H, W, -1)
    output = (
        output.transpose((0, 3, 1, 2))
        .reshape((BS, num_anchors, -1, H, W))
        .transpose((0, 1, 3, 4, 2))
    )

    # Split into components: xy, wh, conf, mask
    xy, wh, conf, mask = np.array_split(output, [2, 4, 4 + num_classes + 1], axis=4)

    # Decode coordinates and dimensions
    xy = (_sigmoid(xy) * 2 + grid) * stride
    wh = (_sigmoid(wh) * 2) ** 2 * anchor_grid

    # Concatenate all components
    out = np.concatenate((xy, wh, _sigmoid(conf), mask), 4)
    out = out.reshape((BS, num_anchors * H * W, -1)).astype(np.float32)

    return out


def _normalize_yolov5_seg_bboxes(
    output: Dict[str, np.ndarray], img_dims: Tuple[int, int]
) -> np.ndarray:
    """
    Normalize YOLOv5 segmentation bounding boxes and change format.

    Args:
        output: Detection output dictionary
        img_dims: Image dimensions (height, width)

    Returns:
        Normalized bounding boxes in format [y_min, x_min, y_max, x_max]
    """
    bboxes = output["detection_boxes"]
    bboxes[:, [0, 2]] /= img_dims[1]  # Normalize x coordinates
    bboxes[:, [1, 3]] /= img_dims[0]  # Normalize y coordinates

    return bboxes


def _finalize_detections_yolov5_seg(
    outputs: List[Dict[str, np.ndarray]], protos: np.ndarray, **kwargs
) -> List[Dict[str, np.ndarray]]:
    """
    Finalize YOLOv5 segmentation detections by processing masks.

    Args:
        outputs: List of detection outputs for each batch
        protos: Mask prototypes with shape (BS, H, W, num_masks)
        **kwargs: Additional arguments including img_dims

    Returns:
        Finalized detection outputs with processed masks
    """
    for batch_idx, output in enumerate(outputs):
        shape = kwargs.get("img_dims", None)
        boxes = output["detection_boxes"]
        masks = output["mask"]
        proto = protos[batch_idx]

        # Process masks using prototypes and coefficients
        masks = process_mask(proto, masks, boxes, shape, upsample=True)
        output["mask"] = masks

    return outputs


@CALLBACK_REGISTRY.registryPostProcessor("yolov5seg")
class YOLOv5SegPostprocessor(BasePostprocessor):
    """
    Postprocessor for YOLOv5 instance segmentation models.

    This class handles the conversion of raw YOLOv5 segmentation outputs into
    structured segmentation results with proper mask generation, NMS, and coordinate scaling.

    Expected input format:
    - endnodes[0]: mask prototypes with shape (BS, 160, 160, 32)
    - endnodes[1]: stride 32 output with shape (BS, 20, 20, 351)
    - endnodes[2]: stride 16 output with shape (BS, 40, 40, 351)
    - endnodes[3]: stride 8 output with shape (BS, 80, 80, 351)
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the YOLOv5 segmentation postprocessor.

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

        # Default number of mask prototypes (YOLOv5 typically uses 32)
        self.num_masks = 32

        logger.info(
            f"Initialized YOLOv5SegPostprocessor with {self.config.num_classes} classes"
        )

    def _validate_config(self) -> None:
        """
        Validate postprocessing configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive, got {self.config.num_classes}"
            )

        if not (0.0 <= self.config.seg_conf_threshold <= 1.0):
            raise ValueError(
                f"seg_conf_threshold must be between 0 and 1, got {self.config.seg_conf_threshold}"
            )

        if not (0.0 <= self.config.nms_iou_threshold <= 1.0):
            raise ValueError(
                f"nms_iou_threshold must be between 0 and 1, got {self.config.nms_iou_threshold}"
            )

        if self.config.seg_max_instances <= 0:
            raise ValueError(
                f"seg_max_instances must be positive, got {self.config.seg_max_instances}"
            )

        if len(self.config.input_shape) != 2:
            raise ValueError(
                f"input_shape must have 2 dimensions, got {len(self.config.input_shape)}"
            )

    NODE_LOCAL_DICT = {
        (160, 32): 0,
        (20, 351): 1,
        (40, 351): 2,
        (80, 351): 3,
    }

    def postprocess(
        self,
        raw_outputs: Union[Dict[str, np.ndarray], List[np.ndarray]],
        original_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> SegmentationResult:
        """
        Postprocess YOLOv5 segmentation model outputs.

        Args:
            raw_outputs: Raw model outputs (dict or list of tensors)
            original_shape: Original image shape (height, width)
            **kwargs: Additional arguments including:
                - img_dims: Image dimensions
                - anchors: Anchor configuration
                - classes: Number of classes
                - score_threshold: Confidence threshold
                - nms_iou_thresh: NMS IoU threshold

        Returns:
            Structured segmentation result

        Raises:
            RuntimeError: If postprocessing fails
        """
        try:
            # Handle both dictionary and list inputs
            if isinstance(raw_outputs, dict):
                endnodes_tmp = list(raw_outputs.values())
            else:
                endnodes_tmp = raw_outputs

            endnodes = [0 for _ in range(len(endnodes_tmp))]
            for i, node in enumerate(endnodes_tmp):
                endnodes[self.NODE_LOCAL_DICT[node.shape[-2:]]] = node

            # Validate input format
            if len(endnodes) != 4:
                raise ValueError(f"Expected 4 endnodes, got {len(endnodes)}")

            # Extract parameters
            img_dims = kwargs.get("img_dims", self.config.input_shape)
            num_classes = kwargs.get("classes", self.config.num_classes)
            score_thres = kwargs.get("score_threshold", self.config.seg_conf_threshold)
            iou_thres = kwargs.get("nms_iou_thresh", self.config.nms_iou_threshold)

            # Check for HPP (Hardware Post-Processing) mode
            if kwargs.get("hpp", False):
                return self._organize_hpp_outputs(endnodes, img_dims)

            # Extract components
            protos = endnodes[0]  # Mask prototypes
            detection_outputs = endnodes[1:]  # Detection outputs for different scales

            # Get anchor configuration
            anchors_config = kwargs.get("anchors", {})
            anchor_list = np.array(
                anchors_config.get(
                    "sizes",
                    [
                        [10, 13, 16, 30, 33, 23],
                        [30, 61, 62, 45, 59, 119],
                        [116, 90, 156, 198, 373, 326],
                    ],
                )[::-1]
            )
            stride_list = anchors_config.get("strides", [8, 16, 32])[::-1]

            # Decode outputs for each scale
            decoded_outputs = []
            for branch_idx, output in enumerate(detection_outputs):
                decoded_info = _yolov5_decoding(
                    branch_idx, output, stride_list, anchor_list, num_classes
                )
                decoded_outputs.append(decoded_info)

            # Concatenate all scale outputs
            predictions = np.concatenate(
                decoded_outputs, 1
            )  # (BS, num_proposals, channels)

            # Apply NMS
            nms_results = non_max_suppression(
                predictions,
                conf_thres=score_thres,
                iou_thres=iou_thres,
                nm=protos.shape[-1],
            )

            # Finalize detections with mask processing
            nms_results = _finalize_detections_yolov5_seg(
                nms_results, protos, img_dims=img_dims, **kwargs
            )
            # Process results for each batch (assuming batch_size=1 for now)
            if len(nms_results) > 0:
                result = nms_results[0]  # Take first batch

                # Normalize bounding boxes
                if result["detection_boxes"].shape[0] > 0:
                    normalized_boxes = _normalize_yolov5_seg_bboxes(result, img_dims)
                else:
                    normalized_boxes = result["detection_boxes"]

                return SegmentationResult(
                    masks=result["mask"],
                    scores=result["detection_scores"],
                    class_ids=result["detection_classes"].astype(int),
                    boxes=normalized_boxes,
                )
            else:
                return self._create_empty_result()

        except Exception as e:
            logger.error(f"Error in YOLOv5 segmentation postprocessing: {str(e)}")
            raise RuntimeError(
                f"YOLOv5 segmentation postprocessing failed: {str(e)}"
            ) from e

    def _organize_hpp_outputs(
        self, outputs: List[np.ndarray], img_dims: Tuple[int, int]
    ) -> SegmentationResult:
        """
        Organize HPP (Hardware Post-Processing) outputs.

        Args:
            outputs: Raw HPP outputs
            img_dims: Image dimensions

        Returns:
            Structured segmentation result
        """
        # HPP output structure: [-1, num_of_proposals, 6 + h*w]
        # Format: [x_min, y_min, x_max, y_max, score, class, flattened_mask]
        predictions = []
        batch_size, num_of_proposals = outputs.shape[0], outputs.shape[-1]
        outputs = np.transpose(np.squeeze(outputs, axis=1), [0, 2, 1])

        for i in range(batch_size):
            # Reorder boxes from [x_min, y_min, x_max, y_max] to [y_min, x_min, y_max, x_max]
            boxes = outputs[i, :, :4][:, [1, 0, 3, 2]]
            scores = outputs[i, :, 4]
            classes = outputs[i, :, 5]
            masks = outputs[i, :, 6:].reshape((num_of_proposals, *img_dims))

            predictions.append(
                {
                    "detection_boxes": boxes,
                    "detection_scores": scores,
                    "detection_classes": classes,
                    "mask": masks,
                }
            )

        # Return first batch result
        if len(predictions) > 0:
            result = predictions[0]
            return SegmentationResult(
                masks=result["mask"],
                scores=result["detection_scores"],
                class_ids=result["detection_classes"].astype(int),
                boxes=result["detection_boxes"],
            )
        else:
            return self._create_empty_result()

    def _create_empty_result(self) -> SegmentationResult:
        """
        Create an empty segmentation result.

        Returns:
            Empty segmentation result
        """
        return SegmentationResult(
            masks=np.empty((0, *self.config.input_shape), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            class_ids=np.empty(0, dtype=np.int32),
            boxes=np.empty((0, 4), dtype=np.float32),
        )

    def get_class_name(self, class_id: int) -> str:
        """
        Get class name for given class ID.

        Args:
            class_id: Class identifier

        Returns:
            Class name string
        """
        if self.config.class_names is not None and 0 <= class_id < len(
            self.config.class_names
        ):
            return self.config.class_names[class_id]
        return f"class_{class_id}"

    def filter_by_classes(
        self, result: SegmentationResult, class_ids: List[int]
    ) -> SegmentationResult:
        """
        Filter segmentation results by specific class IDs.

        Args:
            result: Input segmentation result
            class_ids: List of class IDs to keep

        Returns:
            Filtered segmentation result
        """
        if result.class_ids is None or len(result.class_ids) == 0:
            return result

        # Find indices of detections with specified class IDs
        keep_indices = np.isin(result.class_ids, class_ids)

        # Filter all components
        filtered_masks = (
            result.masks[keep_indices] if result.masks is not None else None
        )
        filtered_scores = (
            result.scores[keep_indices] if result.scores is not None else None
        )
        filtered_class_ids = result.class_ids[keep_indices]
        filtered_boxes = (
            result.boxes[keep_indices] if result.boxes is not None else None
        )

        return SegmentationResult(
            masks=filtered_masks,
            scores=filtered_scores,
            class_ids=filtered_class_ids,
            boxes=filtered_boxes,
        )

    def filter_by_confidence(
        self, result: SegmentationResult, min_confidence: float
    ) -> SegmentationResult:
        """
        Filter segmentation results by minimum confidence threshold.

        Args:
            result: Input segmentation result
            min_confidence: Minimum confidence threshold (0.0 to 1.0)

        Returns:
            Filtered segmentation result

        Raises:
            ValueError: If min_confidence is not in valid range
        """
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be between 0 and 1, got {min_confidence}"
            )

        if result.scores is None or len(result.scores) == 0:
            return result

        # Find indices of detections above threshold
        keep_indices = result.scores >= min_confidence

        # Filter all components
        filtered_masks = (
            result.masks[keep_indices] if result.masks is not None else None
        )
        filtered_scores = result.scores[keep_indices]
        filtered_class_ids = (
            result.class_ids[keep_indices] if result.class_ids is not None else None
        )
        filtered_boxes = (
            result.boxes[keep_indices] if result.boxes is not None else None
        )

        return SegmentationResult(
            masks=filtered_masks,
            scores=filtered_scores,
            class_ids=filtered_class_ids,
            boxes=filtered_boxes,
        )
