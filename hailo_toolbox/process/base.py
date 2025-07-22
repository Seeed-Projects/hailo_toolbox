from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field


@dataclass
class PostprocessConfig:
    """
    Configuration class for postprocessing operations.

    This class defines all the parameters needed for various postprocessing tasks
    including detection, segmentation, and keypoint detection.
    """

    # Task type flags
    det: bool = False
    seg: bool = False
    kp: bool = False

    # NMS (Non-Maximum Suppression) configuration
    nms: bool = False
    nms_threshold: float = 0.5
    nms_iou_threshold: float = 0.5
    nms_top_k: int = 100
    nms_max_det: int = 100
    nms_score_threshold: float = 0.5

    # Detection specific configuration
    det_conf_threshold: float = 0.2
    det_max_detections: int = 300
    det_class_agnostic: bool = False

    # Segmentation specific configuration
    seg_conf_threshold: float = 0.25
    seg_mask_threshold: float = 0.5
    seg_max_instances: int = 100

    # Keypoint specific configuration
    kp_conf_threshold: float = 0.25
    kp_visibility_threshold: float = 0.5
    kp_max_persons: int = 100

    # Input/Output shape configuration
    input_shape: Tuple[int, int] = (640, 640)  # (height, width)
    original_shape: Optional[Tuple[int, int]] = None  # Original image shape

    # Model specific configuration
    num_classes: int = 80
    num_keypoints: int = 17  # COCO format

    # Additional metadata
    class_names: Optional[List[str]] = None
    keypoint_names: Optional[List[str]] = None
    keypoint_connections: Optional[List[Tuple[int, int]]] = None

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
        self._set_default_names()

    def _validate_parameters(self):
        """Validate all configuration parameters."""
        # Validate confidence thresholds (should be between 0 and 1)
        if not (0.0 <= self.det_conf_threshold <= 1.0):
            raise ValueError(
                f"det_conf_threshold must be between 0 and 1, got {self.det_conf_threshold}"
            )

        if not (0.0 <= self.seg_conf_threshold <= 1.0):
            raise ValueError(
                f"seg_conf_threshold must be between 0 and 1, got {self.seg_conf_threshold}"
            )

        if not (0.0 <= self.kp_conf_threshold <= 1.0):
            raise ValueError(
                f"kp_conf_threshold must be between 0 and 1, got {self.kp_conf_threshold}"
            )

        if not (0.0 <= self.kp_visibility_threshold <= 1.0):
            raise ValueError(
                f"kp_visibility_threshold must be between 0 and 1, got {self.kp_visibility_threshold}"
            )

        if not (0.0 <= self.seg_mask_threshold <= 1.0):
            raise ValueError(
                f"seg_mask_threshold must be between 0 and 1, got {self.seg_mask_threshold}"
            )

        # Validate NMS parameters
        if not (0.0 <= self.nms_iou_threshold <= 1.0):
            raise ValueError(
                f"nms_iou_threshold must be between 0 and 1, got {self.nms_iou_threshold}"
            )

        if not (0.0 <= self.nms_threshold <= 1.0):
            raise ValueError(
                f"nms_threshold must be between 0 and 1, got {self.nms_threshold}"
            )

        if not (0.0 <= self.nms_score_threshold <= 1.0):
            raise ValueError(
                f"nms_score_threshold must be between 0 and 1, got {self.nms_score_threshold}"
            )

        # Validate positive integers
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

        if self.num_keypoints <= 0:
            raise ValueError(
                f"num_keypoints must be positive, got {self.num_keypoints}"
            )

        if self.det_max_detections <= 0:
            raise ValueError(
                f"det_max_detections must be positive, got {self.det_max_detections}"
            )

        if self.seg_max_instances <= 0:
            raise ValueError(
                f"seg_max_instances must be positive, got {self.seg_max_instances}"
            )

        if self.kp_max_persons <= 0:
            raise ValueError(
                f"kp_max_persons must be positive, got {self.kp_max_persons}"
            )

        if self.nms_top_k <= 0:
            raise ValueError(f"nms_top_k must be positive, got {self.nms_top_k}")

        if self.nms_max_det <= 0:
            raise ValueError(f"nms_max_det must be positive, got {self.nms_max_det}")

        # Validate input shape
        if len(self.input_shape) != 2 or any(dim <= 0 for dim in self.input_shape):
            raise ValueError(
                f"input_shape must be a tuple of two positive integers, got {self.input_shape}"
            )

        # Validate original shape if provided
        if self.original_shape is not None:
            if len(self.original_shape) != 2 or any(
                dim <= 0 for dim in self.original_shape
            ):
                raise ValueError(
                    f"original_shape must be a tuple of two positive integers, got {self.original_shape}"
                )

    def _set_default_names(self):
        """Set default class and keypoint names if not provided."""
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        elif len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Length of class_names ({len(self.class_names)}) must match num_classes ({self.num_classes})"
            )

        if self.keypoint_names is None:
            # Default COCO keypoint names for 17 keypoints
            if self.num_keypoints == 17:
                self.keypoint_names = [
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
            else:
                self.keypoint_names = [
                    f"keypoint_{i}" for i in range(self.num_keypoints)
                ]
        elif len(self.keypoint_names) != self.num_keypoints:
            raise ValueError(
                f"Length of keypoint_names ({len(self.keypoint_names)}) must match num_keypoints ({self.num_keypoints})"
            )


@dataclass
class DetectionResult:
    """
    Result structure for object detection.
    """

    boxes: np.ndarray  # Shape: (N, 4) - [x1, y1, x2, y2]
    scores: np.ndarray  # Shape: (N,) - confidence scores
    class_ids: np.ndarray  # Shape: (N,) - class indices
    masks: Optional[np.ndarray] = None  # Shape: (N, H, W) - instance masks

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass
class SegmentationResult:
    """
    Result structure for segmentation.
    """

    masks: np.ndarray  # Shape: (N, H, W) or (H, W) for semantic segmentation
    scores: Optional[np.ndarray] = None  # Shape: (N,) - confidence scores
    class_ids: Optional[np.ndarray] = None  # Shape: (N,) - class indices
    boxes: Optional[np.ndarray] = None  # Shape: (N, 4) - bounding boxes

    def __len__(self) -> int:
        return len(self.masks) if len(self.masks.shape) == 3 else 1


@dataclass
class KeypointResult:
    """
    Result structure for keypoint detection.
    """

    keypoints: np.ndarray  # Shape: (N, K, 3) - [x, y, visibility]
    scores: np.ndarray  # Shape: (N,) - person confidence scores
    boxes: Optional[np.ndarray] = None  # Shape: (N, 4) - person bounding boxes
    joint_scores: Optional[np.ndarray] = (
        None  # Shape: (N, K, 1) - joint confidence scores
    )

    def __len__(self) -> int:
        return len(self.keypoints)


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors.

    Preprocessors are responsible for preparing input data for model inference.
    """

    @abstractmethod
    def preprocess(self, data: Any) -> Dict[str, np.ndarray]:
        """
        Preprocess input data for model inference.

        Args:
            data: Input data to preprocess

        Returns:
            Dictionary mapping input names to preprocessed arrays
        """
        pass


class BasePostprocessor(ABC):
    """
    Abstract base class for all postprocessors.

    Postprocessors are responsible for converting raw model outputs
    into structured, usable results.
    """

    def __init__(self, config: Optional[PostprocessConfig] = None):
        """
        Initialize the postprocessor with configuration.

        Args:
            config: Postprocessing configuration
        """
        self.config = config or PostprocessConfig()

    @abstractmethod
    def postprocess(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> Any:
        """
        Postprocess raw model outputs into structured results.

        Args:
            raw_outputs: Dictionary of raw model outputs
            original_shape: Original image shape (height, width) for coordinate scaling

        Returns:
            Structured postprocessing results
        """
        pass

    def __call__(
        self,
        raw_outputs: Dict[str, np.ndarray],
        original_shape: Optional[Tuple[int, int]] = None,
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> Any:
        """
        Callable interface for postprocessing.

        Args:
            raw_outputs: Dictionary of raw model outputs
            original_shape: Original image shape for coordinate scaling

        Returns:
            Structured postprocessing results
        """
        return self.postprocess(raw_outputs, original_shape, input_shape)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    max_detections: int = 100,
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.

    Args:
        boxes: Array of bounding boxes with shape (N, 4) in format [x1, y1, x2, y2]
        scores: Array of confidence scores with shape (N,)
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum score threshold
        max_detections: Maximum number of detections to keep

    Returns:
        Array of indices of boxes to keep
    """
    # Filter by score threshold
    valid_indices = scores >= score_threshold
    if not np.any(valid_indices):
        return np.array([], dtype=np.int32)

    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    original_indices = np.where(valid_indices)[0]

    # Sort by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    # Calculate areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    keep = []
    while len(sorted_indices) > 0 and len(keep) < max_detections:
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

    return original_indices[keep]


def scale_boxes(
    boxes: np.ndarray, original_shape: Tuple[int, int], input_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Scale bounding boxes from input image size to original image size.

    Args:
        boxes: Bounding boxes with shape (N, 4) in format [x1, y1, x2, y2]
        original_shape: Original image shape (height, width)
        input_shape: Input image shape (height, width)

    Returns:
        Scaled bounding boxes
    """
    if boxes.size == 0:
        return boxes

    # Calculate scaling factors
    scale_y = original_shape[0]  # / input_shape[0]
    scale_x = original_shape[1]  # / input_shape[1]

    # Scale coordinates
    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x  # x coordinates
    scaled_boxes[:, [1, 3]] *= scale_y  # y coordinates

    return scaled_boxes


def scale_keypoints(
    keypoints: np.ndarray, original_shape: Tuple[int, int], input_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Scale keypoints from input image size to original image size.

    Args:
        keypoints: Keypoints with shape (N, K, 3) where last dim is [x, y, visibility]
        original_shape: Original image shape (height, width)
        input_shape: Input image shape (height, width)

    Returns:
        Scaled keypoints
    """
    if keypoints.size == 0:
        return keypoints

    # Calculate scaling factors
    scale_y = original_shape[0] / input_shape[0]
    scale_x = original_shape[1] / input_shape[1]

    # Scale coordinates
    scaled_keypoints = keypoints.copy()
    scaled_keypoints[:, :, 0] *= scale_x  # x coordinates
    scaled_keypoints[:, :, 1] *= scale_y  # y coordinates
    # Keep visibility unchanged

    return scaled_keypoints
