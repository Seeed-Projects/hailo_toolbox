"""
Advanced Visualization Module for Deep Learning Model Results

This module provides comprehensive visualization capabilities for different types of
deep learning model outputs including object detection, instance segmentation,
and keypoint detection results.

Features:
- Multi-task visualization support (detection, segmentation, keypoints)
- Customizable visual styles and colors
- High-quality rendering with anti-aliasing
- Flexible configuration system
- Support for class names and confidence scores
- Keypoint skeleton visualization
- Mask overlay with transparency
- Performance optimized drawing operations
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
import colorsys
import random
import time
import gc  # Import garbage collector for memory management
from abc import ABC, abstractmethod

from ..base import DetectionResult, SegmentationResult, KeypointResult
from hailo_toolbox.inference.core import CALLBACK_REGISTRY


logger = logging.getLogger(__name__)


def _generate_color(class_id: int) -> Tuple[int, int, int]:
    """
    Generate a consistent color for a given class ID.

    Args:
        class_id: Class identifier

    Returns:
        BGR color tuple
    """
    # Use a simple hash-based color generation
    np.random.seed(class_id * 3)  # Multiply by 3 for better color distribution
    color = tuple(np.random.randint(0, 256, 3).tolist())
    return color


@dataclass
class VisualizationConfig:
    """
    Configuration class for visualization settings.

    This class defines all visual parameters for rendering detection, segmentation,
    and keypoint results with customizable styles and colors.
    """

    # Task-specific enable flags
    det: bool = False
    seg: bool = False
    kp: bool = False

    # Detection visualization settings
    draw_bbox: bool = True
    draw_text: bool = True
    draw_score: bool = True
    draw_class: bool = True

    # Segmentation visualization settings
    draw_mask: bool = True
    mask_alpha: float = 0.5
    mask_color_mode: str = "random"  # "random", "class", "fixed"

    # Keypoint visualization settings
    draw_keypoints: bool = True
    draw_skeleton: bool = True
    draw_person_bbox: bool = True
    keypoint_radius: int = 4
    skeleton_thickness: int = 2

    # Text and font settings
    font_scale: float = 0.6
    font_thickness: int = 2
    font_color: Tuple[int, int, int] = (255, 255, 255)
    text_background: bool = True
    text_background_color: Tuple[int, int, int] = (0, 0, 0)
    text_background_alpha: float = 0.7

    # Box and line settings
    bbox_thickness: int = 2
    bbox_color_mode: str = "class"  # "class", "random", "fixed"
    fixed_bbox_color: Tuple[int, int, int] = (0, 255, 0)

    # Color palette settings
    color_palette: str = "default"  # "default", "pastel", "bright", "dark"
    random_seed: Optional[int] = 42

    # Display settings
    show_confidence_threshold: float = 0.1
    max_labels_per_image: int = 100
    label_offset: Tuple[int, int] = (5, -5)

    # Output settings
    output_format: str = "bgr"  # "bgr", "rgb"
    anti_aliasing: bool = True

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_parameters()
        self._setup_color_palette()

    def _validate_parameters(self):
        """Validate all configuration parameters."""
        if not (0.0 <= self.mask_alpha <= 1.0):
            raise ValueError(
                f"mask_alpha must be between 0 and 1, got {self.mask_alpha}"
            )

        if not (0.0 <= self.text_background_alpha <= 1.0):
            raise ValueError(
                f"text_background_alpha must be between 0 and 1, got {self.text_background_alpha}"
            )

        if not (0.0 <= self.show_confidence_threshold <= 1.0):
            raise ValueError(
                f"show_confidence_threshold must be between 0 and 1, got {self.show_confidence_threshold}"
            )

        if self.font_scale <= 0:
            raise ValueError(f"font_scale must be positive, got {self.font_scale}")

        if self.font_thickness <= 0:
            raise ValueError(
                f"font_thickness must be positive, got {self.font_thickness}"
            )

        if self.bbox_thickness <= 0:
            raise ValueError(
                f"bbox_thickness must be positive, got {self.bbox_thickness}"
            )

        if self.keypoint_radius <= 0:
            raise ValueError(
                f"keypoint_radius must be positive, got {self.keypoint_radius}"
            )

        if self.skeleton_thickness <= 0:
            raise ValueError(
                f"skeleton_thickness must be positive, got {self.skeleton_thickness}"
            )

        if self.max_labels_per_image <= 0:
            raise ValueError(
                f"max_labels_per_image must be positive, got {self.max_labels_per_image}"
            )

    def _setup_color_palette(self):
        """Setup color palette based on configuration."""
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Generate color palette based on selected style
        if self.color_palette == "pastel":
            self.colors = self._generate_pastel_colors(100)
        elif self.color_palette == "bright":
            self.colors = self._generate_bright_colors(100)
        elif self.color_palette == "dark":
            self.colors = self._generate_dark_colors(100)
        else:  # default
            self.colors = self._generate_default_colors(100)

    def _generate_default_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate default color palette."""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors

    def _generate_pastel_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate pastel color palette."""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.3, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors

    def _generate_bright_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate bright color palette."""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors

    def _generate_dark_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate dark color palette."""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.6)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors


class BaseVisualization(ABC):
    """
    Abstract base class for all visualization implementations.

    This class provides the common interface and utility methods that all
    visualization classes should implement. Each specific task visualization
    should inherit from this base class.

    Features:
    - Common visualization utilities and helper methods
    - Standardized color management
    - Text rendering with background support
    - Image saving and display functionality
    - Configuration validation
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the base visualization system.

        Args:
            config: Visualization configuration. If None, default config is used.

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config or VisualizationConfig()

        # Setup class names for common datasets
        self.coco_class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        logger.info(f"Initialized {self.__class__.__name__} visualization system")
        self.prev_time = time.time()

    def __call__(
        self,
        image: np.ndarray,
        results: Union[DetectionResult, SegmentationResult, KeypointResult],
    ) -> np.ndarray:
        """
        Main visualization entry point.

        Args:
            image: Input image as numpy array
            results: Model results to visualize

        Returns:
            Visualized image as numpy array
        """
        return self.visualize(image, results)

    @abstractmethod
    def visualize(
        self,
        image: np.ndarray,
        results: Union[DetectionResult, SegmentationResult, KeypointResult],
    ) -> np.ndarray:
        """
        Core visualization method that each subclass must implement.

        Args:
            image: Input image as numpy array (H, W, C)
            results: Model results to visualize

        Returns:
            Visualized image as numpy array
        """
        pass

    def _validate_image(self, image: np.ndarray) -> None:
        """
        Validate input image format.

        Args:
            image: Input image to validate

        Raises:
            TypeError: If image is not numpy array
            ValueError: If image format is not supported
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(image)}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

    def _get_color(self, class_id: int, instance_id: int) -> Tuple[int, int, int]:
        """
        Get color for visualization based on class ID or instance ID.

        Args:
            class_id: Class ID
            instance_id: Instance ID

        Returns:
            BGR color tuple
        """
        if self.config.bbox_color_mode == "fixed":
            return self.config.fixed_bbox_color
        elif self.config.bbox_color_mode == "random":
            return self.config.colors[instance_id % len(self.config.colors)]
        else:  # class-based coloring
            return self.config.colors[class_id % len(self.config.colors)]

    def _draw_fps(self, image: np.ndarray, fps: float):
        """
        Draw FPS on the image.
        """
        font_scale = 0.8
        cv2.putText(
            image,
            f"FPS: {fps:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            2,
        )
        self.prev_time = time.time()
        return image

    def _get_class_name(self, class_id: int) -> str:
        """
        Get class name for a given class ID.

        Args:
            class_id: Class ID

        Returns:
            Class name string
        """
        if class_id < len(self.coco_class_names):
            return self.coco_class_names[class_id]
        else:
            return f"class_{class_id}"

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """
        Draw text label with background.

        Args:
            image: Image to draw on
            text: Text to draw
            position: Position (x, y) for text
            color: Text color
        """
        x, y = position
        x += self.config.label_offset[0]
        y += self.config.label_offset[1]

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.font_thickness,
        )

        # Draw background rectangle if enabled
        if self.config.text_background:
            bg_color = self.config.text_background_color

            # Create background rectangle
            bg_x1 = x - 2
            bg_y1 = y - text_height - baseline - 2
            bg_x2 = x + text_width + 2
            bg_y2 = y + baseline + 2

            # Draw semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            image[:] = cv2.addWeighted(
                image,
                1 - self.config.text_background_alpha,
                overlay,
                self.config.text_background_alpha,
                0,
            )

        # Draw text
        if self.config.anti_aliasing:
            cv2.putText(
                image,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.font_color,
                self.config.font_thickness,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.font_color,
                self.config.font_thickness,
            )

    def save(self, image: np.ndarray, path: str) -> bool:
        """
        Save visualized image to file.

        Args:
            image: Image to save
            path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            success = cv2.imwrite(path, image)
            if success:
                logger.info(f"Saved visualization to {path}")
            else:
                logger.error(f"Failed to save visualization to {path}")
            return success
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            return False

    def show(
        self,
        image: np.ndarray,
        window_name: str = "Visualization",
    ) -> None:
        """
        Display image in a window.

        Args:
            image: Image to display
            window_name: Window name
            wait_key: Whether to wait for key press
        """
        self._draw_fps(image, 1 / (time.time() - self.prev_time))
        try:
            cv2.imshow(window_name, image)
            cv2.waitKey(1)
        except Exception as e:
            logger.error(f"Error displaying image: {str(e)}")

    def create_grid(
        self, images: List[np.ndarray], grid_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Create a grid of images for batch visualization.

        Args:
            images: List of images to arrange in grid
            grid_size: Grid size (rows, cols). If None, automatically determined.

        Returns:
            Grid image

        Raises:
            ValueError: If no images provided
        """
        if not images:
            raise ValueError("No images provided for grid creation")

        num_images = len(images)

        # Determine grid size if not provided
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            grid_size = (rows, cols)

        rows, cols = grid_size

        # Get image dimensions (assume all images have same size)
        h, w = images[0].shape[:2]

        # Create grid image
        grid_image = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            y1, y2 = row * h, (row + 1) * h
            x1, x2 = col * w, (col + 1) * w

            # Resize image if necessary
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))

            grid_image[y1:y2, x1:x2] = img

        return grid_image


@CALLBACK_REGISTRY.registryVisualizer("yolov8det", "yolo11det")
class DetectionVisualization(BaseVisualization):
    """
    Specialized visualization class for object detection results.

    This class handles visualization of bounding boxes, class labels,
    and confidence scores for object detection models.

    Features:
    - Bounding box rendering with customizable styles
    - Class label and confidence score display
    - Multi-class color coding
    - Confidence threshold filtering
    - Performance optimized for detection tasks
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize detection visualization system.

        Args:
            config: Visualization configuration. If None, default config is used.
        """
        super().__init__(config)
        # Enable detection-specific settings
        self.config.det = True

    def visualize(
        self,
        image: np.ndarray,
        results: DetectionResult,
    ) -> np.ndarray:
        """
        Visualize object detection results.

        Args:
            image: Input image as numpy array (H, W, C)
            results: Detection results to visualize

        Returns:
            Visualized image as numpy array

        Raises:
            ValueError: If image format is not supported
            TypeError: If results type is not supported
        """
        # Validate inputs
        self._validate_image(image)

        if not isinstance(results, DetectionResult):
            raise TypeError(f"Expected DetectionResult, got {type(results)}")

        # Create a copy to avoid modifying the original image
        vis_image = image.copy()

        # Visualize detection results
        vis_image = self._visualize_detection(vis_image, results)

        # Convert color format if needed
        if self.config.output_format == "rgb":
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        return vis_image

    def _visualize_detection(
        self, image: np.ndarray, results: DetectionResult
    ) -> np.ndarray:
        """
        Visualize object detection results.

        Args:
            image: Input image
            results: Detection results

        Returns:
            Image with detection visualizations
        """
        if len(results) == 0:
            return image

        # Filter by confidence threshold
        valid_mask = results.scores >= self.config.show_confidence_threshold
        if not np.any(valid_mask):
            return image

        boxes = results.boxes[valid_mask]
        scores = results.scores[valid_mask]
        class_ids = results.class_ids[valid_mask]

        # Limit number of visualizations
        if len(boxes) > self.config.max_labels_per_image:
            # Sort by confidence and keep top detections
            top_indices = np.argsort(scores)[::-1][: self.config.max_labels_per_image]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            class_ids = class_ids[top_indices]

        # Draw bounding boxes and labels
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)

            # Get color for this detection
            color = self._get_color(class_id, i)

            # Draw bounding box
            if self.config.draw_bbox:
                if self.config.anti_aliasing:
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.config.bbox_thickness,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.rectangle(
                        image, (x1, y1), (x2, y2), color, self.config.bbox_thickness
                    )

            # Draw label
            if self.config.draw_text:
                label_parts = []

                if self.config.draw_class:
                    class_name = self._get_class_name(class_id)
                    label_parts.append(class_name)

                if self.config.draw_score:
                    label_parts.append(f"{score:.2f}")

                if label_parts:
                    label = " ".join(label_parts)
                    self._draw_label(image, label, (x1, y1), color)

        return image


@CALLBACK_REGISTRY.registryVisualizer("yolov8seg", "yolov5seg")
class SegmentationVisualization(BaseVisualization):
    """
    Specialized visualization class for instance segmentation results.

    This class handles visualization of segmentation masks, bounding boxes, and class labels.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self.config.seg = True

    def visualize(self, image: np.ndarray, results: SegmentationResult) -> np.ndarray:
        """
        Visualize segmentation results with optimized memory usage.

        This method has been optimized to prevent memory leaks by:
        1. Minimizing array copies and intermediate allocations
        2. Using in-place operations where possible
        3. Explicitly managing memory for large arrays
        4. Adding garbage collection hints for large operations

        Args:
            image: Input image
            results: Segmentation results

        Returns:
            Image with segmentation visualizations
        """
        if len(results) == 0:
            return image

        # Filter by confidence threshold early to reduce processing
        valid_mask = results.scores >= self.config.show_confidence_threshold
        if not np.any(valid_mask):
            return image

        # Extract valid results with minimal copying
        boxes = results.boxes[valid_mask] if results.boxes is not None else None
        scores = results.scores[valid_mask]
        class_ids = (
            results.class_ids[valid_mask] if results.class_ids is not None else None
        )
        masks = results.masks[valid_mask] if results.masks is not None else None

        # Limit number of visualizations to prevent memory overflow
        if len(scores) > self.config.max_labels_per_image:
            # Sort by confidence and keep top detections
            top_indices = np.argsort(scores)[::-1][: self.config.max_labels_per_image]
            boxes = boxes[top_indices] if boxes is not None else None
            scores = scores[top_indices]
            class_ids = class_ids[top_indices] if class_ids is not None else None
            masks = masks[top_indices] if masks is not None else None

        # Work with float32 image to avoid repeated conversions
        img_float = image.astype(np.float32) / 255.0

        # Draw masks with memory-optimized blending
        if masks is not None and self.config.draw_mask:
            num_dets = len(masks)

            # Pre-allocate arrays to avoid repeated allocations
            mask_colors = np.zeros((num_dets, 3), dtype=np.float32)

            # Prepare mask colors efficiently
            for i in range(num_dets):
                class_id = class_ids[i] if class_ids is not None else i
                color = self._get_color(class_id, i)
                mask_colors[i] = np.array(color, dtype=np.float32) / 255.0

            # Process masks with memory optimization
            processed_masks = []
            target_shape = image.shape[:2]

            for i, mask in enumerate(masks):
                # Ensure mask is 2D with minimal operations
                if mask.ndim == 3:
                    mask = (
                        np.squeeze(mask, axis=-1)
                        if mask.shape[-1] == 1
                        else mask[:, :, 0]
                    )

                # Resize mask only if necessary
                if mask.shape != target_shape:
                    # Use more memory-efficient resize
                    mask_resized = cv2.resize(
                        mask.astype(np.float32),
                        (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_NEAREST,  # Faster for binary masks
                    )
                    binary_mask = (mask_resized > 0.5).astype(
                        np.float32
                    )  # Use 127 threshold for uint8
                else:
                    binary_mask = (mask > 0.5).astype(np.float32)

                processed_masks.append(binary_mask)

            # Apply optimized mask blending
            if processed_masks:
                # Process masks in smaller batches to reduce memory usage
                batch_size = min(8, num_dets)  # Process max 8 masks at once

                for batch_start in range(0, num_dets, batch_size):
                    batch_end = min(batch_start + batch_size, num_dets)
                    batch_masks = processed_masks[batch_start:batch_end]
                    batch_colors = mask_colors[batch_start:batch_end]

                    if not batch_masks:
                        continue

                    # Stack batch masks efficiently
                    masks_batch = np.stack(batch_masks, axis=0)  # (B, H, W)

                    # Expand dimensions for broadcasting
                    masks_expanded = masks_batch[:, :, :, np.newaxis]  # (B, H, W, 1)
                    colors_expanded = batch_colors[
                        :, np.newaxis, np.newaxis, :
                    ]  # (B, 1, 1, 3)

                    # Create colored masks with alpha
                    colored_masks = (
                        masks_expanded * colors_expanded * self.config.mask_alpha
                    )

                    # Apply masks with proper alpha compositing
                    for j in range(len(batch_masks)):
                        mask_alpha = masks_expanded[j] * self.config.mask_alpha
                        inv_alpha = 1.0 - mask_alpha

                        # In-place blending to reduce memory allocation
                        img_float *= inv_alpha
                        img_float += colored_masks[j]

                    # Clean up batch arrays
                    del masks_batch, masks_expanded, colors_expanded, colored_masks

                # Force garbage collection after processing all masks
                gc.collect()

                # Draw mask contours with optimized approach
                for i, mask in enumerate(processed_masks):
                    if not np.any(mask):  # Skip empty masks
                        continue

                    class_id = class_ids[i] if class_ids is not None else i
                    color = self._get_color(class_id, i)

                    # Find contours on binary mask
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Draw contours directly on float image (converted to uint8 temporarily)
                    if contours:
                        img_uint8_temp = (img_float * 255).astype(np.uint8)
                        cv2.drawContours(
                            img_uint8_temp,
                            contours,
                            -1,
                            color,
                            thickness=1,
                            lineType=(
                                cv2.LINE_AA if self.config.anti_aliasing else cv2.LINE_8
                            ),
                        )
                        img_float = img_uint8_temp.astype(np.float32) / 255.0

                        # Clean up temporary array
                        del img_uint8_temp

        # Convert back to uint8 for drawing operations
        image_result = (img_float * 255).astype(np.uint8)

        # Clean up float image
        del img_float

        # Draw bounding boxes and labels with enhanced styling
        if boxes is not None and self.config.draw_bbox:
            for i, box in enumerate(boxes):
                # Scale to image dimensions (assuming normalized coordinates)
                # box = box * image.shape[1]  # TODO: Make this configurable
                box[::2] *= image.shape[1]
                box[1::2] *= image.shape[0]
                x1, y1, x2, y2 = box.astype(int)

                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

                class_id = class_ids[i] if class_ids is not None else i
                score = scores[i] if scores is not None else 1.0
                color = self._get_color(class_id, i)

                # Draw bounding box
                if self.config.anti_aliasing:
                    cv2.rectangle(
                        image_result,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.config.bbox_thickness,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.rectangle(
                        image_result,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.config.bbox_thickness,
                    )

                # Draw enhanced label with background
                if self.config.draw_text:
                    label_parts = []

                    if self.config.draw_class and class_ids is not None:
                        class_name = self._get_class_name(class_id)
                        label_parts.append(class_name)

                    if self.config.draw_score and scores is not None:
                        label_parts.append(f"{score:.2f}")

                    if label_parts:
                        label = " ".join(label_parts)
                        self._draw_enhanced_label(image_result, label, (x1, y1), color)

        # Final garbage collection to ensure cleanup
        gc.collect()

        return image_result

    def _draw_enhanced_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ):
        """
        Draw enhanced label with background and improved styling.
        Based on the reference implementation approach.

        Args:
            image: Image to draw on
            text: Text to draw
            position: Position (x, y) for the label
            color: Text color
        """
        x, y = position
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        font_thickness = self.config.font_thickness

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font_face, font_scale, font_thickness
        )

        # Calculate label background rectangle
        padding = 4
        bg_x1 = x
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline

        # Ensure background rectangle is within image bounds
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(image.shape[1], bg_x2)
        bg_y2 = min(image.shape[0], bg_y2)

        # Draw background rectangle with same color as bounding box
        if self.config.text_background:
            cv2.rectangle(
                image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1  # Filled rectangle
            )

        # Calculate text position (centered in background)
        text_x = bg_x1 + padding // 2
        text_y = bg_y2 - baseline - padding // 2

        # Ensure text position is within bounds
        text_x = max(0, min(text_x, image.shape[1] - text_width))
        text_y = max(text_height, min(text_y, image.shape[0]))

        # Draw text with white color for better contrast
        text_color = (255, 255, 255)  # White text on colored background

        if self.config.anti_aliasing:
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font_face,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font_face,
                font_scale,
                text_color,
                font_thickness,
            )


@CALLBACK_REGISTRY.registryVisualizer("yolov8pe", "openpose")
class KeypointVisualization(BaseVisualization):
    """
    Specialized visualization class for keypoint detection results.

    This class handles visualization of keypoints, skeletons, and person bounding boxes.

    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self.config.kp = True

    def visualize(
        self,
        image: np.ndarray,
        results: Union[Dict[str, np.ndarray], KeypointResult],
        detection_threshold: float = 0.2,
        joint_threshold: float = 0.2,
        **kwargs,
    ) -> np.ndarray:
        """
        Visualize YOLOv8 pose estimation results.

        Args:
            image: Input image as numpy array
            results: Pose estimation results (either dict format or KeypointResult)
            detection_threshold: Minimum confidence threshold for person detection
            joint_threshold: Minimum confidence threshold for joint visibility
            **kwargs: Additional visualization parameters

        Returns:
            Visualized image with pose estimation results
        """
        # Handle different input formats
        if isinstance(results, dict):
            if "predictions" in results:
                # Legacy format from original pose_estimation_postprocessing.py
                predictions = results["predictions"]
                bboxes = predictions.get("bboxes", np.array([]))
                scores = predictions.get("scores", np.array([]))
                keypoints = predictions.get("keypoints", np.array([]))
                joint_scores = predictions.get("joint_scores", np.array([]))
            else:
                # Direct format
                bboxes = results.get("bboxes", np.array([]))
                scores = results.get("scores", np.array([]))
                keypoints = results.get("keypoints", np.array([]))
                joint_scores = results.get("joint_scores", np.array([]))
        elif isinstance(results, KeypointResult):
            bboxes = results.boxes if results.boxes is not None else np.array([])
            scores = results.scores
            keypoints = results.keypoints
            joint_scores = results.joint_scores
        else:
            logger.warning("Unsupported results format for YOLOv8 pose visualization")
            return image

        # Handle batch dimension
        if len(bboxes.shape) == 3:
            bboxes = bboxes[0]
        if len(scores.shape) == 2:
            scores = scores[0]
        if len(keypoints.shape) == 4:
            keypoints = keypoints[0]
        if len(joint_scores.shape) == 4:
            joint_scores = joint_scores[0]

        # Create a copy of the image for visualization
        vis_image = image.copy()

        # COCO keypoint connections for skeleton
        joint_pairs = [
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

        # Visualize each person
        for i, (bbox, score, person_keypoints) in enumerate(
            zip(bboxes, scores[0], keypoints)
        ):
            # Extract scalar score value
            if isinstance(score, np.ndarray):
                if score.ndim > 0:
                    score_val = score.item() if score.size == 1 else score.flat[0]
                else:
                    score_val = score.item()
            else:
                score_val = float(score)
            if score_val < detection_threshold:
                continue

            # Draw person bounding box
            if len(bbox) >= 4:
                x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw person confidence score
                label = f"Person {score_val:.2f}"
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # Determine joint visibility
            if len(joint_scores) > i:
                joint_visible = joint_scores[i].squeeze() > joint_threshold
            else:
                # Fallback: use keypoint visibility from keypoints array
                joint_visible = person_keypoints[:, 2] > joint_threshold

            # Ensure keypoints are in the right format
            if len(person_keypoints.shape) == 2 and person_keypoints.shape[1] >= 2:
                kpts_2d = person_keypoints[:, :2]
            else:
                kpts_2d = (
                    person_keypoints.reshape(-1, 2)
                    if person_keypoints.size > 0
                    else np.array([])
                )

            # Draw skeleton connections
            for joint0, joint1 in joint_pairs:
                if (
                    joint0 < len(joint_visible)
                    and joint1 < len(joint_visible)
                    and joint_visible[joint0]
                    and joint_visible[joint1]
                    and joint0 < len(kpts_2d)
                    and joint1 < len(kpts_2d)
                ):
                    pt1 = (int(kpts_2d[joint0][0]), int(kpts_2d[joint0][1]))
                    pt2 = (int(kpts_2d[joint1][0]), int(kpts_2d[joint1][1]))

                    # Draw skeleton line
                    cv2.line(vis_image, pt1, pt2, (0, 255, 255), 3)

            # Draw individual keypoints
            for j, (x, y) in enumerate(kpts_2d):
                if j < len(joint_visible) and joint_visible[j]:
                    center = (int(x), int(y))

                    # Use different colors for different body parts
                    if j < 5:  # Head keypoints
                        color = (0, 255, 255)  # Yellow
                    elif j < 11:  # Arm keypoints
                        color = (255, 0, 255)  # Magenta
                    else:  # Leg keypoints
                        color = (255, 255, 0)  # Cyan

                    cv2.circle(vis_image, center, 4, color, -1)
                    cv2.circle(vis_image, center, 5, (0, 0, 0), 1)

        return vis_image


class Visualization:
    """
    Advanced visualization class for deep learning model results.

    This class provides comprehensive visualization capabilities for different types
    of model outputs including object detection, instance segmentation, and keypoint detection.

    Features:
    - Multi-task support with automatic result type detection
    - Customizable visual styles and colors
    - High-quality rendering with anti-aliasing
    - Performance optimized drawing operations
    - Support for batch processing
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the visualization system.

        Args:
            config: Visualization configuration. If None, default config is used.
        """
        self.config = config or VisualizationConfig()

        # Setup class names for common datasets
        self.coco_class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        # COCO keypoint connections for skeleton drawing
        self.coco_skeleton = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # Head
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),  # Arms
            (5, 11),
            (6, 12),
            (11, 12),  # Torso
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),  # Legs
        ]

        logger.info("Initialized Visualization system")

    def __call__(
        self,
        image: np.ndarray,
        results: Union[DetectionResult, SegmentationResult, KeypointResult],
    ) -> np.ndarray:
        """
        Visualize results on the input image.

        Args:
            image: Input image as numpy array
            results: Model results (DetectionResult, SegmentationResult, or KeypointResult)

        Returns:
            Visualized image as numpy array
        """
        return self.visualize(image, results)

    def visualize(
        self,
        image: np.ndarray,
        results: Union[DetectionResult, SegmentationResult, KeypointResult],
    ) -> np.ndarray:
        """
        Main visualization method that handles different result types.

        Args:
            image: Input image as numpy array (H, W, C)
            results: Model results to visualize

        Returns:
            Visualized image as numpy array

        Raises:
            ValueError: If image format is not supported
            TypeError: If results type is not supported
        """
        # Validate input image
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(image)}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

        # Create a copy to avoid modifying the original image
        vis_image = image.copy()

        # Determine result type and visualize accordingly
        if isinstance(results, DetectionResult):
            vis_image = self._visualize_detection(vis_image, results)
        elif isinstance(results, SegmentationResult):
            vis_image = self._visualize_segmentation(vis_image, results)
        elif isinstance(results, KeypointResult):
            vis_image = self._visualize_keypoints(vis_image, results)
        else:
            raise TypeError(f"Unsupported result type: {type(results)}")

        # Convert color format if needed
        if self.config.output_format == "rgb":
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        return vis_image

    def _visualize_detection(
        self, image: np.ndarray, results: DetectionResult
    ) -> np.ndarray:
        """
        Visualize object detection results.

        Args:
            image: Input image
            results: Detection results

        Returns:
            Image with detection visualizations
        """
        if len(results) == 0:
            return image

        # Filter by confidence threshold
        valid_mask = results.scores >= self.config.show_confidence_threshold
        if not np.any(valid_mask):
            return image

        boxes = results.boxes[valid_mask]
        scores = results.scores[valid_mask]
        class_ids = results.class_ids[valid_mask]

        # Limit number of visualizations
        if len(boxes) > self.config.max_labels_per_image:
            # Sort by confidence and keep top detections
            top_indices = np.argsort(scores)[::-1][: self.config.max_labels_per_image]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            class_ids = class_ids[top_indices]

        # Draw bounding boxes and labels
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)

            # Get color for this detection
            color = self._get_color(class_id, i)

            # Draw bounding box
            if self.config.draw_bbox:
                if self.config.anti_aliasing:
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.config.bbox_thickness,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.rectangle(
                        image, (x1, y1), (x2, y2), color, self.config.bbox_thickness
                    )

            # Draw label
            if self.config.draw_text:
                label_parts = []

                if self.config.draw_class:
                    class_name = self._get_class_name(class_id)
                    label_parts.append(class_name)

                if self.config.draw_score:
                    label_parts.append(f"{score:.2f}")

                if label_parts:
                    label = " ".join(label_parts)
                    self._draw_label(image, label, (x1, y1), color)

        return image

    def _visualize_segmentation(
        self, image: np.ndarray, results: SegmentationResult
    ) -> np.ndarray:
        """
        Visualize segmentation results with optimized memory usage.

        This method has been optimized to prevent memory leaks by:
        1. Minimizing array copies and intermediate allocations
        2. Using in-place operations where possible
        3. Explicitly managing memory for large arrays
        4. Adding garbage collection hints for large operations

        Args:
            image: Input image
            results: Segmentation results

        Returns:
            Image with segmentation visualizations
        """
        if len(results) == 0:
            return image

        # Filter by confidence threshold early to reduce processing
        valid_mask = results.scores >= self.config.show_confidence_threshold
        if not np.any(valid_mask):
            return image

        # Extract valid results with minimal copying
        boxes = results.boxes[valid_mask] if results.boxes is not None else None
        scores = results.scores[valid_mask]
        class_ids = (
            results.class_ids[valid_mask] if results.class_ids is not None else None
        )
        masks = results.masks[valid_mask] if results.masks is not None else None

        # Limit number of visualizations to prevent memory overflow
        if len(scores) > self.config.max_labels_per_image:
            # Sort by confidence and keep top detections
            top_indices = np.argsort(scores)[::-1][: self.config.max_labels_per_image]
            boxes = boxes[top_indices] if boxes is not None else None
            scores = scores[top_indices]
            class_ids = class_ids[top_indices] if class_ids is not None else None
            masks = masks[top_indices] if masks is not None else None

        # Work with float32 image to avoid repeated conversions
        img_float = image.astype(np.float32) / 255.0

        # Draw masks with memory-optimized blending
        if masks is not None and self.config.draw_mask:
            num_dets = len(masks)

            # Pre-allocate arrays to avoid repeated allocations
            mask_colors = np.zeros((num_dets, 3), dtype=np.float32)

            # Prepare mask colors efficiently
            for i in range(num_dets):
                class_id = class_ids[i] if class_ids is not None else i
                color = self._get_color(class_id, i)
                mask_colors[i] = np.array(color, dtype=np.float32) / 255.0

            # Process masks with memory optimization
            processed_masks = []
            target_shape = image.shape[:2]

            for i, mask in enumerate(masks):
                # Ensure mask is 2D with minimal operations
                if mask.ndim == 3:
                    mask = (
                        np.squeeze(mask, axis=-1)
                        if mask.shape[-1] == 1
                        else mask[:, :, 0]
                    )

                # Resize mask only if necessary
                if mask.shape != target_shape:
                    # Use more memory-efficient resize
                    mask_resized = cv2.resize(
                        mask.astype(np.float32),
                        (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_NEAREST,  # Faster for binary masks
                    )
                    binary_mask = (mask_resized > 0.5).astype(
                        np.float32
                    )  # Use 127 threshold for uint8
                else:
                    binary_mask = (mask > 0.5).astype(np.float32)

                processed_masks.append(binary_mask)

            # Apply optimized mask blending
            if processed_masks:
                # Process masks in smaller batches to reduce memory usage
                batch_size = min(8, num_dets)  # Process max 8 masks at once

                for batch_start in range(0, num_dets, batch_size):
                    batch_end = min(batch_start + batch_size, num_dets)
                    batch_masks = processed_masks[batch_start:batch_end]
                    batch_colors = mask_colors[batch_start:batch_end]

                    if not batch_masks:
                        continue

                    # Stack batch masks efficiently
                    masks_batch = np.stack(batch_masks, axis=0)  # (B, H, W)

                    # Expand dimensions for broadcasting
                    masks_expanded = masks_batch[:, :, :, np.newaxis]  # (B, H, W, 1)
                    colors_expanded = batch_colors[
                        :, np.newaxis, np.newaxis, :
                    ]  # (B, 1, 1, 3)

                    # Create colored masks with alpha
                    colored_masks = (
                        masks_expanded * colors_expanded * self.config.mask_alpha
                    )

                    # Apply masks with proper alpha compositing
                    for j in range(len(batch_masks)):
                        mask_alpha = masks_expanded[j] * self.config.mask_alpha
                        inv_alpha = 1.0 - mask_alpha

                        # In-place blending to reduce memory allocation
                        img_float *= inv_alpha
                        img_float += colored_masks[j]

                    # Clean up batch arrays
                    del masks_batch, masks_expanded, colors_expanded, colored_masks

                # Force garbage collection after processing all masks
                gc.collect()

                # Draw mask contours with optimized approach
                for i, mask in enumerate(processed_masks):
                    if not np.any(mask):  # Skip empty masks
                        continue

                    class_id = class_ids[i] if class_ids is not None else i
                    color = self._get_color(class_id, i)

                    # Find contours on binary mask
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Draw contours directly on float image (converted to uint8 temporarily)
                    if contours:
                        img_uint8_temp = (img_float * 255).astype(np.uint8)
                        cv2.drawContours(
                            img_uint8_temp,
                            contours,
                            -1,
                            color,
                            thickness=1,
                            lineType=(
                                cv2.LINE_AA if self.config.anti_aliasing else cv2.LINE_8
                            ),
                        )
                        img_float = img_uint8_temp.astype(np.float32) / 255.0

                        # Clean up temporary array
                        del img_uint8_temp

        # Convert back to uint8 for drawing operations
        image_result = (img_float * 255).astype(np.uint8)

        # Clean up float image
        del img_float

        # Draw bounding boxes and labels with enhanced styling
        if boxes is not None and self.config.draw_bbox:
            for i, box in enumerate(boxes):
                # Scale to image dimensions (assuming normalized coordinates)
                # box = box * image.shape[1]  # TODO: Make this configurable
                box[::2] *= image.shape[1]
                box[1::2] *= image.shape[0]
                x1, y1, x2, y2 = box.astype(int)

                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

                class_id = class_ids[i] if class_ids is not None else i
                score = scores[i] if scores is not None else 1.0
                color = self._get_color(class_id, i)

                # Draw bounding box
                if self.config.anti_aliasing:
                    cv2.rectangle(
                        image_result,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.config.bbox_thickness,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.rectangle(
                        image_result,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.config.bbox_thickness,
                    )

                # Draw enhanced label with background
                if self.config.draw_text:
                    label_parts = []

                    if self.config.draw_class and class_ids is not None:
                        class_name = self._get_class_name(class_id)
                        label_parts.append(class_name)

                    if self.config.draw_score and scores is not None:
                        label_parts.append(f"{score:.2f}")

                    if label_parts:
                        label = " ".join(label_parts)
                        self._draw_enhanced_label(image_result, label, (x1, y1), color)

        # Final garbage collection to ensure cleanup
        gc.collect()

        return image_result

    def _draw_enhanced_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ):
        """
        Draw enhanced label with background and improved styling.
        Based on the reference implementation approach.

        Args:
            image: Image to draw on
            text: Text to draw
            position: Position (x, y) for the label
            color: Text color
        """
        x, y = position
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        font_thickness = self.config.font_thickness

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font_face, font_scale, font_thickness
        )

        # Calculate label background rectangle
        padding = 4
        bg_x1 = x
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline

        # Ensure background rectangle is within image bounds
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(image.shape[1], bg_x2)
        bg_y2 = min(image.shape[0], bg_y2)

        # Draw background rectangle with same color as bounding box
        if self.config.text_background:
            cv2.rectangle(
                image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1  # Filled rectangle
            )

        # Calculate text position (centered in background)
        text_x = bg_x1 + padding // 2
        text_y = bg_y2 - baseline - padding // 2

        # Ensure text position is within bounds
        text_x = max(0, min(text_x, image.shape[1] - text_width))
        text_y = max(text_height, min(text_y, image.shape[0]))

        # Draw text with white color for better contrast
        text_color = (255, 255, 255)  # White text on colored background

        if self.config.anti_aliasing:
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font_face,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font_face,
                font_scale,
                text_color,
                font_thickness,
            )

    def _mask_to_polygons(self, mask: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """
        Convert binary mask to polygon contours.
        Based on the reference implementation.

        Args:
            mask: Binary mask

        Returns:
            Tuple of (polygons, has_holes)
        """
        mask = np.ascontiguousarray(mask)
        contours, hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if hierarchy is None:  # empty mask
            return [], False

        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0

        # Convert contours to flattened polygons
        polygons = []
        for contour in contours:
            if len(contour) >= 3:  # Valid polygon needs at least 3 points
                polygon = contour.flatten()
                if len(polygon) >= 6:  # At least 3 points (x, y pairs)
                    polygons.append(polygon + 0.5)  # Add 0.5 for sub-pixel accuracy

        return polygons, has_holes

    def _get_polygon_area(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """
        Calculate polygon area using the shoelace formula.

        Args:
            x_coords: X coordinates of polygon vertices
            y_coords: Y coordinates of polygon vertices

        Returns:
            Polygon area
        """
        return 0.5 * abs(
            sum(
                x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]
                for i in range(-1, len(x_coords) - 1)
            )
        )

    def visualize_with_polygons(
        self, image: np.ndarray, results: SegmentationResult, draw_polygons: bool = True
    ) -> np.ndarray:
        """
        Visualize segmentation results with polygon contours.
        Alternative visualization method based on reference implementation.

        Args:
            image: Input image
            results: Segmentation results
            draw_polygons: Whether to draw polygon contours

        Returns:
            Image with polygon-based segmentation visualizations
        """
        if len(results) == 0:
            return image

        # Filter by confidence threshold
        valid_mask = results.scores >= self.config.show_confidence_threshold
        if not np.any(valid_mask):
            return image

        boxes = results.boxes[valid_mask] if results.boxes is not None else None
        scores = results.scores[valid_mask]
        class_ids = (
            results.class_ids[valid_mask] if results.class_ids is not None else None
        )
        masks = results.masks[valid_mask] if results.masks is not None else None

        if masks is None:
            return image

        img_out = image.copy()

        for i, mask in enumerate(masks):
            class_id = class_ids[i] if class_ids is not None else i
            score = scores[i]
            color = self._get_color(class_id, i)

            # Ensure mask is 2D and properly sized
            if mask.ndim == 3:
                mask = mask.squeeze()

            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8), (image.shape[1], image.shape[0])
                )
                mask = mask.astype(np.float32) / 255.0

            # Convert to binary mask
            binary_mask = (mask > 0.5).astype(np.uint8)

            if not np.sum(binary_mask):
                continue

            # Convert mask to polygons
            polygons, has_holes = self._mask_to_polygons(binary_mask)

            if not polygons:
                continue

            # Create colored mask overlay
            colored_mask = np.repeat(
                binary_mask[:, :, np.newaxis], 3, axis=2
            ) * np.array(color)

            # Blend mask with image
            img_out = cv2.addWeighted(
                colored_mask.astype(np.uint8),
                self.config.mask_alpha,
                img_out,
                1 - self.config.mask_alpha,
                0,
            )

            # Draw polygon contours
            if draw_polygons:
                polygon_areas = []
                for polygon in polygons:
                    x_coords = polygon[::2]
                    y_coords = polygon[1::2]
                    polygon_areas.append(self._get_polygon_area(x_coords, y_coords))

                    # Draw polygon contour
                    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(
                        img_out,
                        [pts],
                        isClosed=True,
                        color=color,
                        thickness=2,
                        lineType=(
                            cv2.LINE_AA if self.config.anti_aliasing else cv2.LINE_8
                        ),
                    )

                # Draw label at the center of the largest polygon
                if polygon_areas and self.config.draw_text:
                    largest_polygon_idx = np.argmax(polygon_areas)
                    largest_polygon = polygons[largest_polygon_idx]

                    # Calculate centroid
                    x_coords = largest_polygon[::2]
                    y_coords = largest_polygon[1::2]
                    centroid_x = int(np.mean(x_coords))
                    centroid_y = int(np.mean(y_coords))

                    # Create label
                    label_parts = []
                    if self.config.draw_class and class_ids is not None:
                        class_name = self._get_class_name(class_id)
                        label_parts.append(class_name)
                    if self.config.draw_score:
                        label_parts.append(f"{score:.2f}")

                    if label_parts:
                        label = " ".join(label_parts)
                        self._draw_centered_label(
                            img_out, label, (centroid_x, centroid_y), color
                        )

        return img_out

    def _draw_centered_label(
        self,
        image: np.ndarray,
        text: str,
        center: Tuple[int, int],
        color: Tuple[int, int, int],
    ):
        """
        Draw label centered at given position with background.

        Args:
            image: Image to draw on
            text: Text to draw
            center: Center position (x, y)
            color: Text color
        """
        cx, cy = center
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        font_thickness = self.config.font_thickness

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font_face, font_scale, font_thickness
        )

        # Calculate centered position
        text_x = cx - text_width // 2
        text_y = cy + text_height // 2

        # Draw background rectangle
        padding = 4
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_height - padding
        bg_x2 = text_x + text_width + padding
        bg_y2 = text_y + baseline + padding

        # Ensure positions are within image bounds
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(image.shape[1], bg_x2)
        bg_y2 = min(image.shape[0], bg_y2)

        text_x = max(0, min(text_x, image.shape[1] - text_width))
        text_y = max(text_height, min(text_y, image.shape[0]))

        # Draw background
        if self.config.text_background:
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Draw text
        text_color = (255, 255, 255)  # White text
        if self.config.anti_aliasing:
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font_face,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font_face,
                font_scale,
                text_color,
                font_thickness,
            )

    def _visualize_keypoints(
        self, image: np.ndarray, results: KeypointResult
    ) -> np.ndarray:
        """
        Visualize keypoint detection results.

        Args:
            image: Input image
            results: Keypoint results

        Returns:
            Image with keypoint visualizations
        """
        if len(results) == 0:
            return image

        # Filter by confidence threshold
        valid_mask = results.scores >= self.config.show_confidence_threshold
        if not np.any(valid_mask):
            return image

        keypoints = results.keypoints[valid_mask]
        scores = results.scores[valid_mask]
        boxes = results.boxes[valid_mask] if results.boxes is not None else None

        # Limit number of visualizations
        if len(keypoints) > self.config.max_labels_per_image:
            # Sort by confidence and keep top persons
            top_indices = np.argsort(scores)[::-1][: self.config.max_labels_per_image]
            keypoints = keypoints[top_indices]
            scores = scores[top_indices]
            boxes = boxes[top_indices] if boxes is not None else None

        # Draw person bounding boxes
        if boxes is not None and self.config.draw_person_bbox:
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box.astype(int)
                color = self._get_color(0, i)  # Use person class color

                # Draw bounding box
                if self.config.anti_aliasing:
                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        self.config.bbox_thickness,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.rectangle(
                        image, (x1, y1), (x2, y2), color, self.config.bbox_thickness
                    )

                # Draw person confidence
                if self.config.draw_text and self.config.draw_score:
                    label = f"Person {score:.2f}"
                    self._draw_label(image, label, (x1, y1), color)

        # Draw keypoints and skeleton
        for i, person_keypoints in enumerate(keypoints):
            person_color = self._get_color(0, i)

            # Draw skeleton connections first (so they appear behind keypoints)
            if self.config.draw_skeleton:
                self._draw_skeleton(image, person_keypoints, person_color)

            # Draw individual keypoints
            if self.config.draw_keypoints:
                self._draw_keypoints(image, person_keypoints, person_color)

        return image

    def _draw_skeleton(
        self, image: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]
    ):
        """
        Draw skeleton connections between keypoints.

        Args:
            image: Image to draw on
            keypoints: Keypoint array with shape (num_keypoints, 3) - [x, y, visibility]
            color: Color for skeleton lines
        """
        for connection in self.coco_skeleton:
            kp1_idx, kp2_idx = connection

            # Check if both keypoints exist and are visible
            if (
                kp1_idx < len(keypoints)
                and kp2_idx < len(keypoints)
                and keypoints[kp1_idx, 2] > 0.5
                and keypoints[kp2_idx, 2] > 0.5
            ):
                pt1 = (int(keypoints[kp1_idx, 0]), int(keypoints[kp1_idx, 1]))
                pt2 = (int(keypoints[kp2_idx, 0]), int(keypoints[kp2_idx, 1]))

                # Draw line
                if self.config.anti_aliasing:
                    cv2.line(
                        image,
                        pt1,
                        pt2,
                        color,
                        self.config.skeleton_thickness,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.line(image, pt1, pt2, color, self.config.skeleton_thickness)

    def _draw_keypoints(
        self, image: np.ndarray, keypoints: np.ndarray, base_color: Tuple[int, int, int]
    ):
        """
        Draw individual keypoints.

        Args:
            image: Image to draw on
            keypoints: Keypoint array with shape (num_keypoints, 3) - [x, y, visibility]
            base_color: Base color for keypoints
        """
        for i, (x, y, visibility) in enumerate(keypoints):
            if visibility > 0.5:  # Only draw visible keypoints
                center = (int(x), int(y))

                # Use different colors for different keypoint types
                if i < 5:  # Head keypoints
                    kp_color = (0, 255, 255)  # Yellow
                elif i < 11:  # Arm keypoints
                    kp_color = (255, 0, 255)  # Magenta
                else:  # Leg keypoints
                    kp_color = (255, 255, 0)  # Cyan

                # Draw keypoint circle
                if self.config.anti_aliasing:
                    cv2.circle(
                        image,
                        center,
                        self.config.keypoint_radius,
                        kp_color,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        image,
                        center,
                        self.config.keypoint_radius + 1,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.circle(image, center, self.config.keypoint_radius, kp_color, -1)
                    cv2.circle(
                        image, center, self.config.keypoint_radius + 1, (0, 0, 0), 1
                    )

    def _get_color(self, class_id: int, instance_id: int) -> Tuple[int, int, int]:
        """
        Get color for visualization based on class ID or instance ID.

        Args:
            class_id: Class ID
            instance_id: Instance ID

        Returns:
            BGR color tuple
        """
        if self.config.bbox_color_mode == "fixed":
            return self.config.fixed_bbox_color
        elif self.config.bbox_color_mode == "random":
            return self.config.colors[instance_id % len(self.config.colors)]
        else:  # class-based coloring
            return self.config.colors[class_id % len(self.config.colors)]

    def _get_class_name(self, class_id: int) -> str:
        """
        Get class name for a given class ID.

        Args:
            class_id: Class ID

        Returns:
            Class name string
        """
        if class_id < len(self.coco_class_names):
            return self.coco_class_names[class_id]
        else:
            return f"class_{class_id}"

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ):
        """
        Draw text label with background.

        Args:
            image: Image to draw on
            text: Text to draw
            position: Position (x, y) for text
            color: Text color
        """
        x, y = position
        x += self.config.label_offset[0]
        y += self.config.label_offset[1]

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.font_thickness,
        )

        # Draw background rectangle if enabled
        if self.config.text_background:
            bg_color = self.config.text_background_color

            # Create background rectangle
            bg_x1 = x - 2
            bg_y1 = y - text_height - baseline - 2
            bg_x2 = x + text_width + 2
            bg_y2 = y + baseline + 2

            # Draw semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            image[:] = cv2.addWeighted(
                image,
                1 - self.config.text_background_alpha,
                overlay,
                self.config.text_background_alpha,
                0,
            )

        # Draw text
        if self.config.anti_aliasing:
            cv2.putText(
                image,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.font_color,
                self.config.font_thickness,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                image,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.font_color,
                self.config.font_thickness,
            )

    def save(self, image: np.ndarray, path: str) -> bool:
        """
        Save visualized image to file.

        Args:
            image: Image to save
            path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            success = cv2.imwrite(path, image)
            if success:
                logger.info(f"Saved visualization to {path}")
            else:
                logger.error(f"Failed to save visualization to {path}")
            return success
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            return False

    def show(
        self,
        image: np.ndarray,
        window_name: str = "Visualization",
        wait_key: bool = True,
    ) -> None:
        """
        Display image in a window.

        Args:
            image: Image to display
            window_name: Window name
            wait_key: Whether to wait for key press
        """
        try:
            cv2.imshow(window_name, image)
            if wait_key:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error displaying image: {str(e)}")

    def create_grid(
        self, images: List[np.ndarray], grid_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Create a grid of images for batch visualization.

        Args:
            images: List of images to arrange in grid
            grid_size: Grid size (rows, cols). If None, automatically determined.

        Returns:
            Grid image
        """
        if not images:
            raise ValueError("No images provided for grid creation")

        num_images = len(images)

        # Determine grid size if not provided
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            grid_size = (rows, cols)

        rows, cols = grid_size

        # Get image dimensions (assume all images have same size)
        h, w = images[0].shape[:2]

        # Create grid image
        grid_image = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            y1, y2 = row * h, (row + 1) * h
            x1, x2 = col * w, (col + 1) * w

            # Resize image if necessary
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))

            grid_image[y1:y2, x1:x2] = img

        return grid_image


def visualize_yolov5_seg_results(
    image: np.ndarray,
    detections: List[Dict[str, np.ndarray]],
    meta_arch: str = "yolov5_seg",
    **kwargs,
) -> np.ndarray:
    """
    Visualize YOLOv5 instance segmentation results on an image.

    Args:
        image: Input image as numpy array (H, W, C)
        detections: List of detection dictionaries containing:
            - detection_boxes: Bounding boxes in format [y_min, x_min, y_max, x_max]
            - detection_scores: Confidence scores
            - detection_classes: Class IDs
            - mask: Segmentation masks
        meta_arch: Model architecture name (default: "yolov5_seg")
        **kwargs: Additional visualization parameters:
            - class_names: List of class names
            - mask_threshold: Threshold for mask visualization (default: 0.5)
            - show_boxes: Whether to show bounding boxes (default: True)
            - show_labels: Whether to show class labels (default: True)
            - show_scores: Whether to show confidence scores (default: True)
            - box_thickness: Thickness of bounding box lines (default: 2)
            - font_scale: Font scale for text (default: 0.5)
            - alpha: Transparency for mask overlay (default: 0.5)

    Returns:
        Visualized image with segmentation results

    Raises:
        ValueError: If input parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be empty")

    if not detections or len(detections) == 0:
        return image.copy()

    # Get visualization parameters
    class_names = kwargs.get("class_names", None)
    mask_threshold = kwargs.get("mask_threshold", 0.5)
    show_boxes = kwargs.get("show_boxes", True)
    show_labels = kwargs.get("show_labels", True)
    show_scores = kwargs.get("show_scores", True)
    box_thickness = kwargs.get("box_thickness", 2)
    font_scale = kwargs.get("font_scale", 0.5)
    alpha = kwargs.get("alpha", 0.5)

    # Create a copy of the image for visualization
    vis_image = image.copy()
    img_height, img_width = image.shape[:2]

    # Process each detection batch (typically only one batch)
    for batch_detections in detections:
        if not isinstance(batch_detections, dict):
            continue

        boxes = batch_detections.get("detection_boxes", np.array([]))
        scores = batch_detections.get("detection_scores", np.array([]))
        classes = batch_detections.get("detection_classes", np.array([]))
        masks = batch_detections.get("mask", np.array([]))

        if len(boxes) == 0:
            continue

        # Scale bounding boxes to image dimensions
        scaled_boxes = boxes.copy()

        # Check if boxes are normalized (values between 0 and 1)
        if np.all(boxes <= 1.0) and np.all(boxes >= 0.0):
            # Boxes are normalized, scale to image dimensions
            scaled_boxes[:, [0, 2]] *= img_height  # y coordinates
            scaled_boxes[:, [1, 3]] *= img_width  # x coordinates

        # Convert to integer coordinates
        scaled_boxes = scaled_boxes.astype(int)

        # Process each detection
        for i in range(len(boxes)):
            # Get detection components
            box = scaled_boxes[i]
            score = scores[i] if i < len(scores) else 0.0
            class_id = int(classes[i]) if i < len(classes) else 0
            mask = masks[i] if i < len(masks) else None

            # Get class name
            if class_names and 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"class_{class_id}"

            # Generate color for this class
            color = _generate_color(class_id)

            # Draw segmentation mask
            if mask is not None and mask.size > 0:
                # Apply threshold to mask
                binary_mask = (mask > mask_threshold).astype(np.uint8)

                # Resize mask to image dimensions if needed
                if binary_mask.shape != (img_height, img_width):
                    binary_mask = cv2.resize(
                        binary_mask,
                        (img_width, img_height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # Create colored mask
                colored_mask = np.zeros_like(vis_image)
                colored_mask[binary_mask > 0] = color

                # Blend mask with image
                vis_image = cv2.addWeighted(
                    vis_image, 1 - alpha, colored_mask, alpha, 0
                )

                # Draw mask contours
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis_image, contours, -1, color, 2)

            # Draw bounding box
            if show_boxes:
                y_min, x_min, y_max, x_max = box
                cv2.rectangle(
                    vis_image, (x_min, y_min), (x_max, y_max), color, box_thickness
                )

            # Draw label and score
            if show_labels or show_scores:
                label_parts = []
                if show_labels:
                    label_parts.append(class_name)
                if show_scores:
                    label_parts.append(f"{score:.2f}")

                label = " ".join(label_parts)

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )

                y_min, x_min = box[:2]
                label_y = max(y_min - 10, text_height + 10)

                # Draw text background
                cv2.rectangle(
                    vis_image,
                    (x_min, label_y - text_height - baseline),
                    (x_min + text_width, label_y + baseline),
                    color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    vis_image,
                    label,
                    (x_min, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                )

    return vis_image


def visualize_instance_segmentation_result(
    image: np.ndarray, detections: List[Dict[str, np.ndarray]], meta_arch: str, **kwargs
) -> np.ndarray:
    """
    Visualize instance segmentation results based on model architecture.

    This function serves as a dispatcher that calls the appropriate visualization
    function based on the specified model architecture.

    Args:
        image: Input image as numpy array (H, W, C)
        detections: List of detection dictionaries
        meta_arch: Model architecture name ("sparseinst", "yolov5_seg", "yolov8_seg", "yolact")
        **kwargs: Additional visualization parameters

    Returns:
        Visualized image with segmentation results

    Raises:
        ValueError: If meta_arch is not supported
    """
    if meta_arch == "sparseinst":
        # Use existing sparseinst visualization (if available)
        return visualize_sparseinst_results(image, detections, **kwargs)
    elif meta_arch == "yolov5_seg":
        return visualize_yolov5_seg_results(image, detections, meta_arch, **kwargs)
    elif meta_arch == "yolov8_seg":
        # Use YOLOv8 segmentation visualization
        return _visualize_segmentation(image, detections, **kwargs)
    elif meta_arch == "yolact":
        # Use existing YOLACT visualization (if available)
        return visualize_yolact_results(image, detections, **kwargs)
    else:
        raise ValueError(f"Unsupported meta_arch: {meta_arch}")


def visualize_sparseinst_results(
    image: np.ndarray, detections: List[Dict[str, np.ndarray]], **kwargs
) -> np.ndarray:
    """
    Placeholder for SparseInst visualization.

    Args:
        image: Input image
        detections: Detection results
        **kwargs: Additional parameters

    Returns:
        Visualized image (currently returns original image)
    """
    # TODO: Implement SparseInst-specific visualization
    logger.warning("SparseInst visualization not implemented, returning original image")
    return image.copy()


def visualize_yolact_results(
    image: np.ndarray, detections: List[Dict[str, np.ndarray]], **kwargs
) -> np.ndarray:
    """
    Placeholder for YOLACT visualization.

    Args:
        image: Input image
        detections: Detection results
        **kwargs: Additional parameters

    Returns:
        Visualized image (currently returns original image)
    """
    # TODO: Implement YOLACT-specific visualization
    logger.warning("YOLACT visualization not implemented, returning original image")
    return image.copy()


def visualize_openpose_results(
    image: np.ndarray,
    results: Union[Dict[str, np.ndarray], KeypointResult],
    detection_threshold: float = 0.5,
    joint_threshold: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """
    Visualize OpenPose-style pose estimation results.

    Args:
        image: Input image as numpy array
        results: Pose estimation results
        detection_threshold: Minimum confidence threshold for person detection
        joint_threshold: Minimum confidence threshold for joint visibility
        **kwargs: Additional visualization parameters

    Returns:
        Visualized image with pose estimation results
    """
    # Handle different input formats
    if isinstance(results, dict):
        if "predictions" in results:
            # COCO format predictions
            predictions = results["predictions"]
            keypoints_list = []
            scores_list = []

            for pred in predictions:
                if pred.get("score", 0) >= detection_threshold:
                    # Convert COCO keypoints format to array
                    coco_kpts = pred[
                        "keypoints"
                    ]  # List of [x1, y1, v1, x2, y2, v2, ...]
                    kpts_array = np.array(coco_kpts).reshape(-1, 3)
                    keypoints_list.append(kpts_array)
                    scores_list.append(pred["score"])

            if keypoints_list:
                keypoints = np.array(keypoints_list)
                scores = np.array(scores_list)
                bboxes = None
            else:
                return image
        else:
            # Direct format
            keypoints = results.get("keypoints", np.array([]))
            scores = results.get("scores", np.array([]))
            bboxes = results.get("bboxes", None)
    elif isinstance(results, KeypointResult):
        keypoints = results.keypoints
        scores = results.scores
        bboxes = results.boxes
    else:
        logger.warning("Unsupported results format for OpenPose visualization")
        return image

    if len(keypoints) == 0:
        return image

    # Create a copy of the image for visualization
    vis_image = image.copy()

    # COCO keypoint connections for skeleton
    joint_pairs = [
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

    # Visualize each person
    for i, (person_keypoints, score) in enumerate(zip(keypoints, scores)):
        if score < detection_threshold:
            continue

        # Draw person bounding box if available
        if bboxes is not None and i < len(bboxes):
            bbox = bboxes[i]
            x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw person confidence score
            label = f"Person {score:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Determine joint visibility
        joint_visible = person_keypoints[:, 2] > joint_threshold

        # Draw skeleton connections
        for joint0, joint1 in joint_pairs:
            if (
                joint0 < len(joint_visible)
                and joint1 < len(joint_visible)
                and joint_visible[joint0]
                and joint_visible[joint1]
            ):
                pt1 = (
                    int(person_keypoints[joint0][0]),
                    int(person_keypoints[joint0][1]),
                )
                pt2 = (
                    int(person_keypoints[joint1][0]),
                    int(person_keypoints[joint1][1]),
                )

                # Draw skeleton line with different colors for different body parts
                if joint0 < 5 or joint1 < 5:  # Head connections
                    color = (0, 255, 0)  # Green
                elif joint0 < 11 or joint1 < 11:  # Arm connections
                    color = (255, 0, 0)  # Blue
                else:  # Leg connections
                    color = (0, 0, 255)  # Red

                cv2.line(vis_image, pt1, pt2, color, 3)

        # Draw individual keypoints
        for j, (x, y, visibility) in enumerate(person_keypoints):
            if visibility > joint_threshold:
                center = (int(x), int(y))

                # Use different colors for different body parts
                if j < 5:  # Head keypoints
                    color = (0, 255, 255)  # Yellow
                elif j < 11:  # Arm keypoints
                    color = (255, 0, 255)  # Magenta
                else:  # Leg keypoints
                    color = (255, 255, 0)  # Cyan

                cv2.circle(vis_image, center, 5, color, -1)
                cv2.circle(vis_image, center, 6, (0, 0, 0), 1)

    return vis_image


def visualize_pose_estimation_result(
    results: Union[Dict[str, Any], KeypointResult],
    img: np.ndarray,
    dataset_name: str = "cocopose",
    detection_threshold: float = 0.5,
    joint_threshold: float = 0.5,
    meta_arch: str = "yolov8",
    **kwargs,
) -> np.ndarray:
    """
    Universal pose estimation visualization function.

    This function provides a unified interface for visualizing pose estimation results
    from different model architectures (YOLOv8, OpenPose, CenterPose).

    Args:
        results: Pose estimation results in various formats
        img: Input image(s) as numpy array
        dataset_name: Dataset name (default: "cocopose")
        detection_threshold: Minimum confidence threshold for person detection
        joint_threshold: Minimum confidence threshold for joint visibility
        meta_arch: Model architecture ("yolov8", "openpose", "centerpose")
        **kwargs: Additional visualization parameters

    Returns:
        Visualized image with pose estimation results
    """
    # Handle batch dimension for images
    if len(img.shape) == 4:
        image = img[0]  # Take first image from batch
    else:
        image = img

    # Convert color format if needed (for CenterPose compatibility)
    if "centerpose" in meta_arch.lower():
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Route to appropriate visualization function based on architecture
    if "yolov8" in meta_arch.lower() or "nanodet" in meta_arch.lower():
        return visualize_yolov8_pose_results(
            image, results, detection_threshold, joint_threshold, **kwargs
        )
    elif "openpose" in meta_arch.lower() or "pose_estimation" in meta_arch.lower():
        return visualize_openpose_results(
            image, results, detection_threshold, joint_threshold, **kwargs
        )
    elif "centerpose" in meta_arch.lower():
        # CenterPose uses similar visualization to YOLOv8
        return visualize_yolov8_pose_results(
            image, results, detection_threshold, joint_threshold, **kwargs
        )
    else:
        logger.warning(
            f"Unknown pose estimation architecture: {meta_arch}, using default visualization"
        )
        return visualize_yolov8_pose_results(
            image, results, detection_threshold, joint_threshold, **kwargs
        )


def create_pose_visualization_config(
    model_type: str = "yolov8",
    detection_threshold: float = 0.5,
    joint_threshold: float = 0.5,
    draw_skeleton: bool = True,
    draw_keypoints: bool = True,
    draw_person_bbox: bool = True,
    **kwargs,
) -> VisualizationConfig:
    """
    Create a visualization configuration optimized for pose estimation.

    Args:
        model_type: Type of pose estimation model
        detection_threshold: Minimum confidence threshold for person detection
        joint_threshold: Minimum confidence threshold for joint visibility
        draw_skeleton: Whether to draw skeleton connections
        draw_keypoints: Whether to draw individual keypoints
        draw_person_bbox: Whether to draw person bounding boxes
        **kwargs: Additional configuration parameters

    Returns:
        VisualizationConfig optimized for pose estimation
    """
    config = VisualizationConfig()

    # Enable keypoint visualization
    config.kp = True
    config.draw_keypoints = draw_keypoints
    config.draw_skeleton = draw_skeleton
    config.draw_person_bbox = draw_person_bbox

    # Set thresholds
    config.show_confidence_threshold = detection_threshold
    config.kp_visibility_threshold = joint_threshold

    # Optimize for pose visualization
    config.keypoint_radius = 5
    config.skeleton_thickness = 3
    config.bbox_thickness = 2
    config.font_scale = 0.7
    config.anti_aliasing = True

    # Apply any additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
