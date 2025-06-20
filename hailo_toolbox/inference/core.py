from typing import Any, List, Optional, AnyStr, Union, Callable, Dict, Type, Iterable
from enum import Enum

from hailo_toolbox.inference.hailo_engine import HailoInference
from hailo_toolbox.inference.onnx_engine import ONNXInference
from hailo_toolbox.utils.config import Config
import numpy as np
import logging
import os
import os.path as osp


logger = logging.getLogger(__name__)


class CallbackType(Enum):
    """Enumeration of callback types supported by the registry"""

    PRE_PROCESSOR = "pre_processor"
    POST_PROCESSOR = "post_processor"
    VISUALIZER = "visualizer"
    SOURCE = "source"
    COLLAT_INFER = "collat_infer"


def empty_callback(args) -> None:
    """Default empty callback function that does nothing

    Returns:
        None
    """
    return args


class CallbackRegistry:
    """
    A universal registry for managing callbacks/functions/classes by name and type.

    This registry provides a centralized way to register and retrieve different types
    of callbacks, functions, or classes based on their name and category type.

    Features:
    - Type-safe registration and retrieval
    - Support for decorator-style registration with single or multiple names
    - Extensible callback types via Enum
    - Automatic fallback to empty callback for missing entries
    - Comprehensive error handling and validation
    - Support for multiple names mapping to the same callback (shared registration)

    Example:
        # Create registry instance
        registry = CallbackRegistry()

        # Register using decorator with single name
        @registry.register("my_preprocessor", CallbackType.PRE_PROCESSOR)
        def my_preprocess_func(data):
            return processed_data

        # Register using decorator with multiple names
        @registry.register(["yolov8det", "yolov8seg"], CallbackType.PRE_PROCESSOR)
        def yolo_preprocess_func(data):
            return processed_data

        # Register directly with multiple names
        registry.register_callback(["name1", "name2"], CallbackType.POST_PROCESSOR, my_postprocess_func)

        # Retrieve registered callback (works with any of the registered names)
        preprocessor = registry.get_callback("yolov8det", CallbackType.PRE_PROCESSOR)
    """

    def __init__(self):
        """Initialize the callback registry with empty dictionaries for each type"""
        # Main storage: Dict[CallbackType, Dict[str, Union[Callable, Type]]]
        self._callbacks: Dict[CallbackType, Dict[str, Union[Callable, Type]]] = {
            callback_type: {} for callback_type in CallbackType
        }

        # Legacy compatibility - maintain separate dictionaries
        self.engines: Dict[str, Any] = {}
        self.PreProcessor: Dict[str, Callable] = {}
        self.PostProcessor: Dict[str, Callable] = {}
        self.Visualizer: Dict[str, Callable] = {}
        self.Source: Dict[str, Callable] = {}
        self.CollatInfer: Dict[str, Callable] = {}

    def register(
        self, names: Union[str, List[str]], callback_type: CallbackType
    ) -> Callable:
        """
        Decorator for registering callbacks with specified name(s) and type.

        Args:
            names: Single name or list of names to register the callback under
            callback_type: Type of callback from CallbackType enum

        Returns:
            Decorator function

        Example:
            # Single name registration
            @registry.register("yolov8_preprocess", CallbackType.PRE_PROCESSOR)
            def preprocess_func(data):
                return processed_data

            # Multiple names registration
            @registry.register(["yolov8det", "yolov8seg"], CallbackType.PRE_PROCESSOR)
            def yolo_preprocess_func(data):
                return processed_data
        """

        def decorator(func: Union[Callable, Type]) -> Union[Callable, Type]:
            self.register_callback(names, callback_type, func)
            return func

        return decorator

    def register_callback(
        self,
        names: Union[str, List[str]],
        callback_type: CallbackType,
        callback: Union[Callable, Type],
    ) -> None:
        """
        Register a callback/function/class with the specified name(s) and type.

        Args:
            names: Single name or list of names to register the callback under
            callback_type: Type of callback from CallbackType enum
            callback: Function, method, or class to register

        Raises:
            ValueError: If callback_type is not a valid CallbackType or names is empty
            TypeError: If callback is not callable or a class, or names is not string/list
            KeyError: If any name already exists for the given type (overwrites with warning)
        """
        if not isinstance(callback_type, CallbackType):
            raise ValueError(
                f"Invalid callback type: {callback_type}. Must be a CallbackType enum."
            )

        if not (callable(callback) or isinstance(callback, type)):
            raise TypeError(
                f"Callback must be callable or a class, got {type(callback)}"
            )

        # Normalize names to list
        if isinstance(names, str):
            names_list = [names]
        elif isinstance(names, (list, tuple)):
            names_list = list(names)
        else:
            raise TypeError(
                f"Names must be string or list of strings, got {type(names)}"
            )

        if not names_list:
            raise ValueError("At least one name must be provided")

        # Validate all names are strings
        for name in names_list:
            if not isinstance(name, str):
                raise TypeError(
                    f"All names must be strings, got {type(name)} for '{name}'"
                )

        # Register callback under all specified names
        registered_names = []
        for name in names_list:
            # Check for existing registration
            if name in self._callbacks[callback_type]:
                logger.warning(
                    f"Overwriting existing callback '{name}' for type '{callback_type.value}'"
                )

            # Register in main storage
            self._callbacks[callback_type][name] = callback
            registered_names.append(name)

            # Maintain legacy compatibility
            self._update_legacy_storage(name, callback_type, callback)

        logger.debug(
            f"Registered callback under names {registered_names} for type '{callback_type.value}'"
        )

    def _update_legacy_storage(
        self, name: str, callback_type: CallbackType, callback: Union[Callable, Type]
    ) -> None:
        """Update legacy storage dictionaries for backward compatibility"""
        if callback_type == CallbackType.PRE_PROCESSOR:
            self.PreProcessor[name] = callback
        elif callback_type == CallbackType.POST_PROCESSOR:
            self.PostProcessor[name] = callback
        elif callback_type == CallbackType.VISUALIZER:
            self.Visualizer[name] = callback
        elif callback_type == CallbackType.SOURCE:
            self.Source[name] = callback
        elif callback_type == CallbackType.COLLAT_INFER:
            self.CollatInfer[name] = callback

    def get_callback(
        self, name: str, callback_type: CallbackType, default: Optional[Callable] = None
    ) -> Union[Callable, Type]:
        """
        Retrieve a registered callback by name and type.

        Args:
            name: Name of the callback to retrieve
            callback_type: Type of callback to retrieve
            default: Default callback to return if not found (defaults to empty_callback)

        Returns:
            The registered callback, or default callback if not found

        Raises:
            ValueError: If callback_type is not a valid CallbackType
        """
        if not isinstance(callback_type, CallbackType):
            raise ValueError(
                f"Invalid callback type: {callback_type}. Must be a CallbackType enum."
            )

        if name not in self._callbacks[callback_type]:
            if "base" in self._callbacks[callback_type]:
                name = "base"
            else:
                return empty_callback

        callback = self._callbacks[callback_type][name]

        return callback

    def list_callbacks(
        self, callback_type: Optional[CallbackType] = None
    ) -> Dict[str, List[str]]:
        """
        List all registered callbacks, optionally filtered by type.

        Args:
            callback_type: Optional filter by callback type

        Returns:
            Dictionary mapping callback type names to lists of registered names
        """
        if callback_type:
            return {callback_type.value: list(self._callbacks[callback_type].keys())}

        return {
            cb_type.value: list(callbacks.keys())
            for cb_type, callbacks in self._callbacks.items()
        }

    def get_shared_names(self, name: str, callback_type: CallbackType) -> List[str]:
        """
        Get all names that point to the same callback as the given name.

        Args:
            name: Name of the callback to find shared names for
            callback_type: Type of the callback

        Returns:
            List of all names (including the input name) that share the same callback

        Raises:
            ValueError: If callback_type is not valid or name is not registered
        """
        if not isinstance(callback_type, CallbackType):
            raise ValueError(
                f"Invalid callback type: {callback_type}. Must be a CallbackType enum."
            )

        if name not in self._callbacks[callback_type]:
            raise ValueError(
                f"Callback '{name}' not found for type '{callback_type.value}'"
            )

        target_callback = self._callbacks[callback_type][name]
        shared_names = []

        for registered_name, callback in self._callbacks[callback_type].items():
            if callback is target_callback:
                shared_names.append(registered_name)

        return shared_names

    def unregister_callback(
        self, names: Union[str, List[str]], callback_type: CallbackType
    ) -> bool:
        """
        Unregister callback(s) by name(s) and type.

        Args:
            names: Single name or list of names of callbacks to unregister
            callback_type: Type of the callback

        Returns:
            True if at least one callback was found and removed, False otherwise
        """
        if not isinstance(callback_type, CallbackType):
            raise ValueError(
                f"Invalid callback type: {callback_type}. Must be a CallbackType enum."
            )

        # Normalize names to list
        if isinstance(names, str):
            names_list = [names]
        else:
            names_list = list(names)

        removed_count = 0
        for name in names_list:
            if name in self._callbacks[callback_type]:
                del self._callbacks[callback_type][name]
                removed_count += 1

                # Clean up legacy storage
                if (
                    callback_type == CallbackType.PRE_PROCESSOR
                    and name in self.PreProcessor
                ):
                    del self.PreProcessor[name]
                elif (
                    callback_type == CallbackType.POST_PROCESSOR
                    and name in self.PostProcessor
                ):
                    del self.PostProcessor[name]
                elif (
                    callback_type == CallbackType.VISUALIZER and name in self.Visualizer
                ):
                    del self.Visualizer[name]
                elif callback_type == CallbackType.SOURCE and name in self.Source:
                    del self.Source[name]
                elif (
                    callback_type == CallbackType.COLLAT_INFER
                    and name in self.CollatInfer
                ):
                    del self.CollatInfer[name]

                logger.debug(
                    f"Unregistered callback '{name}' for type '{callback_type.value}'"
                )

        return removed_count > 0

    def has_callback(self, name: str, callback_type: CallbackType) -> bool:
        """
        Check if a callback is registered with the given name and type.

        Args:
            name: Name of the callback to check
            callback_type: Type of the callback

        Returns:
            True if callback exists, False otherwise
        """
        if not isinstance(callback_type, CallbackType):
            return False
        return name in self._callbacks[callback_type]

    # Convenient type-specific decorator methods with multi-name support
    def registryPreProcessor(self, *names: str) -> Callable:
        """
        Convenient decorator for registering pre-processors with multiple names.

        Args:
            *names: One or more unique identifiers for the pre-processor

        Returns:
            Decorator function

        Example:
            # Single name
            @CALLBACK_REGISTRY.registryPreProcessor("yolo_preprocess")
            def preprocess_func(image):
                return processed_image

            # Multiple names
            @CALLBACK_REGISTRY.registryPreProcessor("yolov8det", "yolov8seg")
            def yolo_preprocess_func(image):
                return processed_image
        """
        if not names:
            raise ValueError("At least one name must be provided")
        return self.register(list(names), CallbackType.PRE_PROCESSOR)

    def registryPostProcessor(self, *names: str) -> Callable:
        """
        Convenient decorator for registering post-processors with multiple names.

        Args:
            *names: One or more unique identifiers for the post-processor

        Returns:
            Decorator function

        Example:
            # Single name
            @CALLBACK_REGISTRY.registryPostProcessor("yolo_postprocess")
            def postprocess_func(predictions):
                return detections

            # Multiple names
            @CALLBACK_REGISTRY.registryPostProcessor("yolov8det", "yolov8seg")
            def yolo_postprocess_func(predictions):
                return detections
        """
        if not names:
            raise ValueError("At least one name must be provided")
        return self.register(list(names), CallbackType.POST_PROCESSOR)

    def registryVisualizer(self, *names: str) -> Callable:
        """
        Convenient decorator for registering visualizers with multiple names.

        Args:
            *names: One or more unique identifiers for the visualizer

        Returns:
            Decorator function

        Example:
            # Single name
            @CALLBACK_REGISTRY.registryVisualizer("bbox_visualizer")
            class BBoxVisualizer:
                def __init__(self, config):
                    pass

            # Multiple names
            @CALLBACK_REGISTRY.registryVisualizer("yolov8det", "yolov8seg")
            class YoloVisualizer:
                def __init__(self, config):
                    pass
        """
        if not names:
            raise ValueError("At least one name must be provided")
        return self.register(list(names), CallbackType.VISUALIZER)

    def registrySource(self, *names: str) -> Callable:
        """
        Convenient decorator for registering data sources with multiple names.

        Args:
            *names: One or more unique identifiers for the data source

        Returns:
            Decorator function

        Example:
            # Single name
            @CALLBACK_REGISTRY.registrySource("video_source")
            class VideoSource:
                def __init__(self, path):
                    pass

            # Multiple names
            @CALLBACK_REGISTRY.registrySource("video", "webcam")
            class VideoSource:
                def __init__(self, path):
                    pass
        """
        if not names:
            raise ValueError("At least one name must be provided")
        return self.register(list(names), CallbackType.SOURCE)

    def registryCollateInfer(self, *names: str) -> Callable:
        """
        Convenient decorator for registering collaborative inference components with multiple names.

        Args:
            *names: One or more unique identifiers for the collaborative inference component

        Returns:
            Decorator function

        Example:
            # Single name
            @CALLBACK_REGISTRY.registryCollateInfer("yolov8det")
            def collat_infer_func(data):
                return results

            # Multiple names
            @CALLBACK_REGISTRY.registryCollateInfer("yolov8det", "yolov8seg")
            def yolo_collat_infer_func(data):
                return results
        """
        if not names:
            raise ValueError("At least one name must be provided")
        return self.register(list(names), CallbackType.COLLAT_INFER)

    # Legacy methods for backward compatibility
    def getPreProcessor(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.PRE_PROCESSOR)

    def getPostProcessor(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.POST_PROCESSOR)

    def getVisualizer(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.VISUALIZER)

    def getSource(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.SOURCE)

    def getCollatInfer(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.COLLAT_INFER)


# Global registry instance
CALLBACK_REGISTRY = CallbackRegistry()


class InferenceEngine:
    """
    Core inference engine that orchestrates the entire inference pipeline.

    The InferenceEngine manages the complete lifecycle of model inference including:
    - Model initialization (Hailo or ONNX)
    - Data source management
    - Preprocessing pipeline
    - Inference execution
    - Postprocessing pipeline
    - Visualization and output handling

    Features:
    - Support for multiple inference backends (Hailo, ONNX)
    - Flexible callback system for customization
    - Configurable preprocessing and postprocessing
    - Built-in visualization capabilities
    - Frame-by-frame processing with debugging support

    Args:
        config: Configuration object containing all pipeline settings
        callback_name: Name identifier for callback registration lookup

    Example:
        config = Config(model="model.hef", source="video.mp4", task_type="detection")
        engine = InferenceEngine(config, "yolov8det")
        engine.run()
    """

    def __init__(self, config: Config, callback_name: Optional[AnyStr] = None) -> None:
        """
        Initialize the inference engine with configuration and callback settings.

        Args:
            config: Configuration object with model, source, and processing settings
            callback_name: Identifier for callback lookup in the registry
        """
        self.callback_name = callback_name
        if callback_name is None:
            self.preprocess_name = None
            self.postprocess_name = None
            self.visualization_name = None
            self.collate_infer_name = None
        else:
            self.preprocess_name = callback_name
            self.postprocess_name = callback_name
            self.visualization_name = callback_name
            self.collate_infer_name = callback_name

        self.config = config

        # Determine inference mode and backend type
        if self.config.command == "infer":
            self.infer = True
            if self.config.model.endswith(".hef"):
                self.infer_type = "hailo"
            elif self.config.model.endswith(".onnx"):
                self.infer_type = "onnx"
            else:
                raise ValueError(f"Unsupported model type: {self.config.model}")
        else:
            self.infer = False

        self.task_type = self.config.task_type
        self.callback_registry = CALLBACK_REGISTRY
        self.initialized = False

        # Initialize all pipeline components
        if self.callback_name is not None:
            self.init_all()

    def init_all(self):
        """Initialize all pipeline components in the correct order"""
        self.init_model()
        self.init_source()
        self.init_preprocess()
        self.init_postprocess()
        self.init_visualization()
        self.init_callback()
        self.initialized = True

    def init_model(self, model_file: Optional[Any] = None):
        """
        Initialize the inference model based on the backend type.

        Args:
            model_file: Optional override for model file path
        """
        if self.infer_type == "hailo":
            self.infer = HailoInference(self.config.model)
        elif self.infer_type == "onnx":
            self.infer = ONNXInference(self.config.model)

    def init_source(self, source: Optional[Any] = None):
        """
        Initialize the data source component.

        Args:
            source: Optional override for source configuration
        """
        self.source = self.callback_registry.getSource(self.callback_name)(
            self.config.source
        )

    def init_preprocess(self, preprocess: Optional[Any] = None):
        """
        Initialize the preprocessing component.

        Args:
            preprocess: Optional override for preprocessing configuration
        """
        self.preprocess = self.callback_registry.getPreProcessor(self.callback_name)(
            self.config.preprocess
        )

    def init_postprocess(self, postprocess: Optional[Any] = None):
        """
        Initialize the postprocessing component.

        Args:
            postprocess: Optional override for postprocessing configuration
        """
        print(
            f"postprocess: {self.callback_name},{self.callback_registry.getPostProcessor(self.callback_name).__name__}"
        )
        self.postprocess = self.callback_registry.getPostProcessor(self.callback_name)(
            self.config.postprocess
        )

    def init_visualization(self, visualization: Optional[Any] = None):
        """
        Initialize the visualization component based on task type.

        Args:
            visualization: Optional override for visualization configuration
        """
        self.visualization = self.callback_registry.getVisualizer(self.callback_name)(
            self.config.visualization
        )

    def init_callback(self, callback: Optional[Any] = None):
        """
        Initialize the collaborative inference callback.

        Args:
            callback: Optional override for callback configuration
        """
        self.callback = self.infer.add_callback(
            self.callback_registry.getCollatInfer(self.callback_name)
        )

    def setting_preprocess(self, task_name: str) -> None:
        """
        Setting the preprocess function for the inference engine.
        """
        self.preprocess = self.callback_registry.getPreProcessor(task_name)(
            self.config.preprocess
        )

    def setting_postprocess(self, task_name: str) -> None:
        """
        Setting the postprocess function for the inference engine.
        """
        self.postprocess = self.callback_registry.getPostProcessor(task_name)(
            self.config.postprocess
        )

    def setting_visualization(self, task_name: str) -> None:
        """
        Setting the visualization function for the inference engine.
        """
        self.visualization = self.callback_registry.getVisualizer(task_name)(
            self.config.visualization
        )

    def setting_callback(self, task_name: str) -> None:
        """
        Setting the callback function for the inference engine.
        """
        self.callback = self.infer.add_callback(
            self.callback_registry.getCollateInfer(task_name)
        )

    @classmethod
    def load_from_config(cls, config):
        """
        Factory method to create InferenceEngine from configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters

        Returns:
            InferenceEngine instance initialized with the provided configuration
        """
        return cls(**config)

    def run(self):
        """
        Execute the complete inference pipeline.

        Processes frames from the source through the entire pipeline:
        1. Frame acquisition from source
        2. Preprocessing
        3. Model inference
        4. Postprocessing
        5. Visualization (if enabled)
        6. Output saving (if configured)
        """
        if not self.initialized:
            self.init_all()

        with self.source as source:
            for frame_idx, frame in enumerate(source):
                logger.debug(f"Processing frame {frame_idx}, shape: {frame.shape}")

                # Store original frame for visualization
                original_frame = frame.copy()

                # Preprocess frame
                preprocessed_frame = self.preprocess(frame)
                logger.debug(f"Preprocessed frame shape: {preprocessed_frame.shape}")

                # Run inference
                results = self.infer.as_process_inference(preprocessed_frame)

                # Postprocess results
                post_results = self.postprocess(
                    results, original_shape=original_frame.shape[:2]
                )

                # Visualize results if enabled
                if self.config.show:
                    vis_image = self.visualization(original_frame, post_results)

                    # Save visualization if configured
                    if self.config.save:
                        if self.config.save_path is None:
                            output_path = "output"
                        else:
                            output_path = self.config.save_path
                        os.makedirs(osp.dirname(output_path), exist_ok=True)
                        output_path = osp.join(
                            output_path, f"output_frame_{frame_idx:04d}.jpg"
                        )
                        success = self.visualization.save(vis_image, output_path)
                        if success:
                            logger.debug(f"Saved visualization to {output_path}")

                    # Display visualization
                    self.visualization.show(vis_image, f"show")

    def _call_callback(self, callback_type: CallbackType, frame: np.ndarray):
        """
        Internal method to call registered callbacks.

        Args:
            callback_type: Type of callback to invoke
            frame: Frame data to pass to the callback
        """
        if callback_type in self.callback_registry:
            self.callback_registry[callback_type](frame)


if __name__ == "__main__":
    # Demonstrate the enhanced registry functionality
    print("Testing callback registry with multiple names:")

    # Test getting callback by different names
    print(f"yolov8det preprocessor: {CALLBACK_REGISTRY.getPreProcessor('yolov8det')}")
    print(f"yolov8seg preprocessor: {CALLBACK_REGISTRY.getPreProcessor('yolov8seg')}")

    # Test that both names point to the same function
    callback1 = CALLBACK_REGISTRY.getPreProcessor("yolov8det")
    callback2 = CALLBACK_REGISTRY.getPostProcessor("yolov8seg")
    print(f"Same callback object: {callback1 is callback2}")

    # List all shared names
    shared_names = CALLBACK_REGISTRY.get_shared_names(
        "yolov8det", CallbackType.PRE_PROCESSOR
    )
    print(f"Shared names for yolov8det: {shared_names}")
