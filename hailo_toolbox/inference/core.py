from typing import Any, List, Optional, AnyStr, Union, Callable, Dict, Type, Iterable
from enum import Enum
import logging
import os
import os.path as osp
from multiprocessing import Process, Queue

import numpy as np
from hailo_toolbox.inference.hailo_engine import HailoInference
from hailo_toolbox.inference.onnx_engine import ONNXInference
from hailo_toolbox.utils.config import Config
from hailo_toolbox.utils.sharememory import ShareMemoryManager
from hailo_toolbox.utils.timer import Timer


logger = logging.getLogger(__name__)


class CallbackType(Enum):
    """Enumeration of callback types supported by the registry"""

    PRE_PROCESSOR = "pre_processor"
    POST_PROCESSOR = "post_processor"
    VISUALIZER = "visualizer"
    SOURCE = "source"
    COLLATE_INFER = "collate_infer"


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
        self.CollateInfer: Dict[str, Callable] = {}

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
        elif callback_type == CallbackType.COLLATE_INFER:
            self.CollateInfer[name] = callback

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
                    callback_type == CallbackType.COLLATE_INFER
                    and name in self.CollateInfer
                ):
                    del self.CollateInfer[name]

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
            def collate_infer_func(data):
                return results

            # Multiple names
            @CALLBACK_REGISTRY.registryCollateInfer("yolov8det", "yolov8seg")
            def yolo_collate_infer_func(data):
                return results
        """
        if not names:
            raise ValueError("At least one name must be provided")
        return self.register(list(names), CallbackType.COLLATE_INFER)

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

    def getCollateInfer(self, name: str) -> Callable:
        """Legacy method - use get_callback instead"""
        return self.get_callback(name, CallbackType.COLLATE_INFER)


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
    - Server mode for process-based inference with shared memory

    Args:
        model: Path to the model file (.hef for Hailo, .onnx for ONNX)
        task_name: Name identifier for callback registration lookup
        command: Command type (e.g., "infer")
        callback: Callback identifier (default: "yolov8det")
        convert: Whether to convert model format (default: False)
        infer: Whether to perform inference (default: False)
        source: Data source configuration
        output: Output configuration
        preprocess: Preprocessing configuration
        postprocess: Postprocessing configuration
        visualization: Visualization configuration
        task_type: Type of task (e.g., "detection", "segmentation")
        save: Whether to save output (default: False)
        save_path: Path to save output files
        show: Whether to display visualization (default: False)

    Example (New Style - Direct Parameters):
        engine = InferenceEngine(
            model="model.hef",
            source="video.mp4",
            task_type="detection",
            task_name="yolov8det"
        )
        engine.run()

    Example (Legacy Style - Config Object):
        config = Config()
        config.model = "model.hef"
        config.source = "video.mp4"
        config.task_type = "detection"
        engine = InferenceEngine(config, "yolov8det")
        engine.run()

    Example (Server Mode):
        import queue
        import threading
        import numpy as np

        # Initialize engine (new style)
        engine = InferenceEngine(
            model="model.hef",
            task_type="detection",
            task_name="yolov8det"
        )

        # Create queues
        input_queue = queue.Queue()
        output_queue = queue.Queue()

        # Start server in separate thread
        server_thread = threading.Thread(
            target=engine.as_server_inference,
            args=(input_queue, output_queue),
            kwargs={'enable_visualization': True, 'server_timeout': 2.0}
        )
        server_thread.start()

        # Process frames
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Method 1: Send shared memory name directly
        shm_name = InferenceEngine.create_frame_shm(frame, {'frame_id': 0})
        input_queue.put(shm_name)

        # Method 2: Send metadata dictionary
        input_queue.put({
            'shm_name': shm_name,
            'frame_id': 0,
            'timestamp': time.time(),
            'source': 'camera_1'
        })

        # Get results
        try:
            result = output_queue.get(timeout=5.0)
            result_frame = InferenceEngine.load_frame_from_shm(result['shm_name'])
            inference_results = result['inference_results']

            # Clean up shared memory
            InferenceEngine.cleanup_shm(result['shm_name'])

        except queue.Empty:
            print("No result received within timeout")

        # Shutdown server
        input_queue.put("SHUTDOWN")
        server_thread.join()

    Server Mode Features:
        - Shared memory based frame exchange for high performance
        - Automatic batch processing for improved throughput
        - Optional visualization rendering
        - Graceful shutdown handling
        - Comprehensive error handling and logging
        - Support for frame metadata and tracking
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        task_name: AnyStr = "yolov8det",
        model: Optional[AnyStr] = None,
        command: Optional[AnyStr] = None,
        convert: bool = False,
        infer: bool = False,
        source: Any = None,
        output: Any = None,
        preprocess_config: Any = None,
        postprocess_config: Any = None,
        visualization_config: Any = None,
        task_type: Any = None,
        save: bool = False,
        save_dir: Any = None,
        show: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the inference engine with configuration parameters.

        Args:
            config: Configuration object (for backward compatibility)
            task_name: Name identifier for callback registration lookup
            model: Path to the model file (.hef for Hailo, .onnx for ONNX)
            command: Command type (e.g., "infer")
            callback: Callback identifier (default: "yolov8det")
            convert: Whether to convert model format (default: False)
            infer: Whether to perform inference (default: False)
            source: Data source configuration
            output: Output configuration
            preprocess_config: Preprocessing configuration
            postprocess_config: Postprocessing configuration
            visualization_config: Visualization configuration
            task_type: Type of task (e.g., "detection", "segmentation")
            save: Whether to save output (default: False)
            save_dir: Path to save output files
            show: Whether to display visualization (default: False)
            **kwargs: Additional keyword arguments
        """
        # Handle backward compatibility with Config object
        if config is not None:
            # Use Config object values, but allow individual parameters to override
            self.model = model if model is not None else getattr(config, "model", None)
            self.command = (
                command if command is not None else getattr(config, "command", None)
            )
            self.task_name = (
                task_name
                if task_name is not None
                else getattr(config, "task_name", "yolov8det")
            )
            self.convert = convert if convert else getattr(config, "convert", False)
            self.infer = infer if infer else getattr(config, "infer", False)
            self.source = (
                source if source is not None else getattr(config, "source", None)
            )
            self.output = (
                output if output is not None else getattr(config, "output", None)
            )
            self.preprocess_config = (
                preprocess_config
                if preprocess_config is not None
                else getattr(config, "preprocess_config", None)
            )
            self.postprocess_config = (
                postprocess_config
                if postprocess_config is not None
                else getattr(config, "postprocess_config", None)
            )
            self.visualization_config = (
                visualization_config
                if visualization_config is not None
                else getattr(config, "visualization_config", None)
            )
            self.task_type = (
                task_type
                if task_type is not None
                else getattr(config, "task_type", None)
            )
            self.save = save if save else getattr(config, "save", False)
            self.save_dir = (
                save_dir if save_dir is not None else getattr(config, "save_dir", None)
            )
            self.show = show if show else getattr(config, "show", False)
        else:
            # Use individual parameters directly
            self.model = model
            self.command = command
            self.convert = convert
            self.infer = infer
            self.source = source
            self.output = output
            self.preprocess_config = preprocess_config
            self.postprocess_config = postprocess_config
            self.visualization_config = visualization_config
            self.task_type = task_type
            self.save = save
            self.save_dir = save_dir
            self.show = show

        # Set up task-specific callback names
        self.task_name = task_name
        if task_name is None:
            self.preprocess_name = None
            self.postprocess_name = None
            self.visualization_name = None
            self.collate_infer_name = None
        else:
            self.preprocess_name = task_name
            self.postprocess_name = task_name
            self.visualization_name = task_name
            self.collate_infer_name = task_name

        # Determine inference mode and backend type
        if self.command == "infer":
            self.infer = True
            if self.model and self.model.endswith(".hef"):
                self.infer_type = "hailo"
            elif self.model and self.model.endswith(".onnx"):
                self.infer_type = "onnx"
            else:
                raise ValueError(f"Unsupported model type: {self.model}")
        else:
            self.infer = False

        self.callback_registry = CALLBACK_REGISTRY
        self.initialized = False

        # Initialize all pipeline components
        # if self.task_name is not None:
        #     self.init_all()

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
        model_path = model_file if model_file else self.model

        if self.infer_type == "hailo":
            self.infer = HailoInference(model_path)
        elif self.infer_type == "onnx":
            self.infer = ONNXInference(model_path)

    def init_source(self, source: Optional[Any] = None):
        """
        Initialize the data source component.

        Args:
            source: Optional override for source configuration
        """
        self.source = self.callback_registry.getSource(self.task_name)(self.source)

    def init_preprocess(self, preprocess: Optional[Any] = None):
        """
        Initialize the preprocessing component.

        Args:
            preprocess: Optional override for preprocessing configuration
        """
        self.preprocess = self.callback_registry.getPreProcessor(self.task_name)(
            self.preprocess_config
        )

    def init_postprocess(self, postprocess: Optional[Any] = None):
        """
        Initialize the postprocessing component.

        Args:
            postprocess: Optional override for postprocessing configuration
        """
        print(
            f"postprocess: {self.task_name},{self.callback_registry.getPostProcessor(self.task_name).__name__}"
        )
        self.postprocess = self.callback_registry.getPostProcessor(self.task_name)(
            self.postprocess_config
        )

    def init_visualization(self, visualization: Optional[Any] = None):
        """
        Initialize the visualization component based on task type.

        Args:
            visualization: Optional override for visualization configuration
        """
        self.visualization = self.callback_registry.getVisualizer(self.task_name)(
            self.visualization_config
        )

    def init_callback(self, callback: Optional[Any] = None):
        """
        Initialize the collaborative inference callback.

        Args:
            callback: Optional override for callback configuration
        """
        self.callback = self.infer.add_callback(
            self.callback_registry.getCollateInfer(self.task_name)
        )

    def setting_preprocess(self, task_name: str) -> None:
        """
        Setting the preprocess function for the inference engine.
        """
        self.preprocess = self.callback_registry.getPreProcessor(task_name)(
            self.preprocess_config
        )

    def setting_postprocess(self, task_name: str) -> None:
        """
        Setting the postprocess function for the inference engine.
        """
        self.postprocess = self.callback_registry.getPostProcessor(task_name)(
            self.postprocess_config
        )

    def setting_visualization(self, task_name: str) -> None:
        """
        Setting the visualization function for the inference engine.
        """
        self.visualization = self.callback_registry.getVisualizer(task_name)(
            self.visualization_config
        )

    def setting_callback(self, task_name: str) -> None:
        """
        Setting the callback function for the inference engine.
        """
        self.callback = self.infer.add_callback(
            self.callback_registry.getCollateInfer(task_name)
        )

    @classmethod
    def load_from_config(cls, config_dict):
        """
        Factory method to create InferenceEngine from configuration dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            InferenceEngine instance initialized with the provided configuration
        """
        return cls(**config_dict)

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
                if self.show:
                    vis_image = self.visualization(original_frame, post_results)

                    # Save visualization if configured
                    if self.save:
                        if self.save_dir is None:
                            output_path = "output"
                        else:
                            output_path = self.save_dir
                        os.makedirs(osp.dirname(output_path), exist_ok=True)
                        output_path = osp.join(
                            output_path, f"output_frame_{frame_idx:04d}.jpg"
                        )
                        success = self.visualization.save(vis_image, output_path)
                        if success:
                            logger.debug(f"Saved visualization to {output_path}")

                    # Display visualization
                    self.visualization.show(vis_image, f"show")

    def as_server_inference(
        self,
        input_queue: Queue,
        output_queue: Queue,
        enable_visualization: bool = False,
        server_timeout: Optional[float] = None,
        max_batch_size: int = 1,
    ):
        """
        Execute the complete inference pipeline as a server process.

        This method runs as a server process that continuously processes frames
        from shared memory. It receives shared memory names through a queue,
        processes the frames, and puts results back to output queue.

        Args:
            input_queue: Queue containing shared memory names for input frames
            output_queue: Queue to put processed results (shared memory names)
            enable_visualization: Whether to draw visualization on results
            server_timeout: Timeout for queue operations in seconds
            max_batch_size: Maximum batch size for processing multiple frames

        Features:
        - Continuous processing loop for server operation
        - Shared memory based frame exchange for efficiency
        - Optional visualization rendering
        - Batch processing support for improved throughput
        - Graceful shutdown handling
        - Comprehensive error handling and logging
        """
        if not self.initialized:
            self.init_all()

        logger.info("Starting inference server process")
        logger.info(
            f"Server configuration: visualization={enable_visualization}, timeout={server_timeout}s"
        )

        frame_count = 0
        batch_frames = []
        batch_metadata = []

        try:
            while True:
                # try:
                # Get input frame metadata from queue with timeout
                frame_metadata = input_queue.get(timeout=server_timeout)

                # Handle shutdown signal
                if frame_metadata is None or frame_metadata == "SHUTDOWN":
                    logger.info("Received shutdown signal, stopping server")
                    break

                frame = self.shm_manager.read(**frame_metadata)
                # Add to batch
                batch_frames.append(frame)

                # Process batch when it's full or timeout occurs
                if len(batch_frames) >= max_batch_size:
                    with Timer("process_frame_batch"):
                        self._process_frame_batch(
                            batch_frames, output_queue, enable_visualization
                        )
                    batch_frames.clear()

                frame_count += 1

            # except Queue.Empty:
            #     # Timeout occurred, process any remaining frames in batch
            #     if batch_frames:
            #         logger.debug(f"Processing remaining batch of {len(batch_frames)} frames")
            #         self._process_frame_batch(
            #             batch_frames,
            #             output_queue,
            #             enable_visualization
            #         )
            #         batch_frames.clear()
            #         batch_metadata.clear()
            #     continue

            # except Exception as e:
            #     logger.error(f"Error processing frame: {e}")
            #     continue

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in server process: {e}")
        finally:
            # Process any remaining frames
            if batch_frames:
                logger.info(f"Processing final batch of {len(batch_frames)} frames")
                self._process_frame_batch(
                    batch_frames, output_queue, enable_visualization
                )

            logger.info(
                f"Inference server stopped. Total frames processed: {frame_count}"
            )

    def _process_frame_batch(
        self, frames: List[np.ndarray], output_queue: Queue, enable_visualization: bool
    ):
        """
        Process a batch of frames through the inference pipeline.

        Args:
            frames: List of frames to process
            output_queue: Queue to put results
            enable_visualization: Whether to render visualization
        """
        for frame in frames:
            # try:
            # Store original frame for visualization
            with Timer("copy_frame"):
                original_frame = frame.copy()

            # Preprocess frame
            with Timer("preprocess_frame"):
                preprocessed_frame = self.preprocess(frame)

            # Run inference
            with Timer("inference_frame"):
                results = self.infer.as_process_inference(preprocessed_frame)

            # Postprocess results
            with Timer("postprocess_frame"):
                post_results = self.postprocess(
                    results, original_shape=original_frame.shape[:2]
                )

            # Prepare result frame
            if enable_visualization:
                # Render visualization on frame
                with Timer("visualization_frame"):
                    result_frame = self.visualization(original_frame, post_results)
            else:
                # Return original frame without visualization
                result_frame = original_frame

            # Save result to shared memory
            with Timer("write_result_frame"):
                result_shm_info = self.shm_manager.write(result_frame, "result_frame")

            if result_shm_info:
                # Put result to output queue
                with Timer("put_result_frame"):
                    output_queue.put(result_shm_info)
                # logger.debug(f"Processed frame {frame_metadata.get('name', 'unknown')}")
            else:
                logger.error(
                    f"Failed to save result for frame {result_shm_info.get('name', 'unknown')}"
                )
        # except Exception as e:
        #     print(f"Error processing frame: {e}")
        #     # logger.error(f"Error processing frame {frame_metadata.get('name', 'unknown')}: {e}")
        #     continue

    def start_server(
        self,
        input_queue: Optional[Queue] = None,
        output_queue: Optional[Queue] = None,
        enable_visualization: bool = False,
        queue_size: int = 30,
        server_timeout: Optional[float] = None,
        max_batch_size: int = 1,
    ) -> tuple[Queue, Queue]:
        """
        Convenient method to start the inference server with automatic queue creation.

        Args:
            input_queue: Optional input queue (created if None)
            output_queue: Optional output queue (created if None)
            enable_visualization: Whether to enable visualization
            queue_size: Size of the input and output queues
            server_timeout: Server timeout in seconds
            max_batch_size: Maximum batch size for processing

        Returns:
            Tuple of (input_queue, output_queue) for external use
        """
        if input_queue is None:
            input_queue = Queue(maxsize=queue_size)  # Prevent excessive memory usage

        if output_queue is None:
            output_queue = Queue(maxsize=queue_size)

        self.shm_manager = ShareMemoryManager()

        # Start server in current thread (or can be modified to use threading/multiprocessing)
        server_process = Process(
            target=self.as_server_inference,
            args=(
                input_queue,
                output_queue,
                enable_visualization,
                server_timeout,
                max_batch_size,
            ),
        )
        server_process.start()

        return input_queue, output_queue

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
    # print("Testing callback registry with multiple names:")

    # # Test getting callback by different names
    # print(f"yolov8det preprocessor: {CALLBACK_REGISTRY.getPreProcessor('yolov8det')}")
    # print(f"yolov8seg preprocessor: {CALLBACK_REGISTRY.getPreProcessor('yolov8seg')}")

    # # Test that both names point to the same function
    # callback1 = CALLBACK_REGISTRY.getPreProcessor("yolov8det")
    # callback2 = CALLBACK_REGISTRY.getPostProcessor("yolov8seg")
    # print(f"Same callback object: {callback1 is callback2}")

    # # List all shared names
    # shared_names = CALLBACK_REGISTRY.get_shared_names(
    #     "yolov8det", CallbackType.PRE_PROCESSOR
    # )
    # print(f"Shared names for yolov8det: {shared_names}")

    # # Test server inference functionality
    # print("\n" + "="*50)
    # print("Testing server inference functionality:")
    # print("="*50)

    # Example usage with new constructor
    infer = InferenceEngine(
        model="yolov8n.onnx", task_name="yolov8det", command="infer"
    )
