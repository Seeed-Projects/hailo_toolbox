"""
Base class for model converters to ONNX format.
"""

from abc import ABC, abstractmethod
import os
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

import numpy as np
import onnx

try:
    import tensorflow as tf
except ImportError:
    tf = None
from ..utils import get_logger

from ..utils.excep import FileFormatException

from contextlib import suppress
from tarfile import ReadError

from hailo_sdk_client import ClientRunner
from hailo_sdk_common.hailo_nn.hailo_nn import HailoNN
from hailo_sdk_client.hailo_archive.hailo_archive import HailoArchiveLoader
from hailo_toolbox.converters.utils import load_calibration_images

# create logger
logger = get_logger(__name__)

MODEL_EXTENSIONS = (".onnx", ".tflite")


class BaseConverter(ABC):
    """
    Abstract base class for all model converters.

    This class defines the common interface that all model converters must implement.
    Each converter is responsible for converting models from a specific framework to ONNX.
    """

    def __init__(
        self,
        model_path: str,
        hw_arch: str,
        model_script: Optional[Union[str, Path]] = None,
        calibration_dataset_size: int = 100,
        **kwargs,
    ):
        """
        Initialize the base converter.

        Args:
            model_path: Path to the source model (as string).
            hw_arch: Hardware architecture to use.
            model_script: Model script content or path to script file.
            calibration_dataset_size: Size of calibration dataset.
            **kwargs: Additional framework-specific parameters.
        """
        self.model_path = str(model_path)  # Ensure string type
        self.hw_arch = hw_arch
        self.calibration_dataset_size = calibration_dataset_size

        # Determine framework based on file extension
        if os.path.isfile(self.model_path):
            _, ext = os.path.splitext(self.model_path)
            if ext not in MODEL_EXTENSIONS:
                raise ValueError(f"Unsupported model extension: {ext}")

        self.framework = "onnx" if self.model_path.endswith(".onnx") else "tflite"

        # Load model script
        self.load_model_script(model_script)

        # Initialize file paths as strings
        self._onnx_file = None
        self._har_file = None
        self._hn_file = None
        self._tf_file = None

        # Initialize file paths based on model path
        self._setup_file_paths()

        logger.debug(
            "init converter: framework=%s, hw_arch=%s", self.framework, hw_arch
        )
        self._validate_and_prepare()

        self.runner = ClientRunner(hw_arch=hw_arch)

    def _setup_file_paths(self) -> None:
        """
        Setup all file paths based on the model path.
        This method provides a unified way to manage file paths after knowing the model location.
        """
        if os.path.isdir(self.model_path):
            # Handle directory input (e.g., TensorFlow SavedModel)
            base_dir = self.model_path
            base_name = os.path.basename(self.model_path.rstrip(os.sep))
        else:
            # Handle file input
            base_dir = os.path.dirname(self.model_path)
            base_name = os.path.splitext(os.path.basename(self.model_path))[0]

        # Setup all possible file paths
        self._onnx_file = os.path.join(base_dir, f"{base_name}.onnx")
        self._har_file = os.path.join(base_dir, f"{base_name}.har")
        self._hn_file = os.path.join(base_dir, f"{base_name}.hn")

        if os.path.isdir(self.model_path):
            # For TensorFlow SavedModel directory
            self._tf_file = os.path.join(self.model_path, "saved_model.pb")
        else:
            # For regular files
            self._tf_file = os.path.join(base_dir, f"{base_name}.tflite")

        logger.debug("File paths setup completed:")
        logger.debug("  ONNX file: %s", self._onnx_file)
        logger.debug("  HAR file: %s", self._har_file)
        logger.debug("  HN file: %s", self._hn_file)
        logger.debug("  TF file: %s", self._tf_file)

    def set_file_paths(self, model_dir_or_file: str) -> None:
        """
        Set all file paths based on a model directory or file.
        This provides a convenient way to reset all paths when the model location changes.

        Args:
            model_dir_or_file: Path to model directory or file (as string).
        """
        self.model_path = str(model_dir_or_file)
        self._setup_file_paths()
        logger.info("File paths updated based on: %s", self.model_path)

    def parser_model_info(self):
        """Parse model information based on framework type."""
        if self.framework == "onnx":
            self.model_info = self.get_onnx_info()
        elif self.framework == "tflite":
            self.model_info = self.get_tflite_info()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def get_tflite_info(self) -> dict:
        """
        Retrieve information about the TensorFlow Lite model's input and output tensors.

        This function loads a TensorFlow Lite model file, initializes the interpreter,
        and extracts the shapes and names of the input and output tensors from the model.

        Returns
        -------
        dict
            A dictionary containing:
            - "inputs_shape": List of tuples representing the shapes of input tensors.
            - "outputs_shape": List of tuples representing the shapes of output tensors.
            - "start_nodes_name": List of names of the input tensors.
            - "end_nodes_name": List of names of the output tensors.

        Raises
        ------
        ImportError
            If TensorFlow is not installed.
        FileNotFoundError
            If the TensorFlow Lite model file is not found.
        ValueError
            If the model file is not a valid TensorFlow Lite model.
        """
        if tf is None:
            raise ImportError(
                "TensorFlow is required for TensorFlow Lite model parsing. "
                "Please install tensorflow: pip install tensorflow"
            )

        logger.info("checking tflite model...")

        # Check if model file exists
        tflite_file = (
            self.tf_file
            if self.tf_file.endswith(".tflite")
            else f"{os.path.splitext(self.tf_file)[0]}.tflite"
        )
        if not os.path.exists(tflite_file):
            raise FileNotFoundError(
                f"TensorFlow Lite model file not found: {tflite_file}"
            )

        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path=tflite_file)
        interpreter.allocate_tensors()
        logger.info("tflite model loaded successfully")

        # Get input tensor details
        input_details = interpreter.get_input_details()
        inputs_shape = []
        start_nodes_name = []

        for input_detail in input_details:
            # Convert numpy array shape to tuple
            shape = tuple(input_detail["shape"].tolist())
            inputs_shape.append(shape)
            start_nodes_name.append(input_detail["name"])

        # Get output tensor details
        output_details = interpreter.get_output_details()
        outputs_shape = []
        end_nodes_name = []

        # Check if end_node is specified (similar to ONNX implementation)
        if hasattr(self, "end_node") and self.end_node is not None:
            # Filter outputs based on specified end_node
            end_nodes_name = [self.end_node]
            for output_detail in output_details:
                if output_detail["name"] == self.end_node:
                    shape = tuple(output_detail["shape"].tolist())
                    outputs_shape.append(shape)
                    break
            else:
                logger.warning(
                    f"Specified end_node '{self.end_node}' not found in model outputs"
                )
                # Fallback to all outputs if specified end_node is not found
                for output_detail in output_details:
                    shape = tuple(output_detail["shape"].tolist())
                    outputs_shape.append(shape)
                    end_nodes_name.append(output_detail["name"])
        else:
            # Use all output tensors
            for output_detail in output_details:
                shape = tuple(output_detail["shape"].tolist())
                outputs_shape.append(shape)
                end_nodes_name.append(output_detail["name"])

        logger.info(
            f"tflite model parsed - inputs: {len(inputs_shape)}, outputs: {len(outputs_shape)}"
        )

        return {
            "inputs_shape": inputs_shape,
            "outputs_shape": outputs_shape,
            "start_nodes_name": start_nodes_name,
            "end_nodes_name": end_nodes_name,
        }

    def get_onnx_info(self) -> dict:
        """
        Retrieve information about the ONNX model's input and output nodes.

        This function loads an ONNX model file, verifies its correctness, and extracts
        the shapes and names of the input and output nodes from the model's graph.

        Returns
        -------
        dict
            A dictionary containing:
            - "inputs_shape": List of tuples representing the shapes of input nodes.
            - "outputs_shape": List of tuples representing the shapes of output nodes.
            - "start_nodes_name": List of names of the input nodes.
            - "end_nodes_name": List of names of the output nodes.
        """

        logger.info("checking onnx model...")
        onnx_model = onnx.load(self.onnx_file)
        onnx.checker.check_model(onnx_model)
        logger.info("onnx model is valid")
        inputs = []
        start_nodes_name = []
        for i in onnx_model.graph.input:
            inputs.append(
                tuple(map(lambda x: int(x.dim_value), i.type.tensor_type.shape.dim))
            )
            start_nodes_name.append(i.name)

        if getattr(self, "end_node", None) is None:
            outputs = []
            end_nodes_name = []
            for i in onnx_model.graph.output:
                outputs.append(
                    tuple(map(lambda x: int(x.dim_value), i.type.tensor_type.shape.dim))
                )
                end_nodes_name.append(i.name)
        else:
            outputs = []
            end_nodes_name = [self.end_node]
            for i in onnx_model.graph.node:
                if i.name == self.end_node:
                    outputs.append(
                        tuple(
                            map(
                                lambda x: int(x.dim_value),
                                i.output[0].type.tensor_type.shape.dim,
                            )
                        )
                    )

        return {
            "inputs_shape": inputs,
            "outputs_shape": outputs,
            "start_nodes_name": start_nodes_name,
            "end_nodes_name": end_nodes_name,
        }

    def load_calibration_images(
        self,
        image_dir: Optional[str] = None,
        image_shape: Tuple[int, int, int, int] = (100, 640, 640, 3),
        data_type: str = "uint8",
    ) -> np.ndarray:
        """
        Load calibration images for model optimization.

        Args:
            image_dir: Directory containing calibration images.
            image_shape: Shape of calibration images.
            data_type: Data type for calibration images.

        Returns:
            Loaded calibration dataset.
        """
        if image_dir is not None:
            self.calibrat_datasets = load_calibration_images(
                image_dir, self.calibration_dataset_size, image_shape, data_type
            )
        else:
            logger.info(
                "Generating random calibration data with shape: %s", image_shape
            )
            self.calibrat_datasets = np.random.randn(
                *(
                    [
                        self.calibration_dataset_size,
                    ]
                    + list(image_shape)
                )
            ).astype(data_type)

    def _validate_and_prepare(self):
        """
        Validate parameters and prepare for conversion.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model file not found: {self.model_path}")

    def load_model_script(self, model_script: Optional[Union[str, Path]]) -> None:
        """
        Load model script from string or file path.

        Args:
            model_script: Script content as string, or path to script file.
        """
        if model_script is None:
            self.model_script = "normalization1 = normalization([0,0,0],[255,255,255])"
            return

        model_script_str = str(model_script)  # Convert Path to string if needed

        if os.path.exists(model_script_str):
            # Load from file
            with open(model_script_str, "r", encoding="utf-8") as f:
                self.model_script = f.read()
            logger.info("Model script loaded from file: %s", model_script_str)
        else:
            # Use as direct script content
            self.model_script = model_script_str
            logger.info("Model script loaded as direct content")

    # Properties for file paths (all return strings now)
    @property
    def onnx_file(self) -> str:
        """Get path to ONNX file as string."""
        return self._onnx_file

    def set_onnx_file(self, onnx_file: str) -> None:
        """
        Set ONNX file path.

        Args:
            onnx_file: Path to ONNX file as string.
        """
        onnx_file_str = str(onnx_file)  # Ensure string type
        if not onnx_file_str.endswith(".onnx"):
            raise ValueError("The input file must be in ONNX format.")
        self._onnx_file = onnx_file_str

    @property
    def har_file(self) -> str:
        """Get path to HAR file as string."""
        return self._har_file

    def set_har_file(self, har_file: str) -> None:
        """
        Set HAR file path.

        Args:
            har_file: Path to HAR file as string.
        """
        har_file_str = str(har_file)  # Ensure string type
        if not har_file_str.endswith(".har"):
            raise ValueError("The input file must be in HAR format.")
        self._har_file = har_file_str

    @property
    def hn_file(self) -> str:
        """Get path to HN file as string."""
        return self._hn_file

    def set_hn_file(self, hn_file: str) -> None:
        """
        Set HN file path.

        Args:
            hn_file: Path to HN file as string.
        """
        hn_file_str = str(hn_file)  # Ensure string type
        if not hn_file_str.endswith(".hn"):
            raise ValueError("The input file must be in HN format.")
        self._hn_file = hn_file_str

    @property
    def tf_file(self) -> str:
        """Get path to TensorFlow file as string."""
        return self._tf_file

    def set_tf_file(self, tf_file: str) -> None:
        """
        Set TensorFlow file path.

        Args:
            tf_file: Path to TensorFlow file or directory as string.
        """
        tf_file_str = str(tf_file)  # Ensure string type
        if os.path.isdir(tf_file_str):
            self._tf_file = os.path.join(tf_file_str, "saved_model.pb")
        else:
            self._tf_file = tf_file_str

    @property
    def hef_file(self) -> str:
        """
        Property to get the path to the HEF file, which is the ONNX file with the extension replaced with ".hef".

        Returns
        -------
        str
            Path to the HEF file
        """
        return os.path.splitext(self.onnx_file)[0] + ".hef"

    @property
    def hailoonnx_file(self) -> str:
        """
        Property to get the path to the modified ONNX file, which is the ONNX file with "_hailo" added to the filename before the extension.

        Returns
        -------
        str
            Path to the modified ONNX file
        """
        base_name = os.path.splitext(self.onnx_file)[0]
        return f"{base_name}_hailo.onnx"

    @property
    def start_node_names(self) -> List[str]:
        """Get start node names from model info."""
        return getattr(self, "_start_node_names", [])

    @property
    def end_node_names(self) -> List[str]:
        """Get end node names from model info."""
        return getattr(self, "_end_node_names", [])

    @property
    def net_input_shapes(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Get network input shapes from model info."""
        return getattr(self, "_net_input_shapes", {})

    def _fix_har(self):
        """Fix HAR file format issues."""
        if self.har_file.endswith(".har"):
            with suppress(ReadError), HailoArchiveLoader(self.har_file) as har_loader:
                hn = json.loads(har_loader.get_hn())
                hn.pop("direct_control", None)
                return hn

    def _fix_hn(self):
        """Fix HN file format issues."""
        if self.hn_file.endswith(".hn"):
            with suppress(json.JSONDecodeError), open(
                self.hn_file, encoding="utf-8"
            ) as hn_file:
                return json.load(hn_file)

        raise FileFormatException("The given model must be a valid HAR file")

    def _get_hailo_nn(self) -> HailoNN:
        """Get HailoNN instance from HN file."""
        hn = self._fix_hn()
        return HailoNN.from_parsed_hn(hn)

    @abstractmethod
    def convert(
        self, model_path: str, output_name: Optional[str] = None, **kwargs
    ) -> str:
        """
        Convert a model to ONNX format.

        Args:
            model_path: Path to the source model.
            output_name: Optional name for the output ONNX file.
            **kwargs: Additional conversion parameters.

        Returns:
            Path to the converted ONNX model.
        """
        pass

    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load a model from the source framework.

        Args:
            model_path: Path to the model.
            **kwargs: Additional loading parameters.

        Returns:
            Loaded model object.
        """
        pass

    def optimize_onnx(self, onnx_path: str) -> str:
        """
        Optimize the ONNX model after conversion.

        Args:
            onnx_path: Path to the ONNX model.

        Returns:
            Path to the optimized ONNX model.
        """
        # Default implementation just returns the original path
        # Subclasses can override this to implement optimization

        logger.debug("use base optimization: %s", onnx_path)
        return onnx_path

    def validate_onnx(self, onnx_path: str) -> bool:
        """
        Validate the converted ONNX model.

        Args:
            onnx_path: Path to the ONNX model.

        Returns:
            True if the model is valid, False otherwise.
        """
        import onnx

        try:
            logger.info("validate ONNX model: %s", onnx_path)
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX model validated")
            return True
        except Exception as e:
            logger.error("ONNX model validation failed: %s", e)
            return False

    def save_model(self, model, file_path: str):
        """
        Save the model to a file.

        Parameters
        ----------
        model: bytes
            Hailo model in bytes
        file_path: str
            Path to the file to save the model
        """
        file_path_str = str(file_path)  # Ensure string type
        with open(file_path_str, "wb") as f:
            f.write(model)
        logger.info("Model saved to: %s", file_path_str)
