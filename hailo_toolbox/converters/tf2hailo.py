"""
TensorFlow model converter to ONNX format.
"""

import os
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple, Dict

from hailo_toolbox.converters.base import BaseConverter
from hailo_toolbox.utils import get_logger

# create logger
logger = get_logger(__name__)


class TensorFlow2Hailo(BaseConverter):
    """
    Converter for TensorFlow models to ONNX format.

    This class implements the conversion of TensorFlow models to ONNX format
    with support for various configurations and optimizations.
    """

    def __init__(
        self,
        model_path: str,
        hw_arch: str = "hailo8",
        model_script: Optional[Union[str, Path]] = None,
        image_dir: Optional[str] = None,
        calibration_dataset_size: int = 100,
        input_shape: Optional[Union[List[int], Tuple[int, ...]]] = None,
        end_nodes: Optional[Tuple[str]] = None,
        **kwargs,
    ):
        """
        Initialize the TensorFlow converter.

        Args:
            model_path: Path to the model file or directory (as string).
            hw_arch: Hardware architecture (default: "hailo8").
            model_script: Model script content or path to script file.
            image_dir: Directory containing calibration images.
            calibration_dataset_size: Size of calibration dataset.
            input_shape: Input shape for the model (batch_size, height, width, channels).
            end_nodes: End nodes for the model.
            **kwargs: Additional parameters specific to TensorFlow conversion.
        """
        super().__init__(
            model_path, hw_arch, model_script, calibration_dataset_size, **kwargs
        )
        self.end_nodes = end_nodes

        # Handle different model path types
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                # TensorFlow SavedModel directory
                self.model_path = model_path
                self.set_tf_file(self.model_path)
                self.pb2tflite(self.model_path)
            elif model_path.endswith(".tflite"):
                # TensorFlow Lite file
                self.model_path = model_path
                self.set_tf_file(model_path)
            elif model_path.endswith(".h5"):
                # Keras H5 model file
                self.model_path = model_path
                self.set_tf_file(model_path)
                self.h52tflite(model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.image_dir = image_dir

        # Store input shape if provided
        self.input_shape = input_shape

        logger.info("TensorFlow converter initialized for: %s", self.model_path)

    def pb2tflite(self, pb_path: str) -> str:
        """
        Convert TensorFlow SavedModel to TensorFlow Lite format.

        Args:
            pb_path: Path to the SavedModel directory.

        Returns:
            Path to the generated TFLite file.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for TensorFlow model conversion. "
                "Please install tensorflow: pip install tensorflow"
            )

        logger.info("Converting SavedModel to TFLite: %s", pb_path)

        converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()

        # Generate output path
        tflite_path = os.path.splitext(self.tf_file)[0] + ".tflite"

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        logger.info("TFLite model saved to: %s", tflite_path)
        return tflite_path

    def h52tflite(self, h5_path: str) -> str:
        """
        Convert Keras H5 model to TensorFlow Lite format.

        Args:
            h5_path: Path to the H5 model file.

        Returns:
            Path to the generated TFLite file.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for TensorFlow model conversion. "
                "Please install tensorflow: pip install tensorflow"
            )

        logger.info("Converting H5 model to TFLite: %s", h5_path)

        keras_model = tf.keras.models.load_model(h5_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()

        # Generate output path
        tflite_path = os.path.splitext(self.tf_file)[0] + ".tflite"

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        logger.info("TFLite model saved to: %s", tflite_path)
        return tflite_path

    def convert(self) -> str:
        """
        Convert TensorFlow model to HEF format.

        This method performs the complete conversion pipeline:
        1. Parse model information
        2. Load calibration images
        3. Translate the model using Hailo SDK
        4. Optimize and compile the model
        5. Save the HEF model

        Returns:
            Path to the generated HEF file.

        Raises:
            ImportError: If hailo-sdk-client is not installed.
            RuntimeError: If conversion fails.
        """
        try:
            from hailo_sdk_client import ClientRunner
        except ImportError as ie:
            logger.error("Import error: %s", ie)
            raise ImportError("Please install hailo-sdk-client to use this function.")

        logger.info("Starting TensorFlow to HEF conversion")

        # Parse model information
        self.parser_model_info()
        logger.info("Model info parsed successfully")
        logger.debug("Model info: %s", self.model_info)

        # Load calibration images with appropriate shape
        calibration_shape = self.input_shape if self.input_shape else [32, 32, 3]
        self.load_calibration_images(
            image_dir=self.image_dir, image_shape=calibration_shape
        )
        logger.info(
            "Calibration images loaded: %d samples", len(self.calibrat_datasets)
        )

        # Create and configure ClientRunner
        runner = ClientRunner(hw_arch=self.hw_arch)
        logger.info("ClientRunner initialized for architecture: %s", self.hw_arch)

        # Translate TensorFlow Lite model
        tflite_file = (
            self.tf_file
            if self.tf_file.endswith(".tflite")
            else f"{os.path.splitext(self.tf_file)[0]}.tflite"
        )

        logger.info("Translating TFLite model: %s", tflite_file)
        runner.translate_tf_model(
            tflite_file,
            start_node_names=self.model_info["start_nodes_name"],
            end_node_names=(
                self.end_nodes
                if self.end_nodes is not None
                else self.model_info["end_nodes_name"]
            ),
            tensor_shapes=self.model_info["inputs_shape"],
        )
        logger.info("Model translation completed")

        # Load model script for optimization
        logger.info("Loading model script")
        runner.load_model_script(self.model_script)

        # Optimize model with calibration data
        logger.info("Optimizing model with calibration data")
        runner.optimize(self.calibrat_datasets)
        logger.info("Model optimization completed")

        # Compile model to HEF
        logger.info("Compiling model to HEF format")
        hef_model = runner.compile()
        logger.info("Model compilation completed")

        # Save HEF model
        logger.info("Saving HEF model to: %s", self.hef_file)
        self.save_model(hef_model, self.hef_file)
        logger.info("HEF model saved successfully")

        return self.hef_file


if __name__ == "__main__":
    # Example usage
    converter = TensorFlow2Hailo(
        model_path="/home/dq/github/hailo_tutorials/models/dense_example_tf2"
    )
    hef_path = converter.convert()
    print(f"Conversion completed. HEF model saved to: {hef_path}")
