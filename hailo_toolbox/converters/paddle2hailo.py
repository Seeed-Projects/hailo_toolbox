"""
PaddlePaddle model converter to HEF format.
"""

import os
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple, Dict

from hailo_toolbox.converters.base import BaseConverter
from hailo_toolbox.utils import get_logger

# create logger
logger = get_logger(__name__)


class Paddle2Hailo(BaseConverter):
    """
    Converter for PaddlePaddle models to HEF format.

    This class implements the conversion of PaddlePaddle models to HEF format
    by first converting to ONNX format, then to HEF using Hailo SDK.
    The conversion pipeline: PaddlePaddle -> ONNX -> HEF

    Supports various PaddlePaddle model formats:
    - .pdmodel + .pdiparams files
    - .inference.pdmodel + .inference.pdiparams files
    - Directory containing model files
    - Compressed model archives (.tar.gz)
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
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        model_dir: Optional[str] = None,
        model_filename: Optional[str] = None,
        params_filename: Optional[str] = None,
        opset_version: int = 11,
        enable_onnx_checker: bool = True,
        **kwargs,
    ):
        """
        Initialize the PaddlePaddle converter.

        Args:
            model_path: Path to the PaddlePaddle model file or directory.
            hw_arch: Hardware architecture (default: "hailo8").
            model_script: Model script content or path to script file.
            image_dir: Directory containing calibration images.
            calibration_dataset_size: Size of calibration dataset.
            input_shape: Input shape for the model (batch_size, channels, height, width).
            end_nodes: End nodes for the model.
            input_names: Names of input nodes for ONNX export.
            output_names: Names of output nodes for ONNX export.
            model_dir: Directory containing model files (alternative to model_path).
            model_filename: Name of the model file (default: "__model__").
            params_filename: Name of the params file (default: "__params__").
            opset_version: ONNX opset version for export (default: 11).
            enable_onnx_checker: Whether to enable ONNX model checker (default: True).
            **kwargs: Additional parameters specific to PaddlePaddle conversion.
        """
        super().__init__(
            model_path, hw_arch, model_script, calibration_dataset_size, **kwargs
        )

        # Initialize PaddlePaddle-specific attributes
        self.paddle_model_path = model_path
        self.end_nodes = end_nodes
        self.image_dir = image_dir
        self.opset_version = opset_version
        self.enable_onnx_checker = enable_onnx_checker

        # Model file configuration
        self.model_dir = model_dir
        self.model_filename = model_filename or "__model__"
        self.params_filename = params_filename or "__params__"

        # ONNX export parameters
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]

        # Process model path and validate files
        self.model_info_dict = self._process_model_path()

        # Handle input shape - convert to NCHW format if needed
        if input_shape is not None:
            if len(input_shape) == 3:
                # Convert HWC to NCHW (add batch dimension)
                h, w, c = input_shape
                self.input_shape = (1, c, h, w)
                self.calibration_shape = input_shape  # Keep HWC for calibration
            elif len(input_shape) == 4:
                # Already in NCHW format
                self.input_shape = input_shape
                # Convert to HWC for calibration
                n, c, h, w = input_shape
                self.calibration_shape = (h, w, c)
            else:
                raise ValueError("input_shape must be 3D (H,W,C) or 4D (N,C,H,W)")
        else:
            # Default input shape
            self.input_shape = (1, 3, 224, 224)  # NCHW
            self.calibration_shape = (224, 224, 3)  # HWC

        logger.info(
            "PaddlePaddle converter initialized for: %s", self.paddle_model_path
        )
        logger.info("Model directory: %s", self.model_info_dict.get("model_dir"))
        logger.info("Model filename: %s", self.model_info_dict.get("model_filename"))
        logger.info("Params filename: %s", self.model_info_dict.get("params_filename"))
        logger.info("Input shape (NCHW): %s", self.input_shape)
        logger.info("Calibration shape (HWC): %s", self.calibration_shape)

    def _process_model_path(self) -> Dict[str, str]:
        """
        Process the model path and determine model file locations.

        Returns:
            Dictionary containing model directory and file information.

        Raises:
            FileNotFoundError: If model files are not found.
            ValueError: If model path format is not supported.
        """
        model_info = {}

        if self.model_dir:
            # Use explicitly provided model directory
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

            model_info["model_dir"] = self.model_dir
            model_info["model_filename"] = self.model_filename
            model_info["params_filename"] = self.params_filename

        elif os.path.isdir(self.paddle_model_path):
            # Model path is a directory
            model_info["model_dir"] = self.paddle_model_path

            # Try to find model files in the directory
            possible_model_files = [
                "__model__",
                "model.pdmodel",
                "inference.pdmodel",
                f"{os.path.basename(self.paddle_model_path)}.pdmodel",
            ]

            possible_param_files = [
                "__params__",
                "model.pdiparams",
                "inference.pdiparams",
                f"{os.path.basename(self.paddle_model_path)}.pdiparams",
            ]

            # Find model file
            model_file = None
            for fname in possible_model_files:
                full_path = os.path.join(self.paddle_model_path, fname)
                if os.path.exists(full_path):
                    model_file = fname
                    break

            if not model_file:
                raise FileNotFoundError(
                    f"No model file found in directory: {self.paddle_model_path}. "
                    f"Expected one of: {possible_model_files}"
                )

            # Find params file
            params_file = None
            for fname in possible_param_files:
                full_path = os.path.join(self.paddle_model_path, fname)
                if os.path.exists(full_path):
                    params_file = fname
                    break

            if not params_file:
                raise FileNotFoundError(
                    f"No params file found in directory: {self.paddle_model_path}. "
                    f"Expected one of: {possible_param_files}"
                )

            model_info["model_filename"] = model_file
            model_info["params_filename"] = params_file

        elif os.path.isfile(self.paddle_model_path):
            # Handle single model file
            if self.paddle_model_path.endswith(".tar.gz"):
                # Extract compressed model
                model_info = self._extract_compressed_model(self.paddle_model_path)
            elif self.paddle_model_path.endswith(".pdmodel"):
                # Handle .pdmodel file
                model_dir = os.path.dirname(self.paddle_model_path)
                model_filename = os.path.basename(self.paddle_model_path)

                # Find corresponding .pdiparams file
                params_filename = model_filename.replace(".pdmodel", ".pdiparams")
                params_path = os.path.join(model_dir, params_filename)

                if not os.path.exists(params_path):
                    raise FileNotFoundError(f"Params file not found: {params_path}")

                model_info["model_dir"] = model_dir
                model_info["model_filename"] = model_filename
                model_info["params_filename"] = params_filename
            else:
                raise ValueError(
                    f"Unsupported model file format: {self.paddle_model_path}. "
                    "Expected .pdmodel, .tar.gz, or directory containing model files."
                )
        else:
            raise FileNotFoundError(f"Model path not found: {self.paddle_model_path}")

        logger.debug("Processed model path: %s", model_info)
        return model_info

    def _extract_compressed_model(self, compressed_path: str) -> Dict[str, str]:
        """
        Extract compressed PaddlePaddle model.

        Args:
            compressed_path: Path to compressed model file (.tar.gz).

        Returns:
            Dictionary containing extracted model information.
        """
        import tarfile

        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="paddle_model_")

        try:
            # Extract compressed file
            with tarfile.open(compressed_path, "r:gz") as tar:
                tar.extractall(temp_dir)

            logger.info(f"Extracted model to temporary directory: {temp_dir}")

            # Find model files in extracted directory
            extracted_files = os.listdir(temp_dir)
            logger.debug(f"Extracted files: {extracted_files}")

            # Look for model directory or files
            for item in extracted_files:
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path):
                    # Found a directory, use it as model directory
                    self.temp_model_dir = item_path
                    return self._process_model_directory(item_path)

            # If no directory found, treat temp_dir as model directory
            self.temp_model_dir = temp_dir
            return self._process_model_directory(temp_dir)

        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to extract compressed model: {e}")

    def _process_model_directory(self, model_dir: str) -> Dict[str, str]:
        """
        Process a model directory to find model and params files.

        Args:
            model_dir: Path to model directory.

        Returns:
            Dictionary containing model directory and file information.
        """
        files = os.listdir(model_dir)

        # Find model file
        model_file = None
        for fname in files:
            if fname.endswith(".pdmodel") or fname == "__model__":
                model_file = fname
                break

        if not model_file:
            raise FileNotFoundError(f"No model file found in directory: {model_dir}")

        # Find params file
        params_file = None
        if model_file.endswith(".pdmodel"):
            # Look for corresponding .pdiparams file
            expected_params = model_file.replace(".pdmodel", ".pdiparams")
            if expected_params in files:
                params_file = expected_params

        if not params_file:
            # Look for common params file names
            for fname in files:
                if fname.endswith(".pdiparams") or fname == "__params__":
                    params_file = fname
                    break

        if not params_file:
            raise FileNotFoundError(f"No params file found in directory: {model_dir}")

        return {
            "model_dir": model_dir,
            "model_filename": model_file,
            "params_filename": params_file,
        }

    def paddle_to_onnx(self, onnx_path: Optional[str] = None) -> str:
        """
        Convert PaddlePaddle model to ONNX format.

        Args:
            onnx_path: Output path for ONNX model. If None, uses default path.

        Returns:
            Path to the generated ONNX file.

        Raises:
            ImportError: If paddle2onnx is not installed.
            RuntimeError: If conversion fails.
        """
        try:
            import paddle2onnx
        except ImportError:
            raise ImportError(
                "paddle2onnx is required for PaddlePaddle model conversion. "
                "Please install paddle2onnx: pip install paddle2onnx"
            )

        logger.info("Converting PaddlePaddle to ONNX: %s", self.paddle_model_path)

        # Generate output path if not provided
        if onnx_path is None:
            onnx_path = self.onnx_file

        # Prepare conversion parameters
        model_dir = self.model_info_dict["model_dir"]
        model_filename = self.model_info_dict["model_filename"]
        params_filename = self.model_info_dict["params_filename"]

        try:
            # Convert PaddlePaddle to ONNX
            paddle2onnx.command.c_paddle_to_onnx(
                model_dir=model_dir,
                model_filename=model_filename,
                params_filename=params_filename,
                save_file=onnx_path,
                opset_version=self.opset_version,
                enable_onnx_checker=self.enable_onnx_checker,
                input_shape_dict=self._get_input_shape_dict()
                if hasattr(self, "_get_input_shape_dict")
                else None,
                input_names=self.input_names,
                output_names=self.output_names,
            )

            logger.info("ONNX model exported to: %s", onnx_path)

            # Validate the exported ONNX model
            if not self.validate_onnx(onnx_path):
                raise RuntimeError("Exported ONNX model validation failed")

            return onnx_path

        except Exception as e:
            raise RuntimeError(f"Failed to export PaddlePaddle to ONNX: {e}")

    def _get_input_shape_dict(self) -> Optional[Dict[str, List[int]]]:
        """
        Get input shape dictionary for paddle2onnx conversion.

        Returns:
            Dictionary mapping input names to shapes, or None if not needed.
        """
        if self.input_names and len(self.input_names) > 0:
            return {self.input_names[0]: list(self.input_shape)}
        return None

    def convert(self) -> str:
        """
        Convert PaddlePaddle model to HEF format.

        This method performs the complete conversion pipeline:
        1. Convert PaddlePaddle to ONNX
        2. Parse ONNX model information
        3. Load calibration images
        4. Translate ONNX model using Hailo SDK
        5. Optimize and compile the model
        6. Save the HEF model

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

        logger.info("Starting PaddlePaddle to HEF conversion")

        try:
            # Step 1: Convert PaddlePaddle to ONNX
            logger.info("Step 1: Converting PaddlePaddle to ONNX")
            onnx_path = self.paddle_to_onnx()

            # Update ONNX file path in base class
            self.set_onnx_file(onnx_path)

            # Step 2: Parse ONNX model information
            logger.info("Step 2: Parsing ONNX model information")
            self.parser_model_info()
            logger.info("Model info parsed successfully")
            logger.debug("Model info: %s", self.model_info)

            # Step 3: Load calibration images with appropriate shape
            logger.info("Step 3: Loading calibration images")
            self.load_calibration_images(
                image_dir=self.image_dir, image_shape=self.calibration_shape
            )
            logger.info(
                "Calibration images loaded: %d samples", len(self.calibrat_datasets)
            )

            # Step 4: Create and configure ClientRunner
            logger.info("Step 4: Initializing ClientRunner")
            runner = ClientRunner(hw_arch=self.hw_arch)
            logger.info("ClientRunner initialized for architecture: %s", self.hw_arch)

            # Create network input shapes dictionary
            net_input_shapes = {
                name: shape
                for name, shape in zip(
                    self.model_info["start_nodes_name"],
                    self.model_info["inputs_shape"],
                )
            }
            logger.debug("Network input shapes: %s", net_input_shapes)

            # Step 5: Translate ONNX model
            logger.info("Step 5: Translating ONNX model")
            runner.translate_onnx_model(
                onnx_path,
                start_node_names=self.model_info["start_nodes_name"],
                end_node_names=(
                    self.end_nodes
                    if self.end_nodes is not None
                    else self.model_info["end_nodes_name"]
                ),
                net_input_shapes=net_input_shapes,
            )
            runner.save_har(self.har_file)
            logger.info("Model translation completed")

            # Step 6: Load model script for optimization
            logger.info("Step 6: Loading model script")
            runner.load_model_script(self.model_script)

            # Step 7: Optimize model with calibration data
            logger.info("Step 7: Optimizing model with calibration data")
            runner.optimize(self.calibrat_datasets)
            logger.info("Model optimization completed")

            # Step 8: Compile model to HEF
            logger.info("Step 8: Compiling model to HEF format")
            hef_model = runner.compile()
            logger.info("Model compilation completed")

            # Step 9: Save HEF model
            logger.info("Step 9: Saving HEF model")
            self.save_model(hef_model, self.hef_file)
            logger.info("HEF model saved successfully to: %s", self.hef_file)

            return self.hef_file

        finally:
            # Clean up temporary files if they exist
            self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """
        Clean up temporary files created during conversion.
        """
        if hasattr(self, "temp_model_dir") and os.path.exists(self.temp_model_dir):
            try:
                shutil.rmtree(self.temp_model_dir)
                logger.info(
                    "Cleaned up temporary model directory: %s", self.temp_model_dir
                )
            except Exception as e:
                logger.warning("Failed to clean up temporary directory: %s", e)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the PaddlePaddle model.

        Returns:
            Dictionary containing model information.
        """
        model_info = {
            "model_path": self.paddle_model_path,
            "model_dir": self.model_info_dict.get("model_dir"),
            "model_filename": self.model_info_dict.get("model_filename"),
            "params_filename": self.model_info_dict.get("params_filename"),
            "input_shape": self.input_shape,
            "calibration_shape": self.calibration_shape,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "opset_version": self.opset_version,
        }

        # Try to get more detailed information from the model files
        try:
            model_dir = self.model_info_dict["model_dir"]
            model_filename = self.model_info_dict["model_filename"]
            params_filename = self.model_info_dict["params_filename"]

            model_file_path = os.path.join(model_dir, model_filename)
            params_file_path = os.path.join(model_dir, params_filename)

            model_info["model_file_size"] = os.path.getsize(model_file_path)
            model_info["params_file_size"] = os.path.getsize(params_file_path)

        except Exception as e:
            logger.warning("Could not extract detailed model information: %s", e)

        return model_info

    def profile(self):
        """
        Profile the converted HEF model.

        This method provides performance profiling of the compiled HEF model
        using Hailo SDK profiling tools.
        """
        if not os.path.exists(self.hef_file):
            raise FileNotFoundError(f"HEF file not found: {self.hef_file}")

        if not os.path.exists(self.har_file):
            raise FileNotFoundError(f"HAR file not found: {self.har_file}")

        try:
            from hailo_sdk_client import ClientRunner
        except ImportError as ie:
            logger.error("Import error: %s", ie)
            raise ImportError("Please install hailo-sdk-client to use this function.")

        logger.info("Starting model profiling")
        runner = ClientRunner(har=self.har_file)
        profile_result = runner.profile(
            hef_filename=self.hef_file, runtime_data="paddle_profile.json"
        )
        logger.info("Model profiling completed")
        print(profile_result)
        return profile_result


if __name__ == "__main__":
    # Example usage
    converter = Paddle2Hailo(
        model_path="/path/to/paddle/model",  # Directory or .pdmodel file
        input_shape=(224, 224, 3),  # HWC format
        image_dir="/path/to/calibration/images",
        input_names=["input"],
        output_names=["output"],
    )

    try:
        hef_path = converter.convert()
        print(f"Conversion completed successfully!")
        print(f"HEF model saved to: {hef_path}")

        # Optional: Profile the model
        converter.profile()
    except Exception as e:
        logger.error("Conversion failed: %s", e)
        raise
