"""
TorchScript model converter to HEF format.
"""

import os
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple, Dict

from hailo_toolbox.converters.base import BaseConverter
from hailo_toolbox.utils import get_logger

# create logger
logger = get_logger(__name__)


class TorchScript2Hailo(BaseConverter):
    """
    Converter for TorchScript models to HEF format.

    This class implements the conversion of TorchScript models to HEF format
    by first converting to ONNX format, then to HEF using Hailo SDK.
    The conversion pipeline: TorchScript -> ONNX -> HEF
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
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        **kwargs,
    ):
        """
        Initialize the TorchScript converter.

        Args:
            model_path: Path to the TorchScript model file (.pt or .pth).
            hw_arch: Hardware architecture (default: "hailo8").
            model_script: Model script content or path to script file.
            image_dir: Directory containing calibration images.
            calibration_dataset_size: Size of calibration dataset.
            input_shape: Input shape for the model (batch_size, channels, height, width).
            end_nodes: End nodes for the model.
            input_names: Names of input nodes for ONNX export.
            output_names: Names of output nodes for ONNX export.
            dynamic_axes: Dynamic axes configuration for ONNX export.
            **kwargs: Additional parameters specific to TorchScript conversion.
        """
        super().__init__(
            model_path, hw_arch, model_script, calibration_dataset_size, **kwargs
        )

        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TorchScript model file not found: {model_path}")

        # Validate model file extension
        if not model_path.endswith((".pt", ".pth")):
            raise ValueError(
                f"Unsupported TorchScript model extension. Expected .pt or .pth"
            )

        self.torchscript_path = model_path
        self.end_nodes = end_nodes
        self.image_dir = image_dir

        # ONNX export parameters
        self.input_names = input_names or ["input"]
        self.output_names = output_names or ["output"]
        self.dynamic_axes = dynamic_axes

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

        logger.info("TorchScript converter initialized for: %s", self.torchscript_path)
        logger.info("Input shape (NCHW): %s", self.input_shape)
        logger.info("Calibration shape (HWC): %s", self.calibration_shape)

    def torchscript_to_onnx(self, onnx_path: Optional[str] = None) -> str:
        """
        Convert TorchScript model to ONNX format.

        Args:
            onnx_path: Output path for ONNX model. If None, uses default path.

        Returns:
            Path to the generated ONNX file.

        Raises:
            ImportError: If PyTorch is not installed.
            RuntimeError: If conversion fails.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for TorchScript model conversion. "
                "Please install torch: pip install torch"
            )

        logger.info("Converting TorchScript to ONNX: %s", self.torchscript_path)

        # Load TorchScript model
        try:
            model = torch.jit.load(self.torchscript_path, map_location="cpu")
            model.eval()
            logger.info("TorchScript model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load TorchScript model: {e}")

        # Generate output path if not provided
        if onnx_path is None:
            onnx_path = self.onnx_file

        # Create dummy input tensor
        dummy_input = torch.randn(self.input_shape)
        logger.info("Created dummy input with shape: %s", dummy_input.shape)

        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes,
                opset_version=11,  # Use ONNX opset version 11 for better compatibility
                do_constant_folding=True,  # Optimize constant folding
                export_params=True,
                verbose=False,
            )
            logger.info("ONNX model exported to: %s", onnx_path)

            # Validate the exported ONNX model
            if not self.validate_onnx(onnx_path):
                raise RuntimeError("Exported ONNX model validation failed")

            return onnx_path

        except Exception as e:
            raise RuntimeError(f"Failed to export TorchScript to ONNX: {e}")

    def convert(self) -> str:
        """
        Convert TorchScript model to HEF format.

        This method performs the complete conversion pipeline:
        1. Convert TorchScript to ONNX
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

        logger.info("Starting TorchScript to HEF conversion")

        # Step 1: Convert TorchScript to ONNX
        logger.info("Step 1: Converting TorchScript to ONNX")
        onnx_path = self.torchscript_to_onnx()

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

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the TorchScript model.

        Returns:
            Dictionary containing model information.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to get model information")

        # Load the model
        model = torch.jit.load(self.torchscript_path, map_location="cpu")
        model.eval()

        # Get model structure information
        model_info = {
            "model_path": self.torchscript_path,
            "input_shape": self.input_shape,
            "calibration_shape": self.calibration_shape,
            "input_names": self.input_names,
            "output_names": self.output_names,
        }

        # Try to get more detailed information if possible
        try:
            # Get graph information
            graph = model.graph
            model_info["graph_inputs"] = [str(inp) for inp in graph.inputs()]
            model_info["graph_outputs"] = [str(out) for out in graph.outputs()]
        except Exception as e:
            logger.warning("Could not extract detailed graph information: %s", e)

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
            hef_filename=self.hef_file, runtime_data="torchscript_profile.json"
        )
        logger.info("Model profiling completed")
        print(profile_result)
        return profile_result


if __name__ == "__main__":
    # Example usage
    converter = TorchScript2Hailo(
        model_path="/path/to/model.pt",
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
