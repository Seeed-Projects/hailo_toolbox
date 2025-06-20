"""
ONNX to HEF model converter.
"""

import os
import os.path as osp
import sys
from typing import Optional, Tuple, List

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import onnx
import numpy as np
import argparse
import logging
from hailo_toolbox.converters.base import BaseConverter

logger = logging.getLogger("hailo")


class Onnx2Hailo(BaseConverter):
    """
    Converter for ONNX models to HEF format.

    This class implements the conversion of ONNX models to HEF format
    with support for various configurations and optimizations.
    """

    def __init__(
        self,
        onnx_file: str = None,
        hw_arch: str = "hailo8",
        input_shape: Optional[Tuple[int, int, int]] = None,
        calibration_dataset_size: int = 100,
        save_onnx: bool = False,
        model_script: str = None,
        image_dir: str = None,
        end_nodes: Optional[Tuple[str]] = None,
    ):
        """
        Initialize the ONNX to HEF converter.

        Parameters
        ----------
        onnx_file: str
            Path to the ONNX model file (as string)
        hw_arch: str
            Target architecture name, default is "hailo8"
        input_shape: Tuple[int, int, int]
            Input shape (height, width, channels), default is None
        calibration_dataset_size: int
            Size of the calibration dataset, default is 100
        save_onnx: bool
            Whether to save the optimized ONNX model, default is False
        model_script: str
            Model script content or path to script file
        image_dir: str
            Path to the image directory for calibration
        end_nodes: str
            End node name for partial model conversion

        Attributes
        ----------
        onnx_file: str
            Path to the ONNX model file
        target_arch: str
            Target architecture name
        input_shape: Tuple[int, int, int]
            Input shape for the model
        save_onnx: bool
            Whether to save optimized ONNX model
        image_dir: str
            Path to calibration images directory
        end_node: str
            End node name
        """
        super().__init__(onnx_file, hw_arch, model_script, calibration_dataset_size)

        # Set ONNX file path
        self.set_onnx_file(onnx_file)
        self.target_arch = hw_arch
        self.save_onnx = save_onnx
        self.image_dir = image_dir
        self.end_nodes = end_nodes

        # Handle input shape
        if input_shape is not None:
            self.input_shape = input_shape
        else:
            # Get input shape from ONNX model
            input_shapes = self.get_onnx_input_shape()
            if len(input_shapes) > 0:
                # Convert from NCHW to HWC format
                batch, channels, height, width = input_shapes[0]
                self.input_shape = (height, width, channels)
            else:
                raise RuntimeError("ONNX model has no input")

        logger.info("ONNX converter initialized:")
        logger.info("  Model file: %s", self.onnx_file)
        logger.info("  Target architecture: %s", self.target_arch)
        logger.info("  Input shape: %s", self.input_shape)
        logger.info("  Calibration dataset size: %d", self.calibration_dataset_size)

    def get_onnx_input_shape(self) -> List[Tuple[int, int, int, int]]:
        """
        Get input shapes from ONNX model.

        Returns:
            List of input shapes as tuples.

        Raises:
            AssertionError: If ONNX model is not valid.
        """
        if not self.validate_onnx(self.onnx_file):
            raise AssertionError("ONNX model is not valid")

        onnx_model = onnx.load(self.onnx_file)
        inputs = []
        for i in onnx_model.graph.input:
            shape = tuple(map(lambda x: int(x.dim_value), i.type.tensor_type.shape.dim))
            inputs.append(shape)
        return inputs

    def convert(self) -> str:
        """
        Convert ONNX model to HEF format.

        This function performs the complete conversion pipeline:
        1. Validate ONNX model
        2. Parse model information
        3. Load calibration images
        4. Translate ONNX model using Hailo SDK
        5. Optimize model with calibration data
        6. Compile model to HEF format
        7. Optionally save optimized ONNX model

        Returns
        -------
        str
            Path to the generated HEF file

        Raises
        ------
        ImportError
            If hailo-sdk-client is not installed
        RuntimeError
            If model conversion fails
        """
        try:
            from hailo_sdk_client import ClientRunner
        except ImportError as ie:
            logger.error("Import error: %s", ie)
            raise ImportError("Please install hailo-sdk-client to use this function.")

        logger.info("Starting ONNX to HEF conversion")

        # Create and configure ClientRunner
        runner = ClientRunner(hw_arch=self.target_arch)
        logger.info("ClientRunner initialized for architecture: %s", self.target_arch)

        # Get ONNX model information
        onnx_model_graph_info = self.get_onnx_info()
        logger.info("ONNX model info parsed successfully")
        logger.debug("Model info: %s", onnx_model_graph_info)

        # Load calibration images
        logger.info("Loading calibration images with shape: %s", self.input_shape)
        self.load_calibration_images(self.image_dir, self.input_shape)
        logger.info(
            "Calibration images loaded: %d samples", len(self.calibrat_datasets)
        )

        # Create network input shapes dictionary
        net_input_shapes = {
            name: shape
            for name, shape in zip(
                onnx_model_graph_info["start_nodes_name"],
                onnx_model_graph_info["inputs_shape"],
            )
        }
        logger.debug("Network input shapes: %s", net_input_shapes)

        # Translate ONNX model
        logger.info("Translating ONNX model: %s", self.onnx_file)
        runner.translate_onnx_model(
            self.onnx_file,
            start_node_names=onnx_model_graph_info["start_nodes_name"],
            end_node_names=(
                self.end_nodes
                if self.end_nodes is not None
                else onnx_model_graph_info["end_nodes_name"]
            ),
            net_input_shapes=net_input_shapes,
        )
        runner.save_har(self.har_file)
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

        # Optionally save optimized ONNX model
        if self.save_onnx:
            logger.info("Saving optimized ONNX model")
            onnx_model_for_hailo = runner.get_hailo_runtime_model()
            onnx.save(onnx_model_for_hailo, self.hailoonnx_file)
            logger.info("Optimized ONNX model saved to: %s", self.hailoonnx_file)

        return self.hef_file

    def profile(self):
        try:
            from hailo_sdk_client import ClientRunner
        except ImportError as ie:
            logger.error("Import error: %s", ie)
            raise ImportError("Please install hailo-sdk-client to use this function.")
        runner = ClientRunner(har=self.har_file)
        print(runner.profile(hef_filename=self.hef_file, runtime_data="profile.json"))


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert ONNX to HEF")
    parser.add_argument(
        "--onnx_file",
        "-of",
        type=str,
        default="/home/dq/github/PaddleOCR/inference/rec_onnx/rec_v3_tmp.onnx",
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--input_shape",
        "-is",
        type=str,
        default="48,200,3",
        help="Input shape in format height,width,channels",
    )
    parser.add_argument(
        "--image_dir",
        "-id",
        type=str,
        default="/home/dq/github/PaddleOCR/dataset_output/rec",
        help="Path to the image directory for calibration",
    )
    parser.add_argument(
        "--end_node",
        "-en",
        type=str,
        default=None,
        help="End node name for partial model conversion",
    )
    parser.add_argument(
        "--arch", type=str, help="Target architecture name", default="hailo8"
    )
    parser.add_argument(
        "--save_onnx", action="store_true", help="Save optimized ONNX model"
    )
    parser.add_argument(
        "--calibration_size", type=int, default=100, help="Size of calibration dataset"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Parse input shape
    input_shape = None
    if args.input_shape:
        try:
            shape_values = [int(x) for x in args.input_shape.split(",")]
            if len(shape_values) == 3:
                input_shape = tuple(shape_values)
            else:
                raise ValueError("Input shape must have 3 dimensions (H,W,C)")
        except ValueError as e:
            logger.error("Invalid input shape format: %s", e)
            sys.exit(1)

    # Create converter
    converter = Onnx2Hef(
        onnx_file=args.onnx_file,
        hw_arch=args.arch,
        input_shape=input_shape,
        calibration_dataset_size=args.calibration_size,
        save_onnx=args.save_onnx,
        image_dir=args.image_dir,
        end_nodes=args.end_node,
    )

    # Convert model
    # try:
    # hef_path = converter.convert()
    print(f"Conversion completed successfully!")
    # print(f"HEF model saved to: {hef_path}")
    converter.profile()
    # except Exception as e:
    #     logger.error("Conversion failed: %s", e)
    #     sys.exit(1)
