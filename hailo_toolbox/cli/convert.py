"""
Command-line interface for model conversion.
"""

import os
import sys
import argparse
import yaml
import json
from typing import Dict, Any, Optional, List, Tuple, Union

from ..converters import BaseConverter, PyTorchConverter, TensorFlowConverter
from ..utils import setup_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as f:
        if ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")


def get_converter(framework: str, config: Dict[str, Any]) -> BaseConverter:
    """
    Get the appropriate converter for the specified framework.

    Args:
        framework: The source framework ("pytorch", "tensorflow").
        config: Converter configuration.

    Returns:
        Appropriate converter instance.
    """
    framework = framework.lower()

    if framework == "pytorch" or framework == "torch":
        return PyTorchConverter(**config)
    elif framework == "tensorflow" or framework == "tf":
        return TensorFlowConverter(**config)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert deep learning models to ONNX format"
    )

    parser.add_argument("model_path", type=str, help="Path to the source model file")

    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["pytorch", "torch", "tensorflow", "tf"],
        help="Source framework of the model",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="converted_models",
        help="Directory to save the converted model",
    )

    parser.add_argument("--output-name", type=str, help="Name for the output ONNX file")

    parser.add_argument(
        "--input-shape",
        type=str,
        help="Input shape for the model (comma-separated integers, e.g., '1,3,224,224')",
    )

    parser.add_argument(
        "--opset-version", type=int, default=11, help="ONNX opset version to use"
    )

    parser.add_argument(
        "--dynamic-axes", type=str, help="Dynamic axes configuration (JSON string)"
    )

    parser.add_argument(
        "--config", type=str, help="Path to a YAML or JSON configuration file"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--log-file", type=str, default="convert.log", help="Log file name"
    )

    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the model conversion CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(
        name="hailo_toolbox.convert",
        level=log_level,
        log_file=args.log_file,
        log_dir=args.log_dir,
    )

    try:
        # Load config from file if specified
        config = {}
        if args.config:
            config = load_config(args.config)
            logger.info("Loaded config file: %s", args.config)

        # Override config with command-line arguments
        if args.output_dir:
            config["output_dir"] = args.output_dir

        if args.opset_version:
            config["opset_version"] = args.opset_version

        if args.input_shape:
            try:
                # Parse comma-separated input shape
                input_shape = tuple(int(dim) for dim in args.input_shape.split(","))
                config["input_shape"] = input_shape
                logger.info("Using input shape: %s", input_shape)
            except Exception as e:
                logger.error("Error parsing input shape: %s", e)
                return 1

        if args.dynamic_axes:
            try:
                # Parse JSON dynamic axes configuration
                dynamic_axes = json.loads(args.dynamic_axes)
                config["dynamic_axes"] = dynamic_axes
                logger.info("Using dynamic axes config: %s", dynamic_axes)
            except Exception as e:
                logger.error("Error parsing dynamic axes: %s", e)
                return 1

        # Check if model file exists
        if not os.path.exists(args.model_path):
            logger.error("Model file not found: %s", args.model_path)
            return 1

        # Create output directory if it doesn't exist
        output_dir = config.get("output_dir", "converted_models")
        os.makedirs(output_dir, exist_ok=True)

        # Get converter for the specified framework
        try:
            converter = get_converter(args.framework, config)
            logger.info("Using %s converter", args.framework)
        except Exception as e:
            logger.error("Error creating converter: %s", e)
            return 1

        # Convert the model
        try:
            logger.info("Starting model conversion: %s", args.model_path)
            onnx_path = converter.convert(args.model_path, args.output_name)
            logger.info("Model conversion successful: %s", onnx_path)

            # Validate the model
            if converter.validate_onnx(onnx_path):
                logger.info("ONNX model validation passed")
            else:
                logger.warning("ONNX model validation failed")

        except Exception as e:
            logger.exception("Model conversion error")
            return 1

        return 0

    except Exception as e:
        logger.exception("Unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
