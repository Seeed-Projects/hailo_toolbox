#!/usr/bin/env python3
"""
Debug script to isolate the array truth value error.
"""

import warnings
import traceback
import sys
import numpy as np
import tempfile
import os
import cv2
from pathlib import Path
from hailo_toolbox.process.interface import (
    Pipeline,
    ProcessingStage,
    DataType,
    PipelineConfig,
)
from hailo_toolbox.process.adapters import (
    ImageSourceAdapter,
    PreprocessorAdapter,
    InferenceAdapter,
    PostprocessorAdapter,
    VisualizationAdapter,
    OutputAdapter,
)
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig
from hailo_toolbox.process.base import PostprocessConfig

# Enable all warnings and convert them to exceptions for array truth values
warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("error", message=".*truth value.*")


class MockInferenceEngine:
    """Mock inference engine for testing."""

    def __init__(self, task_type: str = "detection"):
        self.task_type = task_type
        print(f"Initialized MockInferenceEngine for {task_type}")

    def infer(self, input_data: dict) -> dict:
        """Mock inference that returns realistic outputs."""
        batch_size = 1

        if self.task_type == "detection":
            # YOLOv8 detection output: [batch, num_detections, 84]
            # 84 = 4 (bbox) + 80 (classes)
            num_detections = 5
            output = np.random.rand(batch_size, num_detections, 84).astype(np.float32)
            # Set some realistic confidence scores
            output[0, :, 4:] = (
                np.random.rand(num_detections, 80) * 0.3
            )  # Low confidence for most classes
            output[
                0, :3, np.random.randint(4, 84, 3)
            ] = 0.9  # High confidence for 3 detections
            return {"output": output}

        elif self.task_type == "segmentation":
            # Segmentation output
            num_detections = 5
            output = np.random.rand(batch_size, num_detections, 84).astype(np.float32)
            masks = np.random.rand(batch_size, num_detections, 160, 160).astype(
                np.float32
            )
            return {"output": output, "masks": masks}

        elif self.task_type == "keypoint":
            # Keypoint detection output
            num_persons = 3
            output = np.random.rand(batch_size, num_persons, 56).astype(
                np.float32
            )  # 4 + 17*3 + 1
            return {"output": output}

        return {"output": np.random.rand(batch_size, 10, 84).astype(np.float32)}


def create_mock_image(width: int = 640, height: int = 640) -> np.ndarray:
    """Create a mock image for testing."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[: height // 2, : width // 2] = [255, 0, 0]  # Red quadrant
    image[: height // 2, width // 2 :] = [0, 255, 0]  # Green quadrant
    image[height // 2 :, : width // 2] = [0, 0, 255]  # Blue quadrant
    image[height // 2 :, width // 2 :] = [255, 255, 0]  # Yellow quadrant
    return image


def debug_with_traceback():
    """Debug the pipeline with detailed traceback information."""
    print("=== DEBUGGING WITH TRACEBACK ===")

    try:
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image = create_mock_image()
            cv2.imwrite(tmp_file.name, image)
            temp_image_path = tmp_file.name

        # Create output path
        output_path = temp_image_path.replace(".jpg", "_debug_output.jpg")

        print(f"Created temporary image: {temp_image_path}")
        print(f"Output will be saved to: {output_path}")

        # Configure pipeline
        preprocess_config = PreprocessConfig(target_size=(640, 640))
        postprocess_config = PostprocessConfig(
            det=True, det_conf_threshold=0.25, num_classes=80
        )

        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            source_config={"path": temp_image_path, "type": "file"},
            preprocess_config={"target_size": (640, 640)},
            inference_config={"task_type": "detection"},
            postprocess_config={
                "det": True,
                "det_conf_threshold": 0.25,
                "num_classes": 80,
            },
            visualization_config={},
            output_config={"path": output_path, "format": "image"},
            continue_on_error=False,  # Disable error continuation to see full traceback
        )

        # Create pipeline
        pipeline = Pipeline(pipeline_config)

        # Add components step by step with detailed logging
        print("\n--- Adding SOURCE component ---")
        source_adapter = ImageSourceAdapter(temp_image_path)
        pipeline.add_component(source_adapter)
        print("✓ SOURCE component added successfully")

        print("\n--- Adding PREPROCESSING component ---")
        preprocess_adapter = PreprocessorAdapter(preprocess_config)
        pipeline.add_component(preprocess_adapter)
        print("✓ PREPROCESSING component added successfully")

        print("\n--- Adding INFERENCE component ---")
        inference_engine = MockInferenceEngine("detection")
        inference_adapter = InferenceAdapter(inference_engine)
        pipeline.add_component(inference_adapter)
        print("✓ INFERENCE component added successfully")

        print("\n--- Adding POSTPROCESSING component ---")
        postprocess_adapter = PostprocessorAdapter("detection", postprocess_config)
        pipeline.add_component(postprocess_adapter)
        print("✓ POSTPROCESSING component added successfully")

        print("\n--- Adding VISUALIZATION component ---")
        visualization_adapter = VisualizationAdapter()
        pipeline.add_component(visualization_adapter)
        print("✓ VISUALIZATION component added successfully")

        print("\n--- Adding OUTPUT component ---")
        output_adapter = OutputAdapter(output_path, "image")
        pipeline.add_component(output_adapter)
        print("✓ OUTPUT component added successfully")

        print("\n--- Initializing pipeline ---")
        pipeline.initialize()
        print("✓ Pipeline initialized successfully")

        print("\n--- Processing data ---")
        result = pipeline.process_single()
        print("✓ Pipeline processing completed successfully")

        print(f"Final result type: {type(result)}")
        if hasattr(result, "metadata"):
            print(f"Final data type: {result.metadata.data_type}")

        # Clean up
        Path(temp_image_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"\n!!! ERROR CAUGHT: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()

        # Try to get more specific information about array truth value errors
        if "truth value" in str(e).lower():
            print("\n=== ARRAY TRUTH VALUE ERROR ANALYSIS ===")
            print("This error typically occurs when:")
            print("1. Using 'and', 'or', or 'not' with numpy arrays")
            print("2. Using arrays in if statements without .any() or .all()")
            print("3. Comparing arrays with ambiguous truth values")

            # Print the specific line that caused the error
            tb = traceback.extract_tb(e.__traceback__)
            for frame in tb:
                if "truth value" in str(e).lower():
                    print(f"\nProblem likely in file: {frame.filename}")
                    print(f"Line {frame.lineno}: {frame.line}")


def debug_step_by_step():
    """Debug each step individually to isolate the error."""
    print("\n=== STEP-BY-STEP DEBUGGING ===")

    try:
        # Create mock data
        image = create_mock_image()
        print(f"✓ Created mock image with shape: {image.shape}")

        # Test each component individually
        print("\n--- Testing SOURCE adapter ---")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            source_adapter = ImageSourceAdapter(tmp_file.name)
            # This should not cause issues
            print("✓ SOURCE adapter created successfully")

        print("\n--- Testing PREPROCESSING adapter ---")
        preprocess_config = PreprocessConfig(target_size=(640, 640))
        preprocess_adapter = PreprocessorAdapter(preprocess_config)
        print("✓ PREPROCESSING adapter created successfully")

        print("\n--- Testing INFERENCE adapter ---")
        inference_engine = MockInferenceEngine("detection")
        inference_adapter = InferenceAdapter(inference_engine)
        print("✓ INFERENCE adapter created successfully")

        print("\n--- Testing POSTPROCESSING adapter ---")
        postprocess_config = PostprocessConfig(
            det=True, det_conf_threshold=0.25, num_classes=80
        )
        postprocess_adapter = PostprocessorAdapter("detection", postprocess_config)
        print("✓ POSTPROCESSING adapter created successfully")

        print("\n--- Testing VISUALIZATION adapter ---")
        visualization_adapter = VisualizationAdapter()
        print("✓ VISUALIZATION adapter created successfully")

        print("\n--- Testing OUTPUT adapter ---")
        output_adapter = OutputAdapter("/tmp/test_output.jpg", "image")
        print("✓ OUTPUT adapter created successfully")

        print("\n--- Testing pipeline creation ---")
        pipeline_config = PipelineConfig(
            continue_on_error=False
        )  # Disable error continuation
        pipeline = Pipeline(pipeline_config)
        print("✓ Pipeline created successfully")

        print("\n--- Testing component addition ---")
        pipeline.add_component(source_adapter)
        pipeline.add_component(preprocess_adapter)
        pipeline.add_component(inference_adapter)
        pipeline.add_component(postprocess_adapter)
        pipeline.add_component(visualization_adapter)
        pipeline.add_component(output_adapter)
        print("✓ All components added successfully")

        print("\n--- Testing pipeline initialization ---")
        pipeline.initialize()
        print("✓ Pipeline initialized successfully")

        print("\n--- Testing pipeline processing ---")
        result = pipeline.process_single()
        print("✓ Pipeline processing completed successfully")

    except Exception as e:
        print(f"\n!!! ERROR in step-by-step debugging: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()


if __name__ == "__main__":
    print("COMPREHENSIVE ARRAY TRUTH VALUE DEBUGGING")
    print("=" * 50)

    # First try the detailed traceback approach
    debug_with_traceback()

    print("\n" + "=" * 50)

    # Then try step-by-step debugging
    debug_step_by_step()

    print("\n" + "=" * 50)
    print("DEBUGGING COMPLETED")
