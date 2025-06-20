import os
from typing import Optional, Union, List, Dict, Tuple, Any
from pathlib import Path
from logging import getLogger
from glob import glob
import cv2
import numpy as np
from hailo_toolbox.converters.onnx2hailo import Onnx2Hef
from hailo_toolbox.converters.tf2hailo import TensorFlowConverter
from hailo_sdk_client.model_translator.exceptions import (
    MisspellNodeError,
    ParsingWithRecommendationException,
    UnsupportedModelError,
)

logger = getLogger(__file__)


IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.gif", "*.webp"]


class HailoConverter:
    def __init__(
        self,
        model_path: str,
        hw_arch: Union[str] = "hailo8",
        input_shape: Optional[Tuple[int]] = None,
        model_script: Optional[Union[str, Path]] = None,
        image_dir: Optional[Union[str, Path]] = None,
        calibration_dataset_size: int = 100,
        save_onnx: bool = False,
        end_nodes: Optional[Tuple[str]] = None,
    ):
        if model_path.endswith(".onnx"):
            self.converter = Onnx2Hef(
                model_path,
                hw_arch=hw_arch,
                input_shape=input_shape,
                model_script=model_script,
                image_dir=image_dir,
                save_onnx=save_onnx,
                end_node=end_nodes,
                calibration_dataset_size=calibration_dataset_size,
            )
        else:
            self.converter = TensorFlowConverter(
                model_path,
                hw_arch=hw_arch,
                input_shape=input_shape,
                model_script=model_script,
                image_dir=image_dir,
                calibration_dataset_size=calibration_dataset_size,
            )

    def convert(self):
        return self.converter.convert()

    def set_calibration_dataset(
        self,
        calibration_dataset: Optional[Union[str, Path]] = None,
        height: int = 224,
        width: int = 224,
        rgb: bool = True,
    ):
        self.calibration_dataset = []
        if calibration_dataset:
            image_paths = []
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(
                    glob(os.path.join(calibration_dataset, ext), recursive=True)
                )
            image_paths = image_paths[: self.calibration_dataset_size]
            for img_path in image_paths:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (height, width))
                if rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.calibration_dataset.append(img)
        else:
            if rgb:
                self.calibration_dataset = np.random.randint(
                    0, 255, (self.calibration_dataset_size, height, width, 3)
                )
            else:
                self.calibration_dataset = np.random.randint(
                    0, 255, (self.calibration_dataset_size, height, width, 1)
                )

    def visualize_har(
        self, har_file: Optional[Union[str, Path, bytes]] = None, verbose: bool = False
    ):
        """
        Visualize the HAR file.
        """
        if har_file:
            self.set_har_file(har_file)

        svg_file = self.har_file.with_suffix(".svg")

        self._get_hailo_nn().visualize(svg_file.absolute().as_posix(), verbose=verbose)

    def visualize_hn(
        self, hn_file: Optional[Union[str, Path, bytes]] = None, verbose: bool = False
    ):
        """
        Visualize the HN file.
        """
        if hn_file:
            self.set_hn_file(hn_file)

        svg_file = self.hn_file.with_suffix(".svg")

        self.converter.runner.visualize(svg_file.absolute().as_posix(), verbose=verbose)

    def add_model_script(
        self, script: Optional[Union[str, Path]] = None, append: bool = False
    ):
        if script:
            self.converter.runner.load_model_script(script, append)
        else:
            self.converter.runner.load_model_script(self.script, append)

    def parse(self):
        self.converter.runner.load_model_script()

    def optimize(self):
        assert hasattr(self, "calibration_dataset"), "Calibration dataset is not set"
        self.converter.runner.optimize(self.calibration_dataset)
        self.converter.runner.save_har(
            self.model_path.with_name(self.model_path.stem + "_optimized.har")
        )

    def compile(self):
        hef_model = self.converter.runner.compile()
        self.converter.runner.save_model(self.model_path.with_suffix(".hef"), hef_model)

    def dump_model_info(self):
        pass


if __name__ == "__main__":
    converter = HailoConverter(
        model_path="/home/dq/github/PaddleOCR/inference/rec_onnx/rec_v3_tmp.onnx",
    )
    hef_path = converter.convert()
    print(hef_path)
