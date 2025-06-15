"""
Base class for model converters to ONNX format.
"""

from abc import ABC, abstractmethod
import os
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from ..utils import get_logger

from ..utils.excep import FileFormatException

from contextlib import suppress
from tarfile import ReadError

from hailo_sdk_client import ClientRunner
from hailo_sdk_common.hailo_nn.hailo_nn import HailoNN
from hailo_sdk_client.hailo_archive.hailo_archive import HailoArchiveLoader

# 创建模块的日志记录器
logger = get_logger(__name__)


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
        save_onnx: bool = False,
        model_script: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize the base converter.

        Args:
            model_path: Path to the source model.
            framework: Framework of the source model.
            hw_arch: Hardware architecture to use.
            **kwargs: Additional framework-specific parameters.
        """
        self.model_path = model_path
        assert model_path.endswith((".onnx", ".tflite"))
        self.framework = "onnx" if model_path.endswith(".onnx") else "tflite"
        self.hw_arch = hw_arch
        self.save_onnx = save_onnx
        self.set_origin_model_file()

        self.load_model_script(model_script)

        self._onnx_file = None
        self._har_file = None
        self._hn_file = None
        self._tf_file = None

        logger.debug("初始化转换器: framework=%s, hw_arch=%s", self.framework, hw_arch)
        self._validate_and_prepare()

        self.runner = ClientRunner(hn=model_path, hw_arch=hw_arch)

    def _validate_and_prepare(self):
        """
        Validate parameters and prepare for conversion.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

    def load_model_script(self, model_script: Optional[Union[str, Path]]) -> None:
        if model_script is None:
            self.model_script = "normalization1 = normalization([0,0,0],[255,255,255])"
            return
        if isinstance(model_script, str):
            if os.path.exists(model_script):
                with open(model_script, "r") as f:
                    self.model_script = f.read()
            else:
                self.model_script = model_script
        elif isinstance(model_script, Path):
            if model_script.exists():
                with open(model_script, "r") as f:
                    self.model_script = f.read()
            else:
                raise FileNotFoundError(f"model_script: {model_script} not found")
        else:
            raise ValueError("model_script must be a string or a Path object")

    def set_origin_model_file(self) -> None:
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if self.model_path.suffix == ".har":
            self.set_har_file(self.model_path)
        elif self.model_path.suffix == ".hn":
            self.set_hn_file(self.model_path)
        elif self.model_path.suffix == ".onnx":
            self.set_onnx_file(self.model_path)
        else:
            raise ValueError("Unsupported model file format.")

    @property
    def onnx_file(self) -> Path:
        return self._onnx_file

    def set_onnx_file(self, onnx_file: str) -> None:
        path = Path(onnx_file)
        if path.suffix != ".onnx":
            raise ValueError("he input file must be in ONNX format.")
        self._onnx_file = path

    @property
    def har_file(self) -> Path:
        return self._har_file

    def set_har_file(self, har_file: Union[str, Path]) -> None:
        path = Path(har_file)
        if path.suffix != ".har":
            raise ValueError("he input file must be in HAR format.")
        self._har_file = path

    @property
    def hn_file(self) -> Path:
        return self._hn_file

    def set_hn_file(self, hn_file: Union[str, Path]) -> None:
        path = Path(hn_file)
        if path.suffix != ".hn":
            raise ValueError("he input file must be in HN format.")
        self._hn_file = path

    @property
    def tf_file(self) -> Path:
        return self._tf_file

    def set_tf_file(self, tf_file: Union[str, Path]) -> None:
        path = Path(tf_file)
        if path.suffix != ".tf":
            raise ValueError("he input file must be in TF format.")
        self._tf_file = path

    @property
    def start_node_names(self) -> List[str]:
        return self._start_node_names

    @property
    def end_node_names(self) -> List[str]:
        return self._end_node_names

    @property
    def net_input_shapes(self) -> Dict[str, Tuple[int, int, int, int]]:
        return self._net_input_shapes

    def _fix_har(self):
        if self.har_file.suffix == ".har":
            with suppress(ReadError), HailoArchiveLoader(self.har_file) as har_loader:
                hn = json.loads(har_loader.get_hn())
                hn.pop("direct_control", None)
                return hn

    def _fix_hn(self):
        if self.hn_file.suffix == ".hn":
            with suppress(json.JSONDecodeError), open(self.hn_file) as hn_file:
                return json.load(hn_file)

        raise FileFormatException("The given model must be a valid HAR file")

    def _get_hailo_nn(self) -> HailoNN:
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

    @abstractmethod
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
        logger.debug("使用基础优化（无优化）: %s", onnx_path)
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
            logger.info("验证ONNX模型: %s", onnx_path)
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX模型验证通过")
            return True
        except Exception as e:
            logger.error("ONNX模型验证失败: %s", e)
            return False
