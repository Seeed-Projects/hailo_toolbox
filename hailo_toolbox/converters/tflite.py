"""
TensorFlow model converter to ONNX format.
"""

import os
import numpy as np
from typing import Any, Optional, Union, List, Tuple, Dict

from .base import BaseConverter
from ..utils import get_logger

# 创建模块的日志记录器
logger = get_logger(__name__)


class TensorFlowConverter(BaseConverter):
    """
    Converter for TensorFlow models to ONNX format.

    This class implements the conversion of TensorFlow models to ONNX format
    with support for various configurations and optimizations.
    """

    def __init__(
        self,
        input_shape: Optional[Union[List[int], Tuple[int, ...]]] = None,
        opset_version: int = 11,
        output_dir: str = "converted_models",
        **kwargs,
    ):
        """
        Initialize the TensorFlow converter.

        Args:
            input_shape: Input shape for the model (batch_size, height, width, channels).
            opset_version: ONNX opset version to use.
            output_dir: Directory to save converted models.
            **kwargs: Additional parameters specific to TensorFlow conversion.
        """
        super().__init__(input_shape, opset_version, output_dir, **kwargs)

        # Default input shape for image models if none provided
        if input_shape is None:
            self.input_shape = (
                1,
                224,
                224,
                3,
            )  # TensorFlow uses channels last by default
            logger.info("使用默认输入形状: %s", self.input_shape)

        # Check for TF requirements
        self._check_requirements()
        logger.debug("已初始化TensorFlow转换器")

    def _check_requirements(self):
        """
        Check if all required packages for TensorFlow conversion are installed.
        """
        try:
            import tensorflow as tf
            import tf2onnx

            logger.debug(
                "所有TensorFlow转换要求已满足: TensorFlow %s, tf2onnx 已安装",
                tf.__version__,
            )
        except ImportError as e:
            logger.warning("缺少依赖: %s. 需要安装 tensorflow 和 tf2onnx", e)
            logger.warning("可以通过执行 'pip install tensorflow tf2onnx' 安装依赖")

    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load a TensorFlow model from a file.

        Args:
            model_path: Path to the TensorFlow model (SavedModel, H5, or frozen graph).
            **kwargs: Additional loading parameters:
                - model_type: Type of the model ('saved_model', 'keras', 'frozen').
                - custom_load_func: Optional function to handle custom loading logic.
                - tags: Tags for loading SavedModel (default: ['serve']).

        Returns:
            Loaded TensorFlow model.
        """
        try:
            import tensorflow as tf

            model_type = kwargs.get("model_type", self._infer_model_type(model_path))
            custom_load_func = kwargs.get("custom_load_func", None)

            logger.info("加载TensorFlow模型: %s (类型=%s)", model_path, model_type)

            if custom_load_func:
                logger.debug("使用自定义加载函数")
                return custom_load_func(model_path, **kwargs)

            if model_type == "saved_model":
                tags = kwargs.get("tags", ["serve"])
                logger.debug("加载SavedModel，使用标签: %s", tags)
                return tf.saved_model.load(model_path, tags=tags)

            elif model_type == "keras":
                logger.debug("加载Keras模型")
                return tf.keras.models.load_model(model_path)

            elif model_type == "frozen":
                logger.debug("加载冻结图")
                with tf.io.gfile.GFile(model_path, "rb") as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())

                with tf.compat.v1.Graph().as_default() as graph:
                    tf.import_graph_def(graph_def, name="")

                return graph

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            logger.error("模型加载失败: %s", e)
            raise RuntimeError(f"Failed to load TensorFlow model: {e}")

    def _infer_model_type(self, model_path: str) -> str:
        """
        Infer the type of TensorFlow model from the path.

        Args:
            model_path: Path to the TensorFlow model.

        Returns:
            String indicating the model type ('saved_model', 'keras', 'frozen').
        """
        import os

        if os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "saved_model.pb")):
                logger.debug("根据目录结构推断为SavedModel")
                return "saved_model"
            elif os.path.exists(os.path.join(model_path, "keras_metadata.pb")):
                logger.debug("根据目录结构推断为Keras模型")
                return "keras"
        elif model_path.endswith(".h5") or model_path.endswith(".hdf5"):
            logger.debug("根据文件扩展名推断为Keras模型")
            return "keras"
        elif model_path.endswith(".pb"):
            logger.debug("根据文件扩展名推断为冻结图")
            return "frozen"

        # Default to saved_model as the most common format
        logger.warning("无法推断模型类型，默认为SavedModel")
        return "saved_model"

    def convert(
        self, model_path: str, output_name: Optional[str] = None, **kwargs
    ) -> str:
        """
        Convert a TensorFlow model to ONNX format.

        Args:
            model_path: Path to the TensorFlow model.
            output_name: Optional name for the output ONNX file.
            **kwargs: Additional conversion parameters:
                - input_names: Names of input nodes (optional).
                - output_names: Names of output nodes (optional).
                - input_shapes: Dict mapping input names to shapes (optional).
                - model_type: Type of the model ('saved_model', 'keras', 'frozen').
                - large_model: Whether to use external data format for large models.
                - Any parameter accepted by load_model.

        Returns:
            Path to the converted ONNX model.
        """
        try:
            import tensorflow as tf
            import tf2onnx
            import tf2onnx.convert

            # Generate output path
            if output_name is None:
                base_name = os.path.basename(model_path)
                if os.path.isdir(model_path):
                    # For directories, use the directory name
                    name_no_ext = os.path.basename(model_path)
                else:
                    # For files, remove the extension
                    name_no_ext = os.path.splitext(base_name)[0]
                output_name = f"{name_no_ext}.onnx"

            output_path = os.path.join(self.output_dir, output_name)
            logger.info("转换TensorFlow模型到ONNX: %s -> %s", model_path, output_path)

            # Get model type
            model_type = kwargs.get("model_type", self._infer_model_type(model_path))

            # Convert based on model type
            input_names = kwargs.get("input_names", None)
            output_names = kwargs.get("output_names", None)

            # Handle large model option
            large_model = kwargs.get("large_model", False)
            external_data = None
            if large_model:
                logger.info("使用外部数据格式处理大型模型")
                external_data_name = f"{os.path.splitext(output_name)[0]}_data"
                external_data = os.path.join(self.output_dir, external_data_name)

            # Process by model type
            if model_type == "keras":
                self._convert_keras(
                    model_path,
                    output_path,
                    input_names,
                    output_names,
                    external_data,
                    **kwargs,
                )
            elif model_type == "saved_model":
                self._convert_saved_model(
                    model_path,
                    output_path,
                    input_names,
                    output_names,
                    external_data,
                    **kwargs,
                )
            elif model_type == "frozen":
                self._convert_frozen(
                    model_path,
                    output_path,
                    input_names,
                    output_names,
                    external_data,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Validate the model
            if kwargs.get("validate", True):
                if self.validate_onnx(output_path):
                    logger.info("ONNX模型验证通过")
                else:
                    logger.warning("ONNX模型验证失败")

            # Optimize if requested
            if kwargs.get("optimize", True):
                output_path = self.optimize_onnx(output_path)

            return output_path

        except Exception as e:
            logger.error("ONNX转换失败: %s", e)
            raise RuntimeError(f"Failed to convert model to ONNX: {e}")

    def _convert_keras(
        self,
        model_path,
        output_path,
        input_names,
        output_names,
        external_data,
        **kwargs,
    ):
        """Convert Keras model to ONNX."""
        import tensorflow as tf
        import tf2onnx

        logger.info("使用Keras路径转换模型")

        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Get model spec
        input_specs = []
        if hasattr(model, "inputs") and model.inputs:
            for inp in model.inputs:
                spec = tf.TensorSpec(inp.shape, inp.dtype, name=inp.name.split(":")[0])
                input_specs.append(spec)

        # Convert model
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_specs if input_specs else None,
            opset=self.opset_version,
            output_path=output_path,
            external_data_format=external_data is not None,
            external_data_name=(
                os.path.basename(external_data) if external_data else None
            ),
        )

        logger.info("Keras模型转换为ONNX成功")

    def _convert_saved_model(
        self,
        model_path,
        output_path,
        input_names,
        output_names,
        external_data,
        **kwargs,
    ):
        """Convert SavedModel to ONNX."""
        import tensorflow as tf
        import tf2onnx

        logger.info("使用SavedModel路径转换模型")

        # Get tags
        tags = kwargs.get("tags", ["serve"])
        signature_def = kwargs.get("signature_def", "serving_default")

        # Convert model
        model_proto, _ = tf2onnx.convert.from_saved_model(
            model_path,
            input_names=input_names,
            output_names=output_names,
            tag=tags,
            signature_def=signature_def,
            opset=self.opset_version,
            output_path=output_path,
            external_data_format=external_data is not None,
            external_data_name=(
                os.path.basename(external_data) if external_data else None
            ),
        )

        logger.info("SavedModel模型转换为ONNX成功")

    def _convert_frozen(
        self,
        model_path,
        output_path,
        input_names,
        output_names,
        external_data,
        **kwargs,
    ):
        """Convert frozen graph to ONNX."""
        import tensorflow as tf
        import tf2onnx

        logger.info("使用冻结图路径转换模型")

        if not input_names or not output_names:
            raise ValueError(
                "For frozen graph conversion, input_names and output_names are required"
            )

        # Load frozen graph
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        # Convert model
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name="")

            # Get input and output tensors
            inputs = [tf_graph.get_tensor_by_name(name) for name in input_names]
            outputs = [tf_graph.get_tensor_by_name(name) for name in output_names]

            model_proto, _ = tf2onnx.convert.from_graph_def(
                graph_def,
                input_names=input_names,
                output_names=output_names,
                opset=self.opset_version,
                output_path=output_path,
                external_data_format=external_data is not None,
                external_data_name=(
                    os.path.basename(external_data) if external_data else None
                ),
            )

        logger.info("冻结图模型转换为ONNX成功")

    def optimize_onnx(self, onnx_path: str) -> str:
        """
        Optimize the ONNX model using ONNX Runtime.

        Args:
            onnx_path: Path to the ONNX model.

        Returns:
            Path to the optimized ONNX model.
        """
        try:
            import onnx
            import onnxruntime as ort

            logger.info("使用ONNX Runtime优化模型: %s", onnx_path)

            # Generate output path
            opt_path = os.path.splitext(onnx_path)[0] + "_optimized.onnx"

            # Load model
            model = onnx.load(onnx_path)

            # Create a session to optimize the model
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.optimized_model_filepath = opt_path
            _ = ort.InferenceSession(onnx_path, sess_options)

            logger.info("优化后的模型已保存: %s", opt_path)
            return opt_path

        except ImportError:
            logger.warning("ONNX Runtime未安装，跳过优化")
            return onnx_path
        except Exception as e:
            logger.error("ONNX优化失败: %s", e)
            return onnx_path
