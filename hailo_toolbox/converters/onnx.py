import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import onnx
import numpy as np
import argparse
import logging
from hailo_toolbox.converters.base import BaseConverter
from hailo_toolbox.converters.utils import load_calibration_images

logger = logging.getLogger("hailo")


def generate_calibrat_datasets(
    image_dir=None, image_shape=(100, 640, 640, 3)
) -> np.ndarray:
    if image_dir is not None:
        return load_calibration_images(image_dir, 100, image_shape, data_type="uint8")
    else:
        return np.random.randn(*image_shape).astype("f")


class Onnx2Hef(BaseConverter):
    def __init__(
        self,
        onnx_file: str = None,
        arch: str = "hailo8",
        input_shape: str = "1,3,48,320",
        image_dir: str = None,
        end_node: str = None,
    ):
        """
        Parameters
        ----------
        onnx_file: str
            Path to the onnx model file
        arch: str
            Target architecture name, default is "hailo8"
        input_shape: str
            Input shape, default is "1,3,48,320"
        image_dir: str
            Path to the image directory
        end_node: str
            End node name
        Attributes
        ----------
        onnx_file: str
            Path to the onnx model file
        target_arch: str
            Target architecture name
        all_available_devices: List[str]
            List of all available Hailo devices
        """

        super().__init__()
        self.onnx_file = onnx_file
        self.target_arch = arch
        self.input_shape = [int(x) for x in input_shape.split(",")]
        self.image_dir = image_dir
        self.end_node = end_node

    @property
    def hef_file(self):
        """
        Property to get the path to the hef file, which is the onnx file with the extension replaced with ".hef".

        Returns
        -------
        str
            Path to the hef file
        """
        return self.onnx_file.replace(".onnx", ".hef")

    @property
    def hailoonnx_file(self):
        """
        Property to get the path to the modified onnx file, which is the onnx file with "_hailo" added to the filename before the extension.

        Returns
        -------
        str
            Path to the modified onnx file
        """
        return self.onnx_file.replace(".onnx", "_hailo.onnx")

    def translate_onnx2hef(self, save_onnx=False) -> str:
        """
        Translate an onnx model to a hef model.

        This function translates an onnx model to a hef model using the Hailo SDK client.
        It first loads the onnx model, then generates calibration datasets, optimizes the model
        using the calibration datasets, compiles the model, saves the model as a hef file, and
        saves the model as an onnx file with "_hailo" added to the filename before the extension.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Path to the hef file
        """
        try:
            from hailo_sdk_client import ClientRunner
        except ImportError as ie:
            logger.error(ie)
            logger.warning("Please install hailo-sdk-client to use this function.")

        runner = ClientRunner(hw_arch=self.target_arch)
        onnx_model_graph_info = self.get_onnx_info()
        print(onnx_model_graph_info)
        net_input_shapes = {
            name: shape
            for name, shape in zip(
                onnx_model_graph_info["start_nodes_name"],
                onnx_model_graph_info["inputs_shape"],
            )
        }

        # load onnx model
        print(
            onnx_model_graph_info["start_nodes_name"],
            onnx_model_graph_info["end_nodes_name"],
            net_input_shapes,
        )
        runner.translate_onnx_model(
            self.onnx_file,
            start_node_names=onnx_model_graph_info["start_nodes_name"],
            end_node_names=onnx_model_graph_info["end_nodes_name"],
            net_input_shapes=net_input_shapes,
        )

        # generate calibration datasets
        calibrat_datasets = generate_calibrat_datasets(
            image_dir=self.image_dir, image_shape=self.input_shape
        )
        print(
            "\033[31m 校验数据集属性：",
            np.max(calibrat_datasets),
            np.min(calibrat_datasets),
            calibrat_datasets.dtype,
            calibrat_datasets.shape,
            "\033[0m",
        )
        model_script = "normalization1 = normalization([127.5, 127.5, 127.5],[127.5, 127.5, 127.5])"
        # model_script = "normalization1 = normalization([123.675,116.28,103.53],[58.395,57.12,57.375])\n"

        print("\033[31m 加载模型脚本 \033[0m")
        runner.load_model_script(model_script)
        print("\033[31m 优化模型 \033[0m")
        runner.optimize(calibrat_datasets)
        print("\033[31m 编译模型 \033[0m")

        # compile model
        hef_model = runner.compile()
        print("\033[31m 保存模型 \033[0m")
        # save hef model
        self.save_model(hef_model, self.hef_file)
        print("\033[31m 保存模型完成 位于：", self.hef_file, "\033[0m")
        # save model
        if save_onnx:
            print("\033[31m 保存onnx模型 \033[0m")
            onnx_model_for_hailo = runner.get_hailo_runtime_model()
            onnx.save(onnx_model_for_hailo, self.hailoonnx_file)
            print("\033[31m 保存onnx模型完成 位于：", self.hailoonnx_file, "\033[0m")
        return self.hef_file

    def save_model(self, model, file_path):
        """
        Save the model to a file.

        Parameters
        ----------
        model: bytes
            Hailo model in bytes
        file_path: str
            Path to the file to save the model
        """
        with open(file_path, "wb") as f:
            f.write(model)

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

        print("checking onnx model...")
        onnx_model = onnx.load(self.onnx_file)
        onnx.checker.check_model(onnx_model)
        print("onnx model is valid")
        inputs = []
        start_nodes_name = []
        for i in onnx_model.graph.input:
            inputs.append(
                tuple(map(lambda x: int(x.dim_value), i.type.tensor_type.shape.dim))
            )
            start_nodes_name.append(i.name)

        if self.end_node is None:
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

    def translate_onnx2hailoonnx(self):
        pass

    def check_onnx_backend(self):
        pass

    def transform(self):
        pass

    def save(self):
        pass


def get_args():
    parser = argparse.ArgumentParser(description="Convert onnx to hef")
    parser.add_argument(
        "--onnx_file",
        "-of",
        type=str,
        default="/home/dq/github/PaddleOCR/inference/rec_onnx/rec_v3_tmp.onnx",
        help="Path to the onnx model file",
    )
    parser.add_argument(
        "--input_shape", "-is", type=str, default="3,48,200", help="Input shape"
    )
    parser.add_argument(
        "--image_dir",
        "-id",
        type=str,
        default="/home/dq/github/PaddleOCR/dataset_output/rec",
        help="Path to the image directory",
        # default="/home/dq/github/PaddleOCR/train_data/rec/train", help="Path to the image directory"
    )
    parser.add_argument(
        "--end_node", "-en", type=str, default=None, help="End node name"
    )
    parser.add_argument(
        "--arch", type=str, help="Target architecture name", default="hailo8"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    onnx2hef = OnnxRT2HailoRT(
        args.onnx_file, args.arch, args.input_shape, args.image_dir, args.end_node
    )
    onnx2hef.translate_onnx2hef()
