import os
import sys

os.environ["HAILO_MONITOR"] = "1"

from typing import Tuple, Callable
import numpy as np
from hailo_toolbox.utils.logging import get_logger
from hailo_toolbox.inference.format import NodeInfo
from multiprocessing import Queue
from typing import List, Dict
from yaml import load, FullLoader
from hailo_toolbox.utils.timer import Timer
import time
from multiprocessing import shared_memory, Process
from hailo_toolbox.utils.sharememory import ShareMemoryManager
from threading import Thread


logger = get_logger(__file__)

try:
    from hailo_platform import (
        HEF,
        VDevice,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        InputVStreams,
        OutputVStreams,
        HailoSchedulingAlgorithm,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
    )
except ImportError:
    logger.error("hailo_platform not found")


class HailoInference:
    def __init__(self, hef_path: str) -> None:
        self.hef = HEF(hef_path)
        self.init_output_quant_info()
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name

        self.input_shape = self.hef.get_input_vstream_infos()[0].shape
        self.output_shape = self.hef.get_output_vstream_infos()[0].shape

        self.inited_predict_flag = False
        self.inited_as_process_flag = False

        self.is_initialized = False
        self.as_process_flag = False
        self.process = None
        self.input_queue = None
        self.output_queue = None

        self.thead_number = 0
        self.callback_list = []

        print(self.input_name, self.output_name, self.input_shape, self.output_shape)

    def load_config(self, config_path: str):
        with open(config_path, "r") as f:
            config = load(f, Loader=FullLoader)
        return config

    def init_as_predict(self):
        self.as_process_flag = False
        self.target = VDevice()

        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef, self.configure_params)

        self.network_group = self.network_groups[0]

        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )

        self.infer = InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params
        )
        self.activater = self.network_group.activate(self.network_group_params)
        self.inited_predict_flag = True

    def init_as_process(
        self, input_queue: Queue, output_queue: Queue, thread_number: int
    ):
        self.as_process_flag = True
        self.shared_params = VDevice.create_params()
        self.shared_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.shared_params.group_id = "SHARED"
        self.shared_params.multi_process_service = False

        with VDevice(self.shared_params) as target:
            self.configure_params = ConfigureParams.create_from_hef(
                hef=self.hef, interface=HailoStreamInterface.PCIe
            )
            self.model_name = self.hef.get_network_group_names()[0]
            batch_size = 1
            self.configure_params[self.model_name].batch_size = batch_size
            # self.configure_params.set_batch_size(batch_size)
            self.network_groups = target.configure(self.hef, self.configure_params)
            self.network_group = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()
            self.input_vstreams_params = InputVStreamParams.make(
                self.network_group, format_type=FormatType.UINT8
            )
            self.output_vstreams_params = OutputVStreamParams.make(
                self.network_group, format_type=FormatType.FLOAT32
            )

            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as infer:
                while True:
                    shm_info = input_queue.get()
                    image = self.share_memory_manager.read(**shm_info)
                    image = np.expand_dims(image.astype(np.uint8), axis=0)
                    # with Timer(f"inference"):
                    results = infer.infer(image)
                    # output_data = self.dequantization(results)
                    results = self.callback(results)
                    shm_info = self.share_memory_manager.write_dict(results)
                    output_queue.put(shm_info)

            self.inited_as_process_flag = True

    def callback(self, results):
        for callback in self.callback_list:
            results = callback(results)
        return results

    def init_async_model(self):
        self.device_params = VDevice.create_params()
        self.device_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.target = VDevice(self.device_params)
        self.infer_model = self.target.create_infer_model(
            self.config[self.infer_type]["hef"]
        )
        self.infer_model.set_batch_size(60 if self.infer_type == "rec" else 1)
        self.infer_model.input().set_format_type(FormatType.UINT8)
        self.infer_model.output().set_format_type(FormatType.UINT8)

    def init_as_async_process(self):
        with self.infer_model.configure() as configured_infer_model:
            while True:
                shm_info = self.input_queue.get()

    def _create_binding(self, configured_infer_model):
        output_buffers = {
            name: np.empty(self.infer_model.output().shape(), dtype=np.uint8)
            for name in configured_infer_model.output().info()
        }
        binding = configured_infer_model.create_binding(output_buffers)
        return binding

    def dequantization(
        self, output_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        for key, value in output_data.items():
            value = np.array(value[0], dtype=object)
            value = (
                value - self.output_quant_info[key]["qp_zero_point"]
            ) * self.output_quant_info[key]["qp_scale"]
            output_data[key] = value
        return output_data

    def _initialize_queues(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.share_memory_manager = ShareMemoryManager(max_size=640 * 48 * 3 * 100)

    def _init_dequantization_info(self):
        a = self.hef.get_output_stream_infos()[0].quant_info
        self.qp_scale = a.qp_scale
        self.qp_zero_point = a.qp_zp

    def init_all_as_process(self):
        self._init_dequantization_info()
        self._initialize_queues()

    def add_callback(self, callback: Callable):
        self.callback_list.append(callback)

    def start_process(self):
        # if self.is_initialized:
        #     return
        try:
            self.init_all_as_process()
            self.process = Thread(
                target=self.init_as_process,
                args=(self.input_queue, self.output_queue, self.thead_number),
            )
            self.thead_number += 1
            self.process.start()
            self.is_initialized = True
        except Exception as e:
            print(f"start process failed: {str(e)}")
            raise RuntimeError(f"start process failed: {str(e)}")

    def stop_process(self):
        # self.share_memory_manager.__del__()
        if not self.is_initialized:
            return

        try:
            # self.process.terminate()
            self.process.join()
            self.is_initialized = False
        except Exception as e:
            print(f"stop process failed: {str(e)}")

    def __enter__(self):
        self.infer_ctx = self.infer.__enter__()
        self.activater.__enter__()
        return self.infer_ctx

    def __exit__(self, exc_type, exc_value, traceback):
        self.infer_ctx.__exit__(exc_type, exc_value, traceback)
        self.activater.__exit__(exc_type, exc_value, traceback)
        del self.activater
        del self.infer_ctx

    def get_input_info(self) -> List[NodeInfo]:
        input_infos = self.hef.get_input_stream_infos()
        res = []
        for info in input_infos:
            node = NodeInfo(info)
            res.append(node)
        return res

    def get_output_info(self) -> List[NodeInfo]:
        output_infos = self.hef.get_output_stream_infos()
        res = []
        for info in output_infos:
            node = NodeInfo(info)
            res.append(node)
        return res

    def init_output_quant_info(self):
        self.output_quant_info = {}
        for info in self.hef.get_output_stream_infos():

            self.output_quant_info[info.name] = {
                "qp_scale": info.quant_info.qp_scale,
                "qp_zero_point": info.quant_info.qp_zp,
            }

    def __del__(self):
        if hasattr(self, "infer_ctx"):
            self.infer_ctx.__exit__(None, None, None)
        if hasattr(self, "activater"):
            self.activater.__exit__(None, None, None)

    def pre_init(self):
        if not self.inited_predict_flag:
            self.init_as_predict()
        if not hasattr(self, "infer_ctx"):
            self.__enter__()

    def as_process_inference(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.is_initialized:
            self.start_process()
        shm_info = self.share_memory_manager.write(image, name=f"image_")
        if shm_info:
            self.input_queue.put(shm_info)
        else:
            logger.error("write share memory failed")
            return
        shm_info = self.output_queue.get()
        results = self.share_memory_manager.read_dict(shm_info)

        return results

    def inference(self, input_data) -> np.ndarray:
        output = self.infer_ctx.infer(input_data)
        output = self.dequantization(output)
        return output


if __name__ == "__main__":
    config_path0 = "hailo_ocr/configs/config.yaml"
    config_path1 = "hailo_ocr/configs/config_back.yaml"
    base_inference0 = BaseInference(config_path0)
    # base_inference1 = BaseInference(config_path1)
    base_inference0.start_process()
    # base_inference1.start_process()

    image0 = np.random.randint(0, 255, (40, 48, 320, 3), dtype=np.uint8)
    image1 = np.random.randint(0, 255, (1, 640, 320, 3), dtype=np.uint8)
    number = 40
    for _ in range(number):
        # with Timer("as_process_inference1"):
        #     base_inference1.as_process_inference(image1)
        with Timer("as_process_inference0"):
            base_inference0.as_process_inference(image0)

    # base_inference0.stop_process()
    # base_inference1.stop_process()
