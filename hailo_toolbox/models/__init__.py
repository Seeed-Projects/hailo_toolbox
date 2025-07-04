from dataclasses import dataclass
from hailo_toolbox.inference import load_model


@dataclass
class ModelsZoo:
    yolov8ndet: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hef"
    )
    yolov8sdet: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s.hef"
    )
    yolov8mdet: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m.hef"
    )
    yolov8ldet: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8l.hef"
    )
    yolov8xdet: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8x.hef"
    )
    yolov8nseg: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n_seg.hef"
    )
    yolov8sseg: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_seg.hef"
    )
    yolov8mseg: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m_seg.hef"
    )
    yolov8spose: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_pose.hef"
    )
    yolov8mpose: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m_pose.hef"
    )

    @classmethod
    def get_model(cls, model_name: str):
        return getattr(cls, model_name)

    @classmethod
    def load_yolov8ndet(cls):
        return load_model(model_name="yolov8det", model_path=cls.yolov8ndet)

    @classmethod
    def load_yolov8mseg(cls):
        return load_model(model_name="yolov8mseg", model_path=cls.yolov8mseg)

    @classmethod
    def load_yolov8sseg(cls):
        return load_model(model_name="yolov8sseg", model_path=cls.yolov8sseg)

    @classmethod
    def load_yolov8lseg(cls):
        return load_model(model_name="yolov8lseg", model_path=cls.yolov8lseg)

    @classmethod
    def load_yolov8spose(cls):
        return load_model(model_name="yolov8spose", model_path=cls.yolov8spose)

    @classmethod
    def load_yolov8mpose(cls):
        return load_model(model_name="yolov8mpose", model_path=cls.yolov8mpose)
