from dataclasses import dataclass
from hailo_toolbox.inference import load_model


@dataclass
class ModelsZoo:

    # Classification models
    mobilenetv1: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/mobilenet_v1.hef"
    )
    resnet18: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/resnet_v1_18.hef"
    )
    # Detection models
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

    # Segmentation models
    yolov8nseg: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n_seg.hef"
    )
    yolov8sseg: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_seg.hef"
    )
    yolov8mseg: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m_seg.hef"
    )

    # Pose estimation models
    yolov8spose: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_pose.hef"
    )
    yolov8mpose: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m_pose.hef"
    )

    # depth estimation models
    fast_depth: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/fast_depth.hef"
    )
    scdepthv3: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scdepthv3.hef"
    )

    # hand landmark models
    hand_landmark: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/hand_landmark_lite.hef"
    )

    # super resolution models
    real_esrgan: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/real_esrgan_x2.hef"
    )

    # face detection models
    scrfd_10g: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_10g.hef"
    )
    scrfd_2_5g: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_2.5g.hef"
    )
    scrfd_500m: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_500m.hef"
    )

    retinaface_mbnet: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/retinaface_mobilenet_v1.hef"
    )

    # face recognition models
    arcface_mbnet: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/arcface_mobilefacenet.hef"
    )
    arcface_r50: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/arcface_r50.hef"
    )
    lprnet: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/lprnet.hef"
    )

    # facial landmark models
    tddfa: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tddfa_mobilenet_v1.hef"
    )

    # person re-identification models
    osnet_x1: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/osnet_x1_0.hef"
    )
    repvgg_a0: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/repvgg_a0_person_reid_512.hef"
    )

    # image denoise models
    dncnn3: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/dncnn3.hef"
    )
    dncnn_color_blind: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/dncnn_color_blind.hef"
    )

    # low light enhancement models
    zero_dce: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/zero_dce.hef"
    )
    zero_dce_pp: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/zero_dce_pp.hef"
    )

    # text image retrieval models
    clip_vit_l: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_text_encoder_vit_l_14_laion2B.hef"
    )

    # video classification models
    r3d_18: str = (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/r3d_18.hef"
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
    def load_yolov8spose(cls):
        return load_model(model_name="yolov8spose", model_path=cls.yolov8spose)

    @classmethod
    def load_yolov8mpose(cls):
        return load_model(model_name="yolov8mpose", model_path=cls.yolov8mpose)

    @classmethod
    def load_mobilenetv1(cls):
        return load_model(model_name="mobilenetv1", model_path=cls.mobilenetv1)

    @classmethod
    def load_resnet18(cls):
        return load_model(model_name="resnet18", model_path=cls.resnet18)

    @classmethod
    def load_fast_depth(cls):
        return load_model(model_name="fast_depth", model_path=cls.fast_depth)

    @classmethod
    def load_scdepthv3(cls):
        return load_model(model_name="scdepthv3", model_path=cls.scdepthv3)

    @classmethod
    def load_hand_landmark(cls):
        return load_model(model_name="hand_landmark", model_path=cls.hand_landmark)

    @classmethod
    def load_real_esrgan(cls):
        return load_model(model_name="real_esrgan", model_path=cls.real_esrgan)

    @classmethod
    def load_scrfd_10g(cls):
        return load_model(model_name="scrfd_10g", model_path=cls.scrfd_10g)

    @classmethod
    def load_scrfd_2_5g(cls):
        return load_model(model_name="scrfd_2_5g", model_path=cls.scrfd_2_5g)

    @classmethod
    def load_scrfd_500m(cls):
        return load_model(model_name="scrfd_500m", model_path=cls.scrfd_500m)

    # @classmethod
    # def load_retinaface_mbnet(cls):
    #     return load_model(model_name="retinaface_mbnet", model_path=cls.retinaface_mbnet)

    @classmethod
    def load_arcface_mbnet(cls):
        return load_model(model_name="arcface_mbnet", model_path=cls.arcface_mbnet)

    @classmethod
    def load_arcface_r50(cls):
        return load_model(model_name="arcface_r50", model_path=cls.arcface_r50)

    @classmethod
    def load_lprnet(cls):
        return load_model(model_name="lprnet", model_path=cls.lprnet)

    @classmethod
    def load_tddfa(cls):
        return load_model(model_name="tddfa", model_path=cls.tddfa)

    @classmethod
    def load_osnet_x1(cls):
        return load_model(model_name="osnet_x1", model_path=cls.osnet_x1)

    @classmethod
    def load_repvgg_a0(cls):
        return load_model(model_name="repvgg_a0", model_path=cls.repvgg_a0)

    @classmethod
    def load_dncnn3(cls):
        return load_model(model_name="dncnn3", model_path=cls.dncnn3)

    @classmethod
    def load_dncnn_color_blind(cls):
        return load_model(
            model_name="dncnn_color_blind", model_path=cls.dncnn_color_blind
        )

    @classmethod
    def load_zero_dce(cls):
        return load_model(model_name="zero_dce", model_path=cls.zero_dce)

    @classmethod
    def load_zero_dce_pp(cls):
        return load_model(model_name="zero_dce_pp", model_path=cls.zero_dce_pp)

    @classmethod
    def load_clip_vit_l(cls):
        return load_model(model_name="clip_vit_l", model_path=cls.clip_vit_l)

    @classmethod
    def load_r3d_18(cls):
        return load_model(model_name="r3d_18", model_path=cls.r3d_18)
