"""
Hailo Toolbox Model Library
===========================

This module provides pre-trained Hailo models organized by task type.

Usage Example:
    ```python
    from hailo_toolbox.models import ModelsZoo
    import numpy as np

    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Load detection model
    model = ModelsZoo.detection.yolov8n()
    results = model.predict(test_image)

    # Process results
    for result in results:
        boxes = result.get_boxes()      # Bounding boxes
        scores = result.get_scores()    # Confidence scores
        class_ids = result.get_class_ids()  # Class IDs
        print(f"Detected {len(result)} objects")
    ```

Detailed Documentation: docs/MODEL_EXAMPLES.md
Complete Examples: examples/model_examples.py
"""

from dataclasses import dataclass
from hailo_toolbox.inference import load_model


class ClassificationModels:
    """Image Classification Models

    Used to identify the main object categories in images. Returns Top5 prediction results.

    Usage Example:
        ```python
        model = ModelsZoo.classification.mobilenetv1()
        results = model.predict(image)

        for result in results:
            class_name = result.get_class_name()        # Top class name
            confidence = result.get_score()             # Confidence score
            top5_names = result.get_top_5_class_names() # Top5 class names
        ```

    Result Methods:
        - get_class_name(): Get top class name
        - get_score(): Get top confidence score
        - get_class_id(): Get top class ID
        - get_top_5_class_names(): Get Top5 class names
        - get_top_5_scores(): Get Top5 scores
    """

    @staticmethod
    def mobilenetv1(arch="hailo8"):
        """Load MobileNetV1 image classification model

        Lightweight classification model suitable for mobile devices and embedded applications.

        Args:
            arch: Target architecture (default: "hailo8")

        Returns:
            Model instance that can call predict() method for inference
        """
        return load_model(
            model_name="mobilenetv1",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/mobilenet_v1.hef",
        )

    @staticmethod
    def resnet18(arch="hailo8"):
        """Load ResNet18 image classification model

        Classic residual network classification model that balances accuracy and speed.

        Args:
            arch: Target architecture (default: "hailo8")

        Returns:
            Model instance that can call predict() method for inference
        """
        return load_model(
            model_name="resnet18",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/resnet_v1_18.hef",
        )


class DetectionModels:
    """Object Detection Models

    Used to detect multiple objects in images, returns bounding boxes, confidence scores and class IDs.

    Usage Example:
        ```python
        model = ModelsZoo.detection.yolov8n()
        results = model.predict(image)

        for result in results:
            boxes = result.get_boxes()          # Bounding boxes [x1, y1, x2, y2]
            scores = result.get_scores()        # Confidence scores
            class_ids = result.get_class_ids()  # Class IDs
            print(f"Detected {len(result)} objects")
        ```

    Result Methods:
        - get_boxes(): Get all bounding boxes
        - get_scores(): Get all confidence scores
        - get_class_ids(): Get all class IDs
        - len(result): Get number of detected objects
        - Supports iteration: for detection in result:
    """

    @staticmethod
    def yolov8n(arch="hailo8"):
        """Load YOLOv8n object detection model

        The smallest YOLOv8 model, fastest speed, suitable for real-time applications.

        Args:
            arch: Target architecture (default: "hailo8")

        Returns:
            Model instance that can call predict() method for inference
        """
        return load_model(
            model_name="yolov8det",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8n.hef",
        )

    @staticmethod
    def yolov8s(arch="hailo8"):
        """Load YOLOv8s object detection model"""
        return load_model(
            model_name="yolov8det",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8s.hef",
        )

    @staticmethod
    def yolov8m(arch="hailo8"):
        """Load YOLOv8m object detection model"""
        return load_model(
            model_name="yolov8det",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8m.hef",
        )

    @staticmethod
    def yolov8l(arch="hailo8"):
        """Load YOLOv8l object detection model"""
        return load_model(
            model_name="yolov8det",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8l.hef",
        )

    @staticmethod
    def yolov8x(arch="hailo8"):
        """Load YOLOv8x object detection model"""
        return load_model(
            model_name="yolov8det",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8x.hef",
        )


class SegmentationModels:
    """Image Segmentation Models

    Used for pixel-level image segmentation, returns segmentation masks, bounding boxes and class IDs.

    Usage Example:
        ```python
        model = ModelsZoo.segmentation.yolov8n_seg()
        results = model.predict(image)

        for result in results:
            masks = result.masks                    # Segmentation masks
            boxes = result.get_boxes_xyxy()         # Bounding boxes
            scores = result.get_scores()            # Confidence scores
        ```

    Result Methods:
        - masks: Segmentation mask arrays
        - get_boxes_xyxy(): Get bounding boxes
        - get_scores(): Get confidence scores
        - get_class_ids(): Get class IDs
    """

    @staticmethod
    def yolov8n_seg(arch="hailo8"):
        """Load YOLOv8n image segmentation model"""
        return load_model(
            model_name="yolov8nseg",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8n_seg.hef",
        )

    @staticmethod
    def yolov8s_seg(arch="hailo8"):
        """Load YOLOv8s image segmentation model"""
        return load_model(
            model_name="yolov8sseg",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8s_seg.hef",
        )

    @staticmethod
    def yolov8m_seg(arch="hailo8"):
        """Load YOLOv8m image segmentation model"""
        return load_model(
            model_name="yolov8mseg",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8m_seg.hef",
        )


class PoseEstimationModels:
    """Pose Estimation Models

    Used to detect human keypoints, returns keypoint coordinates, human confidence and bounding boxes.

    Usage Example:
        ```python
        model = ModelsZoo.pose_estimation.yolov8s_pose()
        results = model.predict(image)

        for result in results:
            for person in result:
                keypoints = person.get_keypoints()      # Keypoint coordinates
                score = person.get_score()              # Human confidence
        ```

    Result Methods:
        - get_keypoints(): Get keypoint coordinates
        - get_score(): Get human confidence
        - get_boxes(): Get human bounding boxes
        - get_joint_scores(): Get joint confidence scores
    """

    @staticmethod
    def yolov8s_pose(arch="hailo8"):
        """Load YOLOv8s pose estimation model"""
        return load_model(
            model_name="yolov8spose",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8s_pose.hef",
        )

    @staticmethod
    def yolov8m_pose(arch="hailo8"):
        """Load YOLOv8m pose estimation model"""
        return load_model(
            model_name="yolov8mpose",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/yolov8m_pose.hef",
        )


class DepthEstimationModels:
    """Depth Estimation Models

    Used to estimate depth information of images, returns depth maps.

    Usage Example:
        ```python
        model = ModelsZoo.depth_estimation.fast_depth()
        results = model.predict(image)

        for result in results:
            depth_map = result.get_depth()              # Raw depth map
            depth_normalized = result.get_depth_normalized() # Normalized depth map
        ```

    Result Methods:
        - get_depth(): Get raw depth map
        - get_depth_normalized(): Get normalized depth map (suitable for saving)
        - get_original_shape(): Get original image size
    """

    @staticmethod
    def fast_depth(arch="hailo8"):
        """Load FastDepth depth estimation model"""
        return load_model(
            model_name="fast_depth",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/fast_depth.hef",
        )

    @staticmethod
    def scdepthv3(arch="hailo8"):
        """Load SCDepthV3 depth estimation model"""
        return load_model(
            model_name="scdepthv3",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/scdepthv3.hef",
        )


class HandLandmarkModels:
    """Hand Landmark Detection Models

    Used to detect hand keypoints, returns hand keypoint coordinates.

    Usage Example:
        ```python
        model = ModelsZoo.hand_landmark_detection.hand_landmark()
        results = model.predict(image)

        for result in results:
            landmarks = result.get_landmarks()  # Hand keypoint coordinates
        ```

    Result Methods:
        - get_landmarks(): Get hand keypoint coordinates
    """

    @staticmethod
    def hand_landmark(arch="hailo8"):
        """Load hand landmark detection model"""
        return load_model(
            model_name="hand_landmark",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/hand_landmark_lite.hef",
        )


class SuperResolutionModels:
    """Super Resolution Models

    Used to improve image resolution, returns enhanced high-resolution images.

    Usage Example:
        ```python
        model = ModelsZoo.super_resolution.real_esrgan()
        results = model.predict(image)

        for result in results:
            enhanced_image = result.get_image()         # Enhanced image
        ```

    Result Methods:
        - get_image(): Get enhanced image
        - get_original_shape(): Get original image size
    """

    @staticmethod
    def real_esrgan(arch="hailo8"):
        """Load Real-ESRGAN super resolution model"""
        return load_model(
            model_name="real_esrgan",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/real_esrgan_x2.hef",
        )


class FaceDetectionModels:
    """Face Detection Models

    Used to detect faces in images, returns bounding boxes, confidence scores and facial landmarks.

    Usage Example:
        ```python
        model = ModelsZoo.face_detection.scrfd_10g()
        results = model.predict(image)

        for result in results:
            boxes = result.get_boxes(pixel_coords=True)     # Bounding boxes
            scores = result.get_scores()                    # Confidence scores
            landmarks = result.get_landmarks(pixel_coords=True) # Landmarks
        ```

    Result Methods:
        - get_boxes(pixel_coords=True/False): Get bounding boxes
        - get_scores(): Get confidence scores
        - get_landmarks(pixel_coords=True/False): Get landmarks
        - len(result): Get number of faces
    """

    @staticmethod
    def scrfd_10g(arch="hailo8"):
        """Load SCRFD-10G face detection model"""
        return load_model(
            model_name="scrfd_10g",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/scrfd_10g.hef",
        )

    @staticmethod
    def scrfd_2_5g(arch="hailo8"):
        """Load SCRFD-2.5G face detection model"""
        return load_model(
            model_name="scrfd_2_5g",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/scrfd_2.5g.hef",
        )

    @staticmethod
    def scrfd_500m(arch="hailo8"):
        """Load SCRFD-500M face detection model"""
        return load_model(
            model_name="scrfd_500m",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/scrfd_500m.hef",
        )

    @staticmethod
    def retinaface_mbnet(arch="hailo8"):
        """Load RetinaFace MobileNet face detection model"""
        return load_model(
            model_name="retinaface_mbnet",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/retinaface_mobilenet_v1.hef",
        )


class FaceRecognitionModels:
    """Face Recognition Models

    Used to extract face feature vectors for face recognition and comparison.

    Usage Example:
        ```python
        model = ModelsZoo.face_recognition.arcface_mbnet()
        results = model.predict(image)

        for result in results:
            embeddings = result.get_embeddings()  # Feature vectors
        ```

    Result Methods:
        - get_embeddings(): Get face feature vectors
    """

    @staticmethod
    def arcface_mbnet(arch="hailo8"):
        """Load ArcFace MobileNet face recognition model"""
        return load_model(
            model_name="arcface_mbnet",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/arcface_mobilefacenet.hef",
        )

    @staticmethod
    def arcface_r50(arch="hailo8"):
        """Load ArcFace ResNet50 face recognition model"""
        return load_model(
            model_name="arcface_r50",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/arcface_r50.hef",
        )


class LicensePlateRecognitionModels:
    """License Plate Recognition Models

    Used to recognize license plate text content.

    Usage Example:
        ```python
        model = ModelsZoo.license_plate_recognition.lprnet()
        results = model.predict(image)

        for result in results:
            # Specific methods depend on actual implementation
            print(f"Result type: {type(result)}")
        ```
    """

    @staticmethod
    def lprnet(arch="hailo8"):
        """Load LPRNet license plate recognition model"""
        return load_model(
            model_name="lprnet",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/lprnet.hef",
        )


class FacialLandmarkModels:
    """Facial Landmark Detection Models

    Used to detect 68 facial landmarks for facial analysis and alignment.

    Usage Example:
        ```python
        model = ModelsZoo.facial_landmark.tddfa()
        results = model.predict(image)

        for result in results:
            landmarks = result.get_landmarks()          # 68 landmarks
        ```

    Result Methods:
        - get_landmarks(): Get 68 facial landmarks
        - get_original_shape(): Get original image size
    """

    @staticmethod
    def tddfa(arch="hailo8"):
        """Load TDDFA facial landmark detection model"""
        return load_model(
            model_name="tddfa",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/tddfa_mobilenet_v1.hef",
        )


class PersonReIDModels:
    """Person Re-identification Models

    Used to extract person feature vectors for person re-identification and tracking.

    Usage Example:
        ```python
        model = ModelsZoo.person_reid.osnet_x1()
        results = model.predict(image)

        for result in results:
            embeddings = result.get_embeddings()        # Feature vectors
        ```

    Result Methods:
        - get_embeddings(): Get person feature vectors
        - get_original_shape(): Get original image size
    """

    @staticmethod
    def osnet_x1(arch="hailo8"):
        """Load OSNet-X1 person re-identification model"""
        return load_model(
            model_name="osnet_x1",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/osnet_x1_0.hef",
        )

    @staticmethod
    def repvgg_a0(arch="hailo8"):
        """Load RepVGG-A0 person re-identification model"""
        return load_model(
            model_name="repvgg_a0",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/repvgg_a0_person_reid_512.hef",
        )


class ImageDenoiseModels:
    """Image Denoising Models

    Used to remove image noise, returns denoised clear images.

    Usage Example:
        ```python
        model = ModelsZoo.image_denoise.dncnn3()
        results = model.predict(image)

        for result in results:
            denoised_image = result.get_denoised_image() # Denoised image
        ```

    Result Methods:
        - get_denoised_image(): Get denoised image
        - get_original_shape(): Get original image size
    """

    @staticmethod
    def dncnn3(arch="hailo8"):
        """Load DnCNN3 image denoising model"""
        return load_model(
            model_name="dncnn3",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/dncnn3.hef",
        )

    @staticmethod
    def dncnn_color_blind(arch="hailo8"):
        """Load DnCNN color blind image denoising model"""
        return load_model(
            model_name="dncnn_color_blind",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/dncnn_color_blind.hef",
        )


class LowLightEnhancementModels:
    """Low Light Enhancement Models

    Used to enhance brightness and clarity of low-light images.

    Usage Example:
        ```python
        model = ModelsZoo.low_light_enhancement.zero_dce()
        results = model.predict(image)

        for result in results:
            enhanced_image = result.get_enhanced_image() # Enhanced image
        ```

    Result Methods:
        - get_enhanced_image(): Get enhanced image
        - get_original_shape(): Get original image size
    """

    @staticmethod
    def zero_dce(arch="hailo8"):
        """Load Zero-DCE low light enhancement model"""
        return load_model(
            model_name="zero_dce",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/zero_dce.hef",
        )

    @staticmethod
    def zero_dce_pp(arch="hailo8"):
        """Load Zero-DCE++ low light enhancement model"""
        return load_model(
            model_name="zero_dce_pp",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/{arch}/zero_dce_pp.hef",
        )


class TextImageRetrievalModels:
    """Text-Image Retrieval Models

    Used to extract image feature vectors for text-image retrieval tasks.

    Usage Example:
        ```python
        model = ModelsZoo.text_image_retrieval.clip_vit_l()
        results = model.predict(image)

        for result in results:
            # CLIP model returns image feature vectors
            print(f"Result type: {type(result)}")
        ```
    """

    @staticmethod
    def clip_vitb_16(arch="hailo8"):
        """Load CLIP ViT-L text-image retrieval model"""
        return load_model(
            model_name="clip_vitb_16",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/{arch}/clip_text_encoder_vitb_16.hef",
        )


class VideoClassificationModels:
    """Video Classification Models

    Used to classify video actions, requires multi-frame input.

    Usage Example:
        ```python
        model = ModelsZoo.video_classification.r3d_18()
        results = model.predict(video_frames)  # Multi-frame input

        for result in results:
            class_names = result.get_class_name_top5()  # Top5 class names
            scores = result.get_score_top5()            # Top5 scores
        ```

    Result Methods:
        - get_class_name_top5(): Get Top5 class names
        - get_class_index_top5(): Get Top5 class indices
        - get_score_top5(): Get Top5 scores
        - get_original_shape(): Get original shape
    """

    @staticmethod
    def r3d_18(arch="hailo8"):
        """Load R3D-18 video classification model"""
        return load_model(
            model_name="r3d_18",
            model_path=f"https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/{arch}/r3d_18.hef",
        )


class ModelsZoo:
    """Hailo Model Zoo - Pre-trained models organized by task type"""

    # Models for different task types
    classification = ClassificationModels()
    detection = DetectionModels()
    segmentation = SegmentationModels()
    pose_estimation = PoseEstimationModels()
    depth_estimation = DepthEstimationModels()
    hand_landmark_detection = HandLandmarkModels()
    super_resolution = SuperResolutionModels()
    face_detection = FaceDetectionModels()
    face_recognition = FaceRecognitionModels()
    license_plate_recognition = LicensePlateRecognitionModels()
    facial_landmark = FacialLandmarkModels()
    person_reid = PersonReIDModels()
    image_denoise = ImageDenoiseModels()
    low_light_enhancement = LowLightEnhancementModels()
    text_image_retrieval = TextImageRetrievalModels()
    video_classification = VideoClassificationModels()

    # Backward compatibility class methods
    @classmethod
    def get_model_url(cls, model_name: str):
        """Get model URL by model name (backward compatibility)"""
        # Model name to URL mapping
        model_urls = {
            "mobilenetv1": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/mobilenet_v1.hef",
            "resnet18": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/resnet_v1_18.hef",
            "yolov8ndet": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hef",
            "yolov8sdet": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s.hef",
            "yolov8mdet": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m.hef",
            "yolov8ldet": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8l.hef",
            "yolov8xdet": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8x.hef",
            "yolov8nseg": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n_seg.hef",
            "yolov8sseg": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_seg.hef",
            "yolov8mseg": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m_seg.hef",
            "yolov8spose": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_pose.hef",
            "yolov8mpose": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8m_pose.hef",
            "fast_depth": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/fast_depth.hef",
            "scdepthv3": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scdepthv3.hef",
            "hand_landmark": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/hand_landmark_lite.hef",
            "real_esrgan": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/real_esrgan_x2.hef",
            "scrfd_10g": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_10g.hef",
            "scrfd_2_5g": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_2.5g.hef",
            "scrfd_500m": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/scrfd_500m.hef",
            "retinaface_mbnet": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/retinaface_mobilenet_v1.hef",
            "arcface_mbnet": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/arcface_mobilefacenet.hef",
            "arcface_r50": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/arcface_r50.hef",
            "lprnet": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/lprnet.hef",
            "tddfa": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/tddfa_mobilenet_v1.hef",
            "osnet_x1": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/osnet_x1_0.hef",
            "repvgg_a0": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/repvgg_a0_person_reid_512.hef",
            "dncnn3": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/dncnn3.hef",
            "dncnn_color_blind": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/dncnn_color_blind.hef",
            "zero_dce": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/zero_dce.hef",
            "zero_dce_pp": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/zero_dce_pp.hef",
            "clip_vit_l": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/clip_text_encoder_vit_l_14_laion2B.hef",
            "r3d_18": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/r3d_18.hef",
        }
        return model_urls.get(model_name)

    # Backward compatibility properties
    @property
    def mobilenetv1(self):
        return self.get_model_url("mobilenetv1")

    @property
    def resnet18(self):
        return self.get_model_url("resnet18")

    @property
    def yolov8ndet(self):
        return self.get_model_url("yolov8ndet")

    @property
    def yolov8sdet(self):
        return self.get_model_url("yolov8sdet")

    @property
    def yolov8mdet(self):
        return self.get_model_url("yolov8mdet")

    @property
    def yolov8ldet(self):
        return self.get_model_url("yolov8ldet")

    @property
    def yolov8xdet(self):
        return self.get_model_url("yolov8xdet")

    @property
    def yolov8nseg(self):
        return self.get_model_url("yolov8nseg")

    @property
    def yolov8sseg(self):
        return self.get_model_url("yolov8sseg")

    @property
    def yolov8mseg(self):
        return self.get_model_url("yolov8mseg")

    @property
    def yolov8spose(self):
        return self.get_model_url("yolov8spose")

    @property
    def yolov8mpose(self):
        return self.get_model_url("yolov8mpose")

    @property
    def fast_depth(self):
        return self.get_model_url("fast_depth")

    @property
    def scdepthv3(self):
        return self.get_model_url("scdepthv3")

    @property
    def hand_landmark(self):
        return self.get_model_url("hand_landmark")

    @property
    def real_esrgan(self):
        return self.get_model_url("real_esrgan")

    @property
    def scrfd_10g(self):
        return self.get_model_url("scrfd_10g")

    @property
    def scrfd_2_5g(self):
        return self.get_model_url("scrfd_2_5g")

    @property
    def scrfd_500m(self):
        return self.get_model_url("scrfd_500m")

    @property
    def retinaface_mbnet(self):
        return self.get_model_url("retinaface_mbnet")

    @property
    def arcface_mbnet(self):
        return self.get_model_url("arcface_mbnet")

    @property
    def arcface_r50(self):
        return self.get_model_url("arcface_r50")

    @property
    def lprnet(self):
        return self.get_model_url("lprnet")

    @property
    def tddfa(self):
        return self.get_model_url("tddfa")

    @property
    def osnet_x1(self):
        return self.get_model_url("osnet_x1")

    @property
    def repvgg_a0(self):
        return self.get_model_url("repvgg_a0")

    @property
    def dncnn3(self):
        return self.get_model_url("dncnn3")

    @property
    def dncnn_color_blind(self):
        return self.get_model_url("dncnn_color_blind")

    @property
    def zero_dce(self):
        return self.get_model_url("zero_dce")

    @property
    def zero_dce_pp(self):
        return self.get_model_url("zero_dce_pp")

    @property
    def clip_vit_l(self):
        return self.get_model_url("clip_vit_l")

    @property
    def r3d_18(self):
        return self.get_model_url("r3d_18")

    # Backward compatibility loading methods
    @classmethod
    def load_yolov8ndet(cls):
        """Load YOLOv8n object detection model"""
        return cls.detection.yolov8n()

    @classmethod
    def load_yolov8mseg(cls):
        """Load YOLOv8m image segmentation model"""
        return cls.segmentation.yolov8m_seg()

    @classmethod
    def load_yolov8sseg(cls):
        """Load YOLOv8s image segmentation model"""
        return cls.segmentation.yolov8s_seg()

    @classmethod
    def load_yolov8spose(cls):
        """Load YOLOv8s pose estimation model"""
        return cls.pose_estimation.yolov8s_pose()

    @classmethod
    def load_yolov8mpose(cls):
        """Load YOLOv8m pose estimation model"""
        return cls.pose_estimation.yolov8m_pose()

    @classmethod
    def load_mobilenetv1(cls):
        """Load MobileNetV1 image classification model"""
        return cls.classification.mobilenetv1()

    @classmethod
    def load_resnet18(cls):
        """Load ResNet18 image classification model"""
        return cls.classification.resnet18()

    @classmethod
    def load_fast_depth(cls):
        """Load FastDepth depth estimation model"""
        return cls.depth_estimation.fast_depth()

    @classmethod
    def load_scdepthv3(cls):
        """Load SCDepthV3 depth estimation model"""
        return cls.depth_estimation.scdepthv3()

    @classmethod
    def load_hand_landmark(cls):
        """Load hand landmark detection model"""
        return cls.hand_landmark_detection.hand_landmark()

    @classmethod
    def load_real_esrgan(cls):
        """Load Real-ESRGAN super resolution model"""
        return cls.super_resolution.real_esrgan()

    @classmethod
    def load_scrfd_10g(cls):
        """Load SCRFD-10G face detection model"""
        return cls.face_detection.scrfd_10g()

    @classmethod
    def load_scrfd_2_5g(cls):
        """Load SCRFD-2.5G face detection model"""
        return cls.face_detection.scrfd_2_5g()

    @classmethod
    def load_scrfd_500m(cls):
        """Load SCRFD-500M face detection model"""
        return cls.face_detection.scrfd_500m()

    @classmethod
    def load_arcface_mbnet(cls):
        """Load ArcFace MobileNet face recognition model"""
        return cls.face_recognition.arcface_mbnet()

    @classmethod
    def load_arcface_r50(cls):
        """Load ArcFace ResNet50 face recognition model"""
        return cls.face_recognition.arcface_r50()

    @classmethod
    def load_lprnet(cls):
        """Load LPRNet license plate recognition model"""
        return cls.license_plate_recognition.lprnet()

    @classmethod
    def load_tddfa(cls):
        """Load TDDFA facial landmark detection model"""
        return cls.facial_landmark.tddfa()

    @classmethod
    def load_osnet_x1(cls):
        """Load OSNet-X1 person re-identification model"""
        return cls.person_reid.osnet_x1()

    @classmethod
    def load_repvgg_a0(cls):
        """Load RepVGG-A0 person re-identification model"""
        return cls.person_reid.repvgg_a0()

    @classmethod
    def load_dncnn3(cls):
        """Load DnCNN3 image denoising model"""
        return cls.image_denoise.dncnn3()

    @classmethod
    def load_dncnn_color_blind(cls):
        """Load DnCNN color blind image denoising model"""
        return cls.image_denoise.dncnn_color_blind()

    @classmethod
    def load_zero_dce(cls):
        """Load Zero-DCE low light enhancement model"""
        return cls.low_light_enhancement.zero_dce()

    @classmethod
    def load_zero_dce_pp(cls):
        """Load Zero-DCE++ low light enhancement model"""
        return cls.low_light_enhancement.zero_dce_pp()

    @classmethod
    def load_clip_vit_l(cls):
        """Load CLIP ViT-L text-image retrieval model"""
        return cls.text_image_retrieval.clip_vit_l()

    @classmethod
    def load_r3d_18(cls):
        """Load R3D-18 video classification model"""
        return cls.video_classification.r3d_18()

    @classmethod
    def list_models_by_task(cls, task_type: str = None):
        """List available model methods by task type

        Args:
            task_type: Task type, such as 'classification', 'detection', 'segmentation', etc.
                      If None, list all task types

        Returns:
            dict: Dictionary containing task types and corresponding model methods
        """
        task_mapping = {
            "classification": cls.classification,
            "detection": cls.detection,
            "segmentation": cls.segmentation,
            "pose_estimation": cls.pose_estimation,
            "depth_estimation": cls.depth_estimation,
            "hand_landmark": cls.hand_landmark_detection,
            "super_resolution": cls.super_resolution,
            "face_detection": cls.face_detection,
            "face_recognition": cls.face_recognition,
            "license_plate_recognition": cls.license_plate_recognition,
            "facial_landmark": cls.facial_landmark,
            "person_reid": cls.person_reid,
            "image_denoise": cls.image_denoise,
            "low_light_enhancement": cls.low_light_enhancement,
            "text_image_retrieval": cls.text_image_retrieval,
            "video_classification": cls.video_classification,
        }

        if task_type:
            if task_type in task_mapping:
                return {task_type: task_mapping[task_type]}
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        else:
            return task_mapping


if __name__ == "__main__":
    # Test new usage patterns
    print("=== New Usage Patterns ===")

    # Directly call model methods to load models
    print("1. Direct method calls to load models:")
    print("   Loading YOLOv8n detection model...")
    model = ModelsZoo.detection.yolov8n()
    print(f"   Model type: {type(model)}")

    print("\n2. List available models by task type:")
    detection_models = ModelsZoo.list_models_by_task("detection")
    detection_methods = [
        method
        for method in dir(detection_models["detection"])
        if not method.startswith("_")
    ]
    print(f"   Detection model methods: {detection_methods}")

    print("\n3. Backward compatibility test:")
    print("   Old way to load models:")
    old_model = ModelsZoo.load_yolov8ndet()
    print(f"   Model type: {type(old_model)}")
    ModelsZoo.detection.yolov8l()
    print("\n4. All task types:")
    all_tasks = ModelsZoo.list_models_by_task()
    for task_name, task_models in all_tasks.items():
        methods = [method for method in dir(task_models) if not method.startswith("_")]
        print(f"   {task_name}: {methods}")

    print("\n=== Usage Examples ===")
    print("# New usage pattern - direct method calls to load models")
    print("model = ModelsZoo.detection.yolov8n()  # Load YOLOv8n detection model")
    print(
        "model = ModelsZoo.segmentation.yolov8m_seg()  # Load YOLOv8m segmentation model"
    )
    print(
        "model = ModelsZoo.classification.mobilenetv1()  # Load MobileNetV1 classification model"
    )
    print("")
    print("# Backward compatibility - old usage patterns still work")
    print("model = ModelsZoo.load_yolov8ndet()  # Old way to load model")
    print("url = ModelsZoo().yolov8ndet  # Get model URL")
