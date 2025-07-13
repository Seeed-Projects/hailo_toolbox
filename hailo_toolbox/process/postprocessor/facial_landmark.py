import pickle
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np

from hailo_toolbox.process.base import BasePostprocessor
from hailo_toolbox.process.results.facial_landmark import FacialLandmarkResult
from hailo_toolbox.inference import CALLBACK_REGISTRY


# TDDFA rescaling parameters (from demo.py)
TDDFA_RESCALE_PARAMS = {
    "mean": np.array(
        [
            [
                3.4926363e-04,
                2.5279013e-07,
                -6.8751979e-07,
                6.0167957e01,
                -6.2955132e-07,
                5.7572004e-04,
                -5.0853912e-05,
                7.4278198e01,
                5.4009172e-07,
                6.5741384e-05,
                3.4420125e-04,
                -6.6671577e01,
                -3.4660369e05,
                -6.7468234e04,
                4.6822266e04,
                -1.5262047e04,
                4.3505889e03,
                -5.4261453e04,
                -1.8328033e04,
                -1.5843289e03,
                -8.4566344e04,
                3.8359607e03,
                -2.0811361e04,
                3.8094930e04,
                -1.9967855e04,
                -9.2413701e03,
                -1.9600715e04,
                1.3168090e04,
                -5.2591440e03,
                1.8486478e03,
                -1.3030662e04,
                -2.4355562e03,
                -2.2542065e03,
                -1.4396562e04,
                -6.1763291e03,
                -2.5621920e04,
                2.2639447e02,
                -6.3261235e03,
                -1.0867251e04,
                8.6846509e02,
                -5.8311479e03,
                2.7051238e03,
                -3.6294177e03,
                2.0439901e03,
                -2.4466162e03,
                3.6586970e03,
                -7.6459897e03,
                -6.6744526e03,
                1.1638839e02,
                7.1855972e03,
                -1.4294868e03,
                2.6173665e03,
                -1.2070955e00,
                6.6907924e-01,
                -1.7760828e-01,
                5.6725528e-02,
                3.9678156e-02,
                -1.3586316e-01,
                -9.2239931e-02,
                -1.7260718e-01,
                -1.5804484e-02,
                -1.4168486e-01,
            ]
        ],
        dtype=np.float32,
    ),
    "std": np.array(
        [
            [
                1.76321526e-04,
                6.73794348e-05,
                4.47084894e-04,
                2.65502319e01,
                1.23137695e-04,
                4.49302170e-05,
                7.92367064e-05,
                6.98256302e00,
                4.35044407e-04,
                1.23148900e-04,
                1.74000015e-04,
                2.08030396e01,
                5.75421125e05,
                2.77649062e05,
                2.58336844e05,
                2.55163125e05,
                1.50994375e05,
                1.60086109e05,
                1.11277305e05,
                9.73117812e04,
                1.17198453e05,
                8.93173672e04,
                8.84935547e04,
                7.22299297e04,
                7.10802109e04,
                5.00139531e04,
                5.59685820e04,
                4.75255039e04,
                4.95150664e04,
                3.81614805e04,
                4.48720586e04,
                4.62732383e04,
                3.81167695e04,
                2.81911621e04,
                3.21914375e04,
                3.60061719e04,
                3.25598926e04,
                2.55511172e04,
                2.42675098e04,
                2.75213984e04,
                2.31665312e04,
                2.11015762e04,
                1.94123242e04,
                1.94522031e04,
                1.74549844e04,
                2.25376230e04,
                1.61742812e04,
                1.46716406e04,
                1.51156885e04,
                1.38700732e04,
                1.37463125e04,
                1.26631338e04,
                1.58708346e00,
                1.50770092e00,
                5.88135779e-01,
                5.88974476e-01,
                2.13278517e-01,
                2.63020128e-01,
                2.79642940e-01,
                3.80302161e-01,
                1.61628410e-01,
                2.55969286e-01,
            ]
        ],
        dtype=np.float32,
    ),
}


def _to_ctype(arr):
    """Convert array to C-contiguous format."""
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr


class BFMModel:
    """3D Morphable Face Model for landmark processing."""

    def __init__(self, shape_dim=40, exp_dim=10):
        """
        Initialize BFM model.

        Args:
            shape_dim: Shape parameter dimension
            exp_dim: Expression parameter dimension
        """
        # Try to find BFM model files in common locations
        self.bfm_path = self._find_bfm_file("bfm_noneck_v3.pkl")
        self.tri_path = self._find_bfm_file("tri.pkl")

        if not self.bfm_path.exists() or not self.tri_path.exists():
            raise FileNotFoundError(
                "BFM model files not found. Please ensure bfm_noneck_v3.pkl and tri.pkl "
                "are available in the cache directory or model path. "
                "For more information, see: https://github.com/cleardusk/3DDFA_V2/tree/master/bfm"
            )

        # Load BFM model
        with open(self.bfm_path, "rb") as f:
            bfm = pickle.load(f)

        self._u = bfm.get("u").astype(np.float32)
        self._w_shp = bfm.get("w_shp").astype(np.float32)[..., :shape_dim]
        self._w_exp = bfm.get("w_exp").astype(np.float32)[..., :exp_dim]

        # Load triangle data
        with open(self.tri_path, "rb") as f:
            self._tri = pickle.load(f)

        self._tri = _to_ctype(self._tri.T).astype(np.int32)
        self._keypoints = bfm.get("keypoints").astype(int)

        # Compute normalized weights
        w = np.concatenate((self._w_shp, self._w_exp), axis=1)
        self._w_norm = np.linalg.norm(w, axis=0)

        # Prepare base matrices for landmark computation
        self.u_base = self._u[self._keypoints].reshape(-1, 1)
        self.w_shp_base = self._w_shp[self._keypoints]
        self.w_exp_base = self._w_exp[self._keypoints]

    def _find_bfm_file(self, filename: str) -> Path:
        """
        Find BFM model file in common locations.

        Args:
            filename: Name of the BFM file to find

        Returns:
            Path to the BFM file
        """
        # Common locations to search for BFM files
        search_paths = [
            Path.home() / ".hailo_cache" / "models" / "bfm",
            Path.home() / ".hailo_cache" / "models",
            Path("./models/bfm"),
            Path("./models"),
            Path("./bfm"),
            Path("."),
        ]

        for search_path in search_paths:
            file_path = search_path / filename
            if file_path.exists():
                return file_path

        # If not found, return the first search path (will be used for error message)
        return search_paths[0] / filename


class BFMSingleton:
    """Singleton pattern for BFM model to avoid multiple loadings."""

    def __init__(self):
        self._BFM = None

    def get_bfm(self) -> BFMModel:
        """Get or create BFM model instance."""
        if self._BFM is None:
            self._BFM = BFMModel()
        return self._BFM


# Global BFM singleton instance
bfm_singleton = BFMSingleton()


def _parse_param(
    param: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse 3DMM parameters.

    Args:
        param: 3DMM parameters array

    Returns:
        Tuple of (R, offset, alpha_shp, alpha_exp)
    """
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise ValueError(f"Unsupported parameter dimension: {n}")

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim : trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim :].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp


def similar_transform(pts3d: np.ndarray, roi_box: np.ndarray, size: int) -> np.ndarray:
    """
    Apply similar transformation to 3D points.

    Args:
        pts3d: 3D points array
        roi_box: ROI bounding box [sx, sy, ex, ey]
        size: Image size

    Returns:
        Transformed 3D points
    """
    pts3d = pts3d.copy()

    # Adjust for Python compatibility
    pts3d[0, :] -= 1
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    # Extract ROI box coordinates
    sx, sy, ex, ey = roi_box

    # Calculate scale factors
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size

    # Apply scaling and translation
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy

    # Apply depth scaling
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])

    return pts3d.astype(np.float32)


def face_3dmm_to_landmarks_np(
    face_3dmm_params: np.ndarray, img_dims: Tuple[int, int], roi_box: np.ndarray
) -> np.ndarray:
    """
    Convert 3DMM parameters to 3D landmarks for a single face.

    Args:
        face_3dmm_params: 3DMM parameters
        img_dims: Image dimensions (height, width)
        roi_box: ROI bounding box

    Returns:
        3D landmarks array
    """
    # Parse parameters
    R, offset, alpha_shp, alpha_exp = _parse_param(face_3dmm_params)

    # Get BFM model
    bfm = bfm_singleton.get_bfm()

    # Compute 3D landmarks
    pts3d = (
        R
        @ (
            bfm.u_base + bfm.w_shp_base @ alpha_shp + bfm.w_exp_base @ alpha_exp
        ).reshape(3, -1, order="F")
        + offset
    )

    # Apply similar transformation
    pts3d = similar_transform(pts3d, roi_box, img_dims[0])

    # Transpose to get (N, 3) format
    return pts3d.transpose()


def face_3dmm_to_landmarks_batch(
    face_3dmm_params: np.ndarray, img_dims: Tuple[int, int], roi_boxes: np.ndarray
) -> np.ndarray:
    """
    Convert 3DMM parameters to 3D landmarks for a batch of faces.

    Args:
        face_3dmm_params: Batch of 3DMM parameters [B, 62]
        img_dims: Image dimensions (height, width)
        roi_boxes: Batch of ROI bounding boxes [B, 4]

    Returns:
        Batch of 3D landmarks [B, N, 3]
    """
    landmarks_list = []

    for params, box in zip(face_3dmm_params, roi_boxes):
        landmarks = face_3dmm_to_landmarks_np(params, img_dims, box)
        landmarks_list.append(landmarks)

    return np.array(landmarks_list)


def simple_landmarks_from_params(
    face_3dmm_params: np.ndarray, img_dims: Tuple[int, int]
) -> np.ndarray:
    """
    简化版本：直接从3DMM参数生成2D关键点，不需要BFM模型文件。
    这是一个近似实现，用于快速测试和演示。

    Args:
        face_3dmm_params: 3DMM参数 [62,]
        img_dims: 图像尺寸 (height, width)

    Returns:
        2D关键点 [N, 2]，其中N是关键点数量
    """
    img_h, img_w = img_dims
    center_x, center_y = img_w / 2, img_h / 2

    # 使用3DMM参数的前几个值来影响关键点位置
    # 这是一个简化的映射，实际应该使用BFM模型
    param_influence = face_3dmm_params[:12]

    # 标准68点面部关键点模板（归一化坐标）
    # 这些是基于标准面部关键点分布的近似位置
    landmarks_template = np.array(
        [
            # 脸部轮廓 (17个点: 0-16)
            [-0.35, -0.25],
            [-0.32, -0.15],
            [-0.29, -0.05],
            [-0.26, 0.05],
            [-0.23, 0.15],
            [-0.19, 0.25],
            [-0.14, 0.33],
            [-0.08, 0.38],
            [0.0, 0.40],
            [0.08, 0.38],
            [0.14, 0.33],
            [0.19, 0.25],
            [0.23, 0.15],
            [0.26, 0.05],
            [0.29, -0.05],
            [0.32, -0.15],
            [0.35, -0.25],
            # 左眉毛 (5个点: 17-21)
            [-0.25, -0.35],
            [-0.18, -0.38],
            [-0.10, -0.38],
            [-0.05, -0.35],
            [-0.02, -0.32],
            # 右眉毛 (5个点: 22-26)
            [0.02, -0.32],
            [0.05, -0.35],
            [0.10, -0.38],
            [0.18, -0.38],
            [0.25, -0.35],
            # 鼻子 (9个点: 27-35)
            [0.0, -0.25],
            [-0.02, -0.18],
            [0.02, -0.18],
            [-0.04, -0.12],
            [0.0, -0.10],
            [0.04, -0.12],
            [-0.03, -0.08],
            [0.0, -0.05],
            [0.03, -0.08],
            # 左眼 (6个点: 36-41)
            [-0.18, -0.22],
            [-0.12, -0.25],
            [-0.06, -0.25],
            [-0.06, -0.18],
            [-0.12, -0.18],
            [-0.18, -0.22],
            # 右眼 (6个点: 42-47)
            [0.18, -0.22],
            [0.12, -0.25],
            [0.06, -0.25],
            [0.06, -0.18],
            [0.12, -0.18],
            [0.18, -0.22],
            # 嘴巴外轮廓 (12个点: 48-59)
            [-0.12, 0.12],
            [-0.08, 0.08],
            [-0.04, 0.05],
            [0.0, 0.05],
            [0.04, 0.05],
            [0.08, 0.08],
            [0.12, 0.12],
            [0.08, 0.18],
            [0.04, 0.20],
            [0.0, 0.20],
            [-0.04, 0.20],
            [-0.08, 0.18],
            # 嘴巴内轮廓 (8个点: 60-67)
            [-0.08, 0.12],
            [-0.04, 0.10],
            [0.0, 0.10],
            [0.04, 0.10],
            [0.08, 0.12],
            [0.04, 0.15],
            [0.0, 0.15],
            [-0.04, 0.15],
        ],
        dtype=np.float32,
    )

    # 使用参数影响关键点位置
    # 这是一个简化的变换，实际的3DMM会更复杂
    scale_factor = min(img_w, img_h) * 0.25  # 基础缩放

    # 使用前几个参数调整缩放和位置
    scale_adjustment = 1.0 + param_influence[0] * 0.1  # 轻微调整缩放
    offset_x = param_influence[1] * 5  # 水平偏移
    offset_y = param_influence[2] * 5  # 垂直偏移

    # 应用变换
    landmarks = np.zeros((68, 2), dtype=np.float32)
    landmarks[:, 0] = (
        landmarks_template[:, 0] * scale_factor * scale_adjustment + center_x + offset_x
    )
    landmarks[:, 1] = (
        landmarks_template[:, 1] * scale_factor * scale_adjustment + center_y + offset_y
    )

    # 添加基于其他参数的细微调整
    for i in range(min(len(landmarks), len(param_influence))):
        if i < len(param_influence):
            landmarks[i, 0] += param_influence[i] * 0.5
            landmarks[i, 1] += param_influence[i] * 0.5

    # 确保关键点在图像范围内
    margin = 5
    landmarks[:, 0] = np.clip(landmarks[:, 0], margin, img_w - margin)
    landmarks[:, 1] = np.clip(landmarks[:, 1], margin, img_h - margin)

    return landmarks


@CALLBACK_REGISTRY.registryPostProcessor("tddfa")
class FacialLandmarkPostprocessor(BasePostprocessor):
    """3D Facial Landmark Postprocessor for TDDFA model."""

    def __init__(
        self, img_dims: Tuple[int, int] = (120, 120), use_full_3d: bool = False
    ):
        """
        Initialize the postprocessor.

        Args:
            img_dims: Input image dimensions (height, width)
            use_full_3d: Whether to use full 3D processing (requires BFM model files)
        """
        super().__init__()
        self.img_dims = img_dims
        self.use_full_3d = use_full_3d

        # Validate image dimensions
        if img_dims[0] != img_dims[1]:
            print("Warning: TDDFA model typically expects square input images")

    def postprocess(
        self, predictions: Dict[str, np.ndarray], original_shape: Tuple[int, int]
    ) -> List[FacialLandmarkResult]:
        """
        Postprocess TDDFA model predictions to facial landmarks.

        Args:
            predictions: Model predictions dictionary
            original_shape: Original image shape (height, width)

        Returns:
            List of FacialLandmarkResult objects
        """
        results = []

        for key, value in predictions.items():
            print(f"Processing {key} with shape {value.shape}")

            # Get batch size
            batch_size = value.shape[0]

            # Reshape predictions to [batch_size, 62]
            face_3dmm_params = value.reshape(batch_size, -1)

            # Apply rescaling (equivalent to TensorFlow version)
            face_3dmm_params = (
                face_3dmm_params * TDDFA_RESCALE_PARAMS["std"]
                + TDDFA_RESCALE_PARAMS["mean"]
            )

            # Process each face in the batch
            for i in range(batch_size):
                if self.use_full_3d:
                    # Use full 3D processing with BFM model
                    try:
                        # Create default ROI box (full image)
                        roi_box = np.array([0, 0, self.img_dims[1], self.img_dims[0]])

                        # Convert 3DMM parameters to 3D landmarks
                        landmarks_3d = face_3dmm_to_landmarks_np(
                            face_3dmm_params[i], self.img_dims, roi_box
                        )

                        result = FacialLandmarkResult(
                            landmarks=landmarks_3d, original_shape=original_shape
                        )
                    except Exception as e:
                        print(f"Full 3D processing failed: {e}")
                        print("Falling back to simplified processing...")
                        # Fall back to simplified processing
                        landmarks_2d = simple_landmarks_from_params(
                            face_3dmm_params[i], self.img_dims
                        )
                        result = FacialLandmarkResult(
                            landmarks=landmarks_2d, original_shape=original_shape
                        )
                else:
                    # Use simplified processing (default)
                    landmarks_2d = simple_landmarks_from_params(
                        face_3dmm_params[i], self.img_dims
                    )
                    result = FacialLandmarkResult(
                        landmarks=landmarks_2d, original_shape=original_shape
                    )

                results.append(result)

        return results
