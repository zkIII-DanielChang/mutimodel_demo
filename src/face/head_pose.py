"""基于 solvePnP 的头部姿态估计，输入为 5 点 landmarks。支持 pitch 零偏校准。"""

from typing import Any, List, Optional

import cv2
import numpy as np

from src.common.logger import get_logger

logger = get_logger("face.head_pose")

# RetinaFace 标准 5 点顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
# 坐标约定：X 向右，Y 向下（与图像一致），Z 朝向场景（与 OpenCV 相机坐标系一致）
MODEL_POINTS_5 = np.array(
    [
        [-225.0, -270.0, 135.0],   # 左眼
        [225.0, -270.0, 135.0],    # 右眼
        [0.0, 0.0, 0.0],           # 鼻尖
        [-150.0, 240.0, 125.0],    # 左嘴角
        [150.0, 240.0, 125.0],     # 右嘴角
    ],
    dtype=np.float64,
)


class HeadPoseEstimator:
    def __init__(self, use_approx_intrinsics: bool = True):
        self.use_approx_intrinsics = use_approx_intrinsics
        self._camera_matrix: Optional[np.ndarray] = None
        self._image_size: Optional[tuple] = None
        self._pitch_bias: float = 0.0

    def set_pitch_bias(self, bias: float) -> None:
        """设置 pitch 零偏，后续输出的 pitch 会减去该值。"""
        self._pitch_bias = float(bias)
        logger.info("pitch 零偏已设置: %.2f 度", self._pitch_bias)

    def get_pitch_bias(self) -> float:
        return self._pitch_bias

    def _get_camera_matrix(self, image_shape) -> np.ndarray:
        h, w = image_shape[:2]
        if self._camera_matrix is not None and self._image_size == (w, h):
            return self._camera_matrix
        focal = float(w)
        cx, cy = w / 2.0, h / 2.0
        self._camera_matrix = np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._image_size = (w, h)
        return self._camera_matrix

    @staticmethod
    def _to_5_points(landmarks: Any) -> Optional[np.ndarray]:
        if landmarks is None:
            return None
        pts = np.asarray(landmarks, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None
        if pts.shape[0] == 5:
            return pts
        if pts.shape[0] > 5:
            return pts[:5]
        return None

    def estimate(self, landmarks: Any, image_shape) -> Optional[List[float]]:
        """返回 [yaw, pitch, roll] 角度（度），pitch 已减去零偏。失败返回 None。"""
        image_points = self._to_5_points(landmarks)
        if image_points is None:
            return None

        camera_matrix = self._get_camera_matrix(image_shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_5,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error as e:
            logger.warning("solvePnP 失败: %s", e)
            return None

        if not ok:
            return None

        try:
            rmat, _ = cv2.Rodrigues(rvec)
        except cv2.error:
            return None

        sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
        if sy > 1e-6:
            pitch = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw = float(np.arctan2(-rmat[2, 0], sy))
            roll = float(np.arctan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = float(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw = float(np.arctan2(-rmat[2, 0], sy))
            roll = 0.0

        return [
            float(np.degrees(yaw)),
            float(np.degrees(pitch)) - self._pitch_bias,
            float(np.degrees(roll)),
        ]