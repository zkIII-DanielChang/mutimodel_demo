"""基于面部关键点计算 EAR、MAR 和眨眼计数。"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe Face Landmarker 标准索引
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14, 78, 308]


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _ear_for_eye(pts: Sequence[Tuple[float, float]]) -> float:
    # pts 顺序: p1..p6，对应左/右眼的 6 个点
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    if len(pts) != 6:
        return 0.0
    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    h = _dist(pts[0], pts[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_ear(
    landmarks: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """返回 (左眼 EAR, 右眼 EAR)。landmarks 为归一化坐标 (x, y)。"""
    if len(landmarks) <= max(max(LEFT_EYE), max(RIGHT_EYE)):
        return 0.0, 0.0
    left = [landmarks[i] for i in LEFT_EYE]
    right = [landmarks[i] for i in RIGHT_EYE]
    return _ear_for_eye(left), _ear_for_eye(right)


def compute_mar(landmarks: List[Tuple[float, float]]) -> float:
    """嘴部纵横比，张嘴时增大。"""
    if len(landmarks) <= max(MOUTH):
        return 0.0
    # 水平宽度：61-291
    h = _dist(landmarks[61], landmarks[291])
    if h < 1e-6:
        return 0.0
    # 垂直开口：13-14 为主，辅以 81-178 和 78-308
    v1 = _dist(landmarks[13], landmarks[14])
    v2 = _dist(landmarks[81], landmarks[178])
    v3 = _dist(landmarks[78], landmarks[308])
    return (v1 + v2 + v3) / (3.0 * h)


class BlinkCounter:
    def __init__(self, ear_threshold: float = 0.21):
        self.ear_threshold = ear_threshold
        self._closed = False
        self.count = 0

    def update(self, ear: float) -> bool:
        """返回本帧是否发生一次眨眼（闭→开）。"""
        if ear < self.ear_threshold:
            self._closed = True
            return False
        if self._closed:
            self._closed = False
            self.count += 1
            return True
        return False