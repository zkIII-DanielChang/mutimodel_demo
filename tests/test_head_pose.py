import numpy as np

from src.face.head_pose import HeadPoseEstimator


def _front_landmarks(w=640, h=480):
    cx, cy = w / 2, h / 2
    return np.array(
        [
            [cx - 60, cy - 40],
            [cx + 60, cy - 40],
            [cx, cy],
            [cx - 40, cy + 50],
            [cx + 40, cy + 50],
        ],
        dtype=np.float64,
    )


def test_estimate_returns_three_angles():
    est = HeadPoseEstimator()
    angles = est.estimate(_front_landmarks(), (480, 640, 3))
    assert angles is not None
    assert len(angles) == 3
    assert all(isinstance(a, float) for a in angles)


def test_estimate_returns_none_for_missing_landmarks():
    est = HeadPoseEstimator()
    assert est.estimate(None, (480, 640, 3)) is None


def test_estimate_returns_none_for_too_few_points():
    est = HeadPoseEstimator()
    pts = np.array([[1, 2], [3, 4]], dtype=np.float64)
    assert est.estimate(pts, (480, 640, 3)) is None