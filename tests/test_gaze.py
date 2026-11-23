from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.face.gaze import GazeEstimator


def test_unavailable_returns_none():
    est = GazeEstimator.__new__(GazeEstimator)
    est._estimator = None
    est._available = False
    assert est.is_available() is False
    assert est.estimate(np.zeros((10, 10, 3), dtype=np.uint8)) is None


def test_available_with_mock_returns_angles():
    est = GazeEstimator.__new__(GazeEstimator)
    est._estimator = MagicMock()
    est._available = True
    # MobileGaze 返回 GazeResult，有 .yaw 和 .pitch（弧度）
    est._estimator.estimate.return_value = SimpleNamespace(yaw=0.1, pitch=-0.05)

    out = est.estimate(np.zeros((10, 10, 3), dtype=np.uint8))
    assert out is not None
    assert "yaw" in out and "pitch" in out
    assert abs(out["yaw"] - float(np.degrees(0.1))) < 1e-6
    assert abs(out["pitch"] - float(np.degrees(-0.05))) < 1e-6


def test_empty_crop_returns_none():
    est = GazeEstimator.__new__(GazeEstimator)
    est._estimator = MagicMock()
    est._available = True
    assert est.estimate(np.zeros((0, 0, 3), dtype=np.uint8)) is None