import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.fingerprint.matcher import FingerprintMatcher


def _make_synthetic_fingerprint(seed: int = 0, size: int = 200) -> np.ndarray:
    """生成有纹理的合成指纹图像，用于测试 SIFT 提取和比对。"""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size), dtype=np.uint8)
    for _ in range(40):
        x1, y1 = rng.randint(0, size, 2)
        x2, y2 = rng.randint(0, size, 2)
        cv2.line(img, (x1, y1), (x2, y2), 255, 2)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def test_compare_same_image():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_synthetic_fingerprint(seed=1)
        p1 = Path(tmp) / "a.png"
        p2 = Path(tmp) / "b.png"
        cv2.imwrite(str(p1), img)
        cv2.imwrite(str(p2), img)

        matcher = FingerprintMatcher(min_good_matches=10)
        r = matcher.compare(str(p1), str(p2))
        assert r is not None
        assert r["decision"] == "SAME"
        assert r["good_matches"] >= 10


def test_compare_different_images():
    with tempfile.TemporaryDirectory() as tmp:
        img1 = _make_synthetic_fingerprint(seed=1)
        img2 = _make_synthetic_fingerprint(seed=2)
        p1 = Path(tmp) / "a.png"
        p2 = Path(tmp) / "b.png"
        cv2.imwrite(str(p1), img1)
        cv2.imwrite(str(p2), img2)

        matcher = FingerprintMatcher(min_good_matches=50)
        r = matcher.compare(str(p1), str(p2))
        assert r is not None
        assert r["decision"] == "DIFFERENT"


def test_compare_missing_file_returns_none():
    matcher = FingerprintMatcher()
    r = matcher.compare("not_exist_a.png", "not_exist_b.png")
    assert r is None


def test_visualize_returns_canvas():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_synthetic_fingerprint(seed=1)
        p1 = Path(tmp) / "a.png"
        p2 = Path(tmp) / "b.png"
        cv2.imwrite(str(p1), img)
        cv2.imwrite(str(p2), img)

        matcher = FingerprintMatcher()
        canvas = matcher.visualize(str(p1), str(p2))
        assert canvas is not None
        assert canvas.ndim == 3
        assert canvas.shape[2] == 3