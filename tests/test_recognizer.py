from unittest.mock import MagicMock

import numpy as np

from src.face.recognizer import ArcFaceRecognizer


def test_extract_from_faces_returns_list():
    rec = ArcFaceRecognizer.__new__(ArcFaceRecognizer)
    rec._recognizer = MagicMock()
    rec._recognizer.get_normalized_embedding.return_value = np.random.randn(512).astype(np.float32)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = [{"landmarks": [(1, 2)] * 5}, {"landmarks": [(3, 4)] * 5}]
    out = rec.extract_from_faces(image, faces)
    assert len(out) == 2
    assert all(e.shape == (512,) for e in out)


def test_extract_skips_faces_without_landmarks():
    rec = ArcFaceRecognizer.__new__(ArcFaceRecognizer)
    rec._recognizer = MagicMock()
    rec._recognizer.get_normalized_embedding.return_value = np.zeros(512, dtype=np.float32)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = [{"bbox": (0, 0, 10, 10)}]  # 无 landmarks
    out = rec.extract_from_faces(image, faces)
    assert out == []