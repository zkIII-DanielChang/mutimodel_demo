import tempfile
from pathlib import Path

import numpy as np

from src.face.gallery import FaceGallery


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def test_empty_gallery_returns_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        g = FaceGallery(Path(tmp) / "g.npz")
        g.load()
        r = g.search(_unit([1, 0, 0, 0]), threshold=0.6)
        assert r["name"] == "Unknown"
        assert r["matched"] is False


def test_enroll_and_search_same_person():
    with tempfile.TemporaryDirectory() as tmp:
        g = FaceGallery(Path(tmp) / "g.npz")
        g.load()
        base = _unit([1, 0, 0, 0])
        g.enroll("alice", base)
        r = g.search(_unit([0.99, 0.01, 0, 0]), threshold=0.6)
        assert r["name"] == "alice"
        assert r["matched"] is True
        assert r["similarity"] > 0.6


def test_different_person_below_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        g = FaceGallery(Path(tmp) / "g.npz")
        g.load()
        g.enroll("alice", _unit([1, 0, 0, 0]))
        r = g.search(_unit([0, 1, 0, 0]), threshold=0.6)
        assert r["name"] == "Unknown"
        assert r["matched"] is False


def test_same_name_overwrites():
    with tempfile.TemporaryDirectory() as tmp:
        g = FaceGallery(Path(tmp) / "g.npz")
        g.load()
        g.enroll("alice", _unit([1, 0, 0, 0]))
        g.enroll("alice", _unit([0, 1, 0, 0]))
        assert len(g) == 1
        r = g.search(_unit([0, 1, 0, 0]), threshold=0.6)
        assert r["name"] == "alice"


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "g.npz"
        g1 = FaceGallery(path)
        g1.load()
        g1.enroll("alice", _unit([1, 0, 0, 0]))
        g1.enroll("bob", _unit([0, 1, 0, 0]))
        g1.save()

        g2 = FaceGallery(path)
        g2.load()
        assert len(g2) == 2
        assert g2.search(_unit([0.98, 0.02, 0, 0]), 0.6)["name"] == "alice"
        assert g2.search(_unit([0.02, 0.98, 0, 0]), 0.6)["name"] == "bob"