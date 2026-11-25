from src.common.config_loader import load_config, resolve_path


def test_camera_config():
    cfg = load_config("camera")
    assert "default_index" in cfg
    assert "width" in cfg
    assert "height" in cfg


def test_paths_config():
    cfg = load_config("paths")
    assert "logs_dir" in cfg


def test_thresholds_config():
    cfg = load_config("thresholds")
    assert "face_detection_score" in cfg


def test_resolve_path_relative():
    p = resolve_path("data/logs")
    assert p.is_absolute()