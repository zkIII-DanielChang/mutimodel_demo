import yaml
from pathlib import Path
from typing import Any, Dict

_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def _project_root() -> Path:
    # src/common/config_loader.py -> src/common -> src -> 项目根
    return Path(__file__).resolve().parents[2]


def load_config(name: str) -> Dict[str, Any]:
    if name in _CONFIG_CACHE:
        return _CONFIG_CACHE[name]
    config_path = _project_root() / "configs" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping: {config_path}")
    _CONFIG_CACHE[name] = data
    return data


def resolve_path(relative_path: str) -> Path:
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()