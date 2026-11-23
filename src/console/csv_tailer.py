"""读 CSV 最后 N 行，解析成字典，供右栏指标显示。

带缓存：只有文件 mtime 或 size 变化时才重新读，避免每 500ms 读一次大文件。
缓存只保留最后 MAX_CACHE_ROWS 行，防止内存无限增长。
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("console.csv_tailer")

MAX_CACHE_ROWS = 500

# path_str -> (mtime, size, rows)
_CACHE: Dict[str, tuple] = {}


def _read_cached(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    try:
        stat = csv_path.stat()
    except Exception:
        return []
    key = str(csv_path)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            all_rows = list(csv.DictReader(f))
    except Exception as e:
        logger.debug("读 CSV 失败 %s: %s", csv_path, e)
        return []
    rows = all_rows[-MAX_CACHE_ROWS:] if len(all_rows) > MAX_CACHE_ROWS else all_rows
    _CACHE[key] = (stat.st_mtime, stat.st_size, rows)
    return rows


def read_last_lines(csv_path: Path, n: int = 10) -> List[Dict[str, str]]:
    rows = _read_cached(csv_path)
    if not rows:
        return []
    return rows[-n:] if len(rows) > n else rows


def latest_metric(csv_path: Path, metric: str) -> Optional[str]:
    rows = read_last_lines(csv_path, n=50)
    for row in reversed(rows):
        if row.get("metric") == metric:
            return row.get("value")
    return None


def latest_metrics(csv_path: Path, metrics: List[str]) -> Dict[str, str]:
    if not metrics:
        return {}
    rows = read_last_lines(csv_path, n=100)
    result: Dict[str, str] = {}
    for row in reversed(rows):
        m = row.get("metric")
        if m in metrics and m not in result:
            result[m] = row.get("value", "")
        if len(result) == len(metrics):
            break
    return result


def estimate_fps_from_csv(csv_path: Path, window_seconds: float = 1.0) -> Optional[float]:
    rows = read_last_lines(csv_path, n=200)
    if len(rows) < 2:
        return None
    try:
        from datetime import datetime

        timestamps = []
        for row in rows:
            if row.get("metric") in ("detection", "label", "ear"):
                ts = row.get("timestamp", "")
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except Exception:
                    continue
        if len(timestamps) < 2:
            return None
        span = (timestamps[-1] - timestamps[0]).total_seconds()
        if span <= 0:
            return None
        return len(timestamps) / span
    except Exception:
        return None


def clear_cache() -> None:
    _CACHE.clear()