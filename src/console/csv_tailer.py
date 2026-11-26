"""增量读取演示 CSV，只缓存最近的完整记录。"""

import csv
from collections import deque
from datetime import datetime
from time import time
from pathlib import Path

from src.common.csv_logger import CSVLogger

_CACHE = {}


def _read_cached(path):
    try:
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        cached = _CACHE.get(path)
        if cached and cached[0] == identity and stat.st_size >= cached[1]:
            _, offset, rows = cached
        else:
            offset, rows = 0, deque(maxlen=500)
        if stat.st_size > offset:
            with path.open('rb') as f:
                f.seek(offset)
                data = f.read(stat.st_size - offset)
            lines = [line + b'\n' for line in data.split(b'\n')[:-1]]
            reader = csv.reader((line.decode('utf-8') for line in lines), strict=True)
            if offset == 0:
                if next(reader, None) != CSVLogger.FIELDS:
                    return ()
                offset = len(lines[0])
            consumed = reader.line_num
            try:
                for record in reader:
                    offset += sum(map(len, lines[consumed:reader.line_num]))
                    consumed = reader.line_num
                    if len(record) == len(CSVLogger.FIELDS):
                        rows.append(dict(zip(CSVLogger.FIELDS, record)))
            except csv.Error:
                # 被引号包住的字段可能跨行；下一次从该记录起点重读。
                pass
            _CACHE[path] = (identity, offset, rows)
        return rows
    except (OSError, UnicodeError, csv.Error):
        _CACHE.pop(path, None)
        return ()


def read_last_lines(csv_path: Path, n: int = 10):
    return list(_read_cached(csv_path))[-n:] if n > 0 else []


def _timestamp(row):
    try:
        return datetime.fromisoformat(row['timestamp']).timestamp()
    except (KeyError, TypeError, ValueError):
        return 0.0


def latest_metric(csv_path: Path, metric: str):
    return latest_metrics(csv_path, [metric]).get(metric)


def latest_metrics(csv_path: Path, metrics, *, since=None, max_age=None):
    cutoff = max(since or 0, time() - max_age if max_age is not None else 0)
    result = {}
    for row in reversed(_read_cached(csv_path)):
        key = row.get('metric')
        if key in metrics and key not in result and _timestamp(row) >= cutoff:
            result[key] = row.get('value') or ''
        if len(result) == len(metrics):
            break
    return result


def estimate_fps_from_csv(csv_path: Path, window_seconds=1.0, *, since=None, metric=None):
    cutoff = max(since or 0, time() - window_seconds)
    allowed = (metric,) if metric else ('detection', 'label', 'ear')
    times = [_timestamp(row) for row in _read_cached(csv_path) if row.get('metric') in allowed]
    times = [stamp for stamp in times if stamp >= cutoff]
    if len(times) > 1 and times[-1] > times[0]:
        return (len(times) - 1) / (times[-1] - times[0])
    return None


def clear_cache():
    _CACHE.clear()
