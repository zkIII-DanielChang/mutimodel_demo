import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List


class CSVLogger:
    FIELDS = ["timestamp", "modality", "metric", "value", "decision"]

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        need_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if need_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, modality: str, metric: str, value: Any, decision: str = "") -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "modality": modality,
            "metric": metric,
            "value": str(value),
            "decision": decision,
        }
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def log_many(self, rows: List[dict]) -> None:
        for r in rows:
            self.log(**r)

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()