
import csv
import tempfile
from pathlib import Path

from src.common.csv_logger import CSVLogger


def test_csv_logger_write_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        with CSVLogger(csv_path) as logger:
            logger.log(modality="face", metric="detection", value="0.95", decision="detected")
            logger.log(modality="face", metric="detection", value="0.87", decision="detected")

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["modality"] == "face"
        assert rows[0]["metric"] == "detection"
        assert rows[0]["value"] == "0.95"


def test_csv_logger_appends_without_duplicate_header():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        with CSVLogger(csv_path) as logger:
            logger.log(modality="face", metric="detection", value="0.95")
        with CSVLogger(csv_path) as logger:
            logger.log(modality="face", metric="detection", value="0.88")

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert lines[0].startswith("timestamp,modality,metric,value,decision")
        assert len(lines) == 3