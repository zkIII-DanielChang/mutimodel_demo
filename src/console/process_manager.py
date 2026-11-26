"""按设备互斥管理演示子进程。"""

import os
import subprocess
import sys
import time
from pathlib import Path

from src.common.logger import get_logger

logger = get_logger("console.process_manager")
CAMERA_RELEASE_WAIT = 2.5


class ModalityProcess:
    def __init__(self, key: str, config: dict, project_root: Path, log_dir: Path):
        self.key = key
        self.label = config["label"]
        self.type = config["type"]
        self.resource = "camera" if self.type == "video" else config.get("resource")
        self.command = config["command"]
        self.csv = config.get("csv", "")
        self.project_root = project_root
        self.log_dir = log_dir
        self.process = None
        self.status = "idle"
        self.started_at = None
        self._log_file = None

    def start(self) -> bool:
        if self.is_running():
            return True
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = (self.log_dir / f"{self.key}.log").open("w", encoding="utf-8")
            self.started_at = time.time()
            self.process = subprocess.Popen(
                [sys.executable, "-u", *self.command[1:]],
                cwd=self.project_root,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self.status = "running"
            logger.info("启动 %s", self.label)
            return True
        except OSError as exc:
            self.status = "error"
            self.started_at = None
            self._close_log()
            logger.error("启动 %s 失败：%s", self.label, exc)
            return False

    def stop(self) -> bool:
        if not self.is_running():
            self.status = "idle"
            return True
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired) as exc:
            if self.is_running():
                self.status = "error"
                logger.error("停止 %s 失败：%s", self.label, exc)
                return False
        self._finish("idle")
        return True

    def _close_log(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _finish(self, status):
        self.process = None
        self.started_at = None
        self.status = status
        self._close_log()

    def is_running(self) -> bool:
        if self.process is None:
            return False
        code = self.process.poll()
        if code is None:
            return True
        self._finish("idle" if code == 0 else "error")
        return False


class ProcessManager:
    def __init__(self, modalities_config: dict, project_root: Path, log_dir: Path):
        self.modalities = {
            key: ModalityProcess(key, cfg, project_root, log_dir)
            for key, cfg in modalities_config.items()
        }
        self.camera_ready_at = 0.0

    def start(self, key: str) -> bool:
        target = self.modalities[key]
        if target.is_running():
            return True
        if target.resource:
            for other in self.modalities.values():
                if other is not target and other.resource == target.resource and other.is_running():
                    if not self.stop(other.key):
                        return False
        if target.resource == "camera":
            delay = self.camera_ready_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return target.start()

    def stop(self, key: str) -> bool:
        modality = self.modalities[key]
        had_camera = modality.resource == "camera" and modality.is_running()
        if not modality.stop():
            return False
        if had_camera and sys.platform == "win32":
            self.camera_ready_at = time.monotonic() + CAMERA_RELEASE_WAIT
        return True

    def activate(self, keys) -> bool:
        ok = True
        for key in self.running_modalities():
            if key not in keys and not self.stop(key):
                ok = False
        for key in keys:
            if not self.start(key):
                ok = False
        return ok

    def stop_all(self) -> bool:
        return self.activate(())

    def get_all_status(self) -> dict:
        for modality in self.modalities.values():
            modality.is_running()
        return {key: modality.status for key, modality in self.modalities.items()}

    def running_modalities(self) -> list:
        return [key for key, modality in self.modalities.items() if modality.is_running()]
