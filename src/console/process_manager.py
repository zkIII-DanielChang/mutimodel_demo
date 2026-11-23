"""模态子进程管理：启动、停止、状态查询。

视频模态同组互斥（group 相同只能运行一个），切换时先 terminate 再启动。
停止后等待摄像头释放（Windows 上有 1–2 秒延迟）。

关键：使用 CREATE_NEW_CONSOLE 让子进程拥有独立控制台，OpenCV 窗口才能创建。
透明面板问题由各 run_*.py 里的 cv2.namedWindow(..., WINDOW_NORMAL) 解决。
子进程输出重定向到 data/logs/{key}.log，方便排查。
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from src.common.logger import get_logger

logger = get_logger("console.process_manager")

CAMERA_RELEASE_WAIT = 2.5


class ModalityProcess:
    def __init__(self, key: str, config: dict, project_root: Path, log_dir: Path):
        self.key = key
        self.label = config["label"]
        self.type = config["type"]
        self.group = config["group"]
        self.command = config["command"]
        self.csv = config.get("csv", "")
        self.project_root = project_root
        self.log_dir = log_dir
        self.process: Optional[subprocess.Popen] = None
        self.status = "idle"
        self.started_at: Optional[float] = None
        self._log_file = None

    def start(self) -> bool:
        if self.process is not None and self.process.poll() is None:
            logger.info("%s 已在运行", self.label)
            return False
        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_CONSOLE

            cmd = [sys.executable] + self.command[1:]

            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{self.key}.log"
            self._log_file = open(log_path, "w", encoding="utf-8")

            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                creationflags=creationflags,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
            )
            self.status = "running"
            self.started_at = time.time()
            logger.info("启动 %s: %s", self.label, " ".join(cmd))
            return True
        except Exception as e:
            self.status = "error"
            logger.error("启动 %s 失败: %s", self.label, e)
            self._close_log()
            return False

    def stop(self, wait_camera: bool = False) -> bool:
        if self.process is None or self.process.poll() is not None:
            self._close_log()
            self.process = None
            self.status = "idle"
            return False
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None
            self.status = "idle"
            self.started_at = None
            self._close_log()
            logger.info("停止 %s", self.label)
            if wait_camera:
                time.sleep(CAMERA_RELEASE_WAIT)
            return True
        except Exception as e:
            logger.error("停止 %s 失败: %s", self.label, e)
            self.status = "error"
            return False

    def _close_log(self):
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def is_running(self) -> bool:
        if self.process is None:
            return False
        if self.process.poll() is not None:
            self.process = None
            if self.status == "running":
                self.status = "idle"
            self._close_log()
            return False
        return True

    def refresh_status(self) -> str:
        if self.status in ("starting", "stopping"):
            return self.status
        if self.is_running():
            self.status = "running"
        elif self.status != "error":
            self.status = "idle"
        return self.status


class ProcessManager:
    def __init__(self, modalities_config: Dict[str, dict], project_root: Path, log_dir: Path):
        self.modalities: Dict[str, ModalityProcess] = {
            key: ModalityProcess(key, cfg, project_root, log_dir)
            for key, cfg in modalities_config.items()
        }

    def start(self, key: str) -> bool:
        if key not in self.modalities:
            return False
        target = self.modalities[key]
        if target.is_running():
            return False

        if target.type == "video":
            for other in self.modalities.values():
                if other.key != key and other.type == "video" and other.group == target.group:
                    if other.is_running():
                        logger.info("同组 %s 运行中，先停止", other.label)
                        other.stop(wait_camera=True)
                        break

        target.status = "starting"
        ok = target.start()
        return ok

    def stop(self, key: str) -> bool:
        if key not in self.modalities:
            return False
        target = self.modalities[key]
        target.status = "stopping"
        ok = target.stop(wait_camera=(target.type == "video"))
        return ok

    def stop_all_video(self) -> None:
        for m in self.modalities.values():
            if m.type == "video" and m.is_running():
                m.stop(wait_camera=True)

    def stop_all(self) -> None:
        for m in self.modalities.values():
            if m.is_running():
                m.stop(wait_camera=False)

    def get_status(self, key: str) -> str:
        if key not in self.modalities:
            return "idle"
        return self.modalities[key].refresh_status()

    def get_all_status(self) -> Dict[str, str]:
        return {key: m.refresh_status() for key, m in self.modalities.items()}

    def running_modalities(self) -> list:
        return [key for key, m in self.modalities.items() if m.is_running()]