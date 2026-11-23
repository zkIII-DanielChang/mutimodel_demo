"""voice-fingerprint 封装：临时 WAV 桥接 + 说话人库加载/保存。

关键修复：VoiceRecognizer() 构造函数不接受文件路径。
先创建空实例，再通过 save() 持久化；加载时尝试 load() 或重新注册，
失败则降级为空库，不阻塞 GUI 启动。
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.common.config_loader import resolve_path
from src.common.logger import get_logger

logger = get_logger("voiceprint.engine")

SAMPLE_RATE = 16000
DEFAULT_BOOK_NAME = "speakers"


class VoiceprintEngine:
    def __init__(
        self,
        book_dir: Optional[str] = None,
        threshold: float = 0.40,
    ):
        self._vp = None
        self._book_path: Optional[Path] = None
        self._speakers_file: Optional[Path] = None
        self._threshold = threshold
        self._init_model(book_dir)

    def _init_model(self, book_dir: Optional[str]) -> None:
        try:
            from voicefingerprint import VoiceRecognizer
        except ImportError as e:
            raise ImportError(
                "未安装 voice-fingerprint，请执行 pip install voice-fingerprint"
            ) from e

        base = resolve_path(book_dir or "data/registered_voices")
        base.mkdir(parents=True, exist_ok=True)
        self._book_path = base / DEFAULT_BOOK_NAME
        self._speakers_file = base / "speakers.txt"

        # 总是先创建干净的实例
        self._vp = VoiceRecognizer()

        # 如果库文件存在，尝试加载；失败则保持空库
        npz_path = self._book_path.with_suffix(".npz")
        if npz_path.exists():
            loaded = False
            # 尝试常见的 load 方法名，兼容不同版本
            for method_name in ("load", "load_book", "from_book"):
                if hasattr(self._vp, method_name):
                    try:
                        getattr(self._vp, method_name)(str(self._book_path))
                        logger.info("说话人库已加载: %s（通过 %s）", npz_path, method_name)
                        loaded = True
                        break
                    except Exception as e:
                        logger.debug("%s 加载失败: %s", method_name, e)
            if not loaded:
                # 尝试构造函数带 book 参数（但不用路径，用名称）
                try:
                    from voicefingerprint import VoiceRecognizer as VR
                    self._vp = VR(book=str(self._book_path))
                    logger.info("说话人库已加载（通过 book 参数）: %s", npz_path)
                    loaded = True
                except Exception as e:
                    logger.debug("book 参数加载失败: %s", e)
            if not loaded:
                logger.warning(
                    "检测到库文件 %s，但自动加载失败；将使用空库，请重新注册或手动加载。",
                    npz_path,
                )
        else:
            logger.info("说话人库为空，请先用 GUI 注册")

    def set_threshold(self, threshold: float) -> None:
        self._threshold = float(threshold)
        logger.info("声纹判定阈值已设为 %.2f", self._threshold)

    def get_threshold(self) -> float:
        return self._threshold

    def list_speakers(self) -> List[str]:
        if not self._speakers_file or not self._speakers_file.exists():
            return []
        try:
            return [
                line.strip()
                for line in self._speakers_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception:
            return []

    def _record_speaker(self, name: str) -> None:
        if not self._speakers_file:
            return
        names = self.list_speakers()
        if name not in names:
            names.append(name)
        try:
            self._speakers_file.write_text(
                "\n".join(names) + "\n", encoding="utf-8"
            )
        except Exception as e:
            logger.warning("记录说话人姓名失败: %s", e)

    def enroll_from_audio(
        self, name: str, audio_clips: List[np.ndarray], sample_rate: int = SAMPLE_RATE
    ) -> int:
        if not audio_clips:
            return 0
        wav_paths = []
        for i, clip in enumerate(audio_clips):
            path = self._write_temp_wav(clip, sample_rate, prefix=f"{name}_{i}_")
            if path:
                wav_paths.append(path)
        if not wav_paths:
            return 0
        try:
            self._vp.enroll(name, [str(p) for p in wav_paths])
            self._vp.save(str(self._book_path))
            self._record_speaker(name)
            logger.info("注册完成: %s，使用 %d 段音频", name, len(wav_paths))
            return len(wav_paths)
        except Exception as e:
            logger.error("声纹注册失败: %s", e)
            return 0
        finally:
            for p in wav_paths:
                p.unlink(missing_ok=True)

    def identify_audio(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
    ) -> Optional[dict]:
        if audio is None or audio.size == 0:
            return None
        path = self._write_temp_wav(audio, sample_rate, prefix="identify_")
        if path is None:
            return None
        try:
            matches = list(self._vp.identify(str(path)))
            if not matches:
                return None
            best = matches[0]
            score = float(best.score)
            accepted = score >= self._threshold
            return {
                "name": best.name if accepted else "Unknown",
                "score": score,
                "accepted": accepted,
            }
        except Exception as e:
            logger.warning("声纹识别失败: %s", e)
            return None
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _write_temp_wav(
        audio: np.ndarray, sample_rate: int, prefix: str = "vp_"
    ) -> Optional[Path]:
        try:
            from scipy.io import wavfile

            a = np.asarray(audio, dtype=np.float32).flatten()
            peak = float(np.max(np.abs(a))) if a.size else 0.0
            if peak > 1.0:
                a = a / peak
            fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=".wav")
            os.close(fd)
            path = Path(tmp)
            wavfile.write(str(path), sample_rate, (a * 32767).astype(np.int16))
            return path
        except Exception as e:
            logger.warning("写临时 WAV 失败: %s", e)
            return None

    def is_ready(self) -> bool:
        return self._vp is not None

    def clear(self) -> None:
        """清空说话人库：删除文件并重置为空实例。"""
        try:
            if self._book_path:
                self._book_path.with_suffix(".npz").unlink(missing_ok=True)
            if self._speakers_file:
                self._speakers_file.unlink(missing_ok=True)
            from voicefingerprint import VoiceRecognizer
            self._vp = VoiceRecognizer()
            logger.info("说话人库已清空")
        except Exception as e:
            logger.warning("清空说话人库失败: %s", e)