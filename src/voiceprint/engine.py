"""声纹模板持久化与说话人判定。"""

from copy import deepcopy
from math import isfinite
from tempfile import TemporaryDirectory

import numpy as np

from src.common.config_loader import load_config, resolve_path
from src.common.logger import get_logger

logger = get_logger("voiceprint.engine")
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 0.65
MAX_AUDIO_SECONDS = 12.0
MATCH_MARGIN = 0.05


class VoiceprintEngine:
    def __init__(self, book_dir=None, threshold=0.40):
        from voicefingerprint import VoiceRecognizer

        default_dir = load_config("paths").get("registered_voices_dir", "data/registered_voices")
        self._book_path = resolve_path(book_dir or default_dir) / "speakers.npz"
        self._threshold = self._validate_threshold(threshold)
        self._vp = VoiceRecognizer(threshold=self._threshold)
        if self._book_path.exists():
            try:
                self._vp.load(self._book_path)
            except Exception as exc:
                raise RuntimeError(f"无法加载说话人库，原文件未修改：{self._book_path}") from exc

    @staticmethod
    def _validate_threshold(threshold):
        threshold = float(threshold)
        if not isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("相似度阈值应在 -1 到 1 之间")
        return threshold

    @staticmethod
    def _prepare_audio(audio, sample_rate):
        if audio is None or sample_rate <= 0:
            return None
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or samples.size < sample_rate * MIN_AUDIO_SECONDS:
            return None
        samples = samples[:int(sample_rate * MAX_AUDIO_SECONDS)]
        if not np.isfinite(samples).all() or np.mean(np.abs(samples) >= 0.99) > 0.01:
            return None
        samples = samples - samples.mean()
        if np.sqrt(np.mean(samples ** 2)) < 0.002:
            return None
        return np.clip(samples, -1, 1)

    def set_threshold(self, threshold):
        self._threshold = self._validate_threshold(threshold)
        self._vp.threshold = self._threshold

    def get_threshold(self):
        return self._threshold

    def speaker_count(self):
        return len(self._vp.book)

    def list_speakers(self):
        return list(self._vp.book.names)

    def enroll_from_audio(self, name, audio_clips, sample_rate=SAMPLE_RATE):
        name = name.strip()
        if not name or not audio_clips:
            return 0
        clips = [self._prepare_audio(clip, sample_rate) for clip in audio_clips]
        if any(clip is None for clip in clips):
            return 0
        previous = self._vp.book
        try:
            embeddings = self._vp.embed_many(clips, source_sr=sample_rate)
            if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) < 1e-6):
                return 0
            if len(embeddings) > 1:
                unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                pairs = (unit @ unit.T)[np.triu_indices(len(unit), k=1)]
                if pairs.min() < 0.50:
                    logger.warning("注册语音段一致性过低，未保存模板")
                    return 0
            proposed = deepcopy(previous)
            proposed.add(name, embeddings)
            self._book_path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=self._book_path.parent) as tmp:
                saved = proposed.save(resolve_path(tmp) / self._book_path.name)
                saved.replace(self._book_path)
            self._vp.book = proposed
            return len(clips)
        except Exception as exc:
            logger.error("声纹注册失败：%s", exc)
            return 0

    def identify_audio(self, audio, sample_rate=SAMPLE_RATE):
        prepared = self._prepare_audio(audio, sample_rate)
        if prepared is None or not self.speaker_count():
            return None
        threshold = self._threshold
        matches = self._vp.identify(prepared, top_k=2, threshold=threshold, source_sr=sample_rate)
        if not matches or not all(isfinite(match.score) for match in matches):
            return None
        best = matches[0]
        gap = float(best.score - matches[1].score) if len(matches) > 1 else None
        reason = "below_threshold" if best.score < threshold else (
            "ambiguous" if gap is not None and gap < MATCH_MARGIN else "matched"
        )
        return {"name": str(best.name) if reason == "matched" else "Unknown",
                "score": float(best.score), "accepted": reason == "matched",
                "reason": reason, "gap": gap, "threshold": threshold}

    def calibrate_threshold(self):
        if self.speaker_count() < 2:
            return None
        if any(self._vp.book[name].n_utterances < 2 for name in self.list_speakers()):
            return None
        try:
            report = self._vp.calibrate()
            self.set_threshold(report["threshold"])
            return report
        except Exception as exc:
            self._vp.threshold = self._threshold
            logger.warning("声纹阈值校准失败：%s", exc)
            return None

    def is_ready(self):
        return True

    def clear(self):
        book = self._vp.book
        empty = type(book)(book.model, book.embedding_dim)
        self._book_path.unlink(missing_ok=True)
        self._vp.book = empty
        try:
            self._book_path.with_suffix(".txt").unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("未删除旧版说话人索引：%s", exc)
