"""人脸注册库：numpy .npz 持久化，余弦相似度 1:N 搜索。"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.logger import get_logger

logger = get_logger("face.gallery")


class FaceGallery:
    def __init__(self, gallery_path: Path):
        self.gallery_path = Path(gallery_path)
        self.names: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D) L2-normalized

    def load(self) -> None:
        if not self.gallery_path.exists():
            logger.info("注册库不存在，初始为空: %s", self.gallery_path)
            self.names = []
            self.embeddings = None
            return
        data = np.load(self.gallery_path, allow_pickle=True)
        self.names = list(data["names"])
        embs = data["embeddings"].astype(np.float32)
        self.embeddings = self._l2_normalize(embs)
        logger.info("注册库已加载: %d 人", len(self.names))

    def save(self) -> None:
        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.names or self.embeddings is None:
            logger.warning("注册库为空，跳过保存")
            return
        np.savez(
            self.gallery_path,
            names=np.array(self.names, dtype=object),
            embeddings=self.embeddings.astype(np.float32),
        )
        logger.info("注册库已保存: %s (%d 人)", self.gallery_path, len(self.names))

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        emb = self._l2_normalize(embedding.reshape(1, -1))
        if self.embeddings is None:
            self.embeddings = emb
            self.names = [name]
            return
        # 同名覆盖，避免重复注册导致库膨胀
        if name in self.names:
            idx = self.names.index(name)
            self.embeddings[idx] = emb[0]
            logger.info("同名已存在，覆盖嵌入: %s", name)
            return
        self.embeddings = np.vstack([self.embeddings, emb])
        self.names.append(name)

    def search(
        self, embedding: np.ndarray, threshold: float
    ) -> Dict[str, object]:
        if self.embeddings is None or not self.names:
            return {"name": "Unknown", "similarity": 0.0, "matched": False}
        query = self._l2_normalize(embedding.reshape(1, -1))[0]
        sims = self.embeddings @ query  # 两者均已 L2 归一化，点积即余弦相似度
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= threshold:
            return {"name": self.names[best_idx], "similarity": best_sim, "matched": True}
        return {"name": "Unknown", "similarity": best_sim, "matched": False}

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return x / norms

    def __len__(self) -> int:
        return len(self.names)