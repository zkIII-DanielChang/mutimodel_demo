"""ArcFace 嵌入提取封装。"""

from typing import Any, List

import numpy as np

from src.common.device import get_onnx_providers
from src.common.logger import get_logger

logger = get_logger("face.recognizer")


class ArcFaceRecognizer:
    def __init__(self, prefer_cuda: bool = False):
        self._recognizer = None
        self._init_model(prefer_cuda)

    def _init_model(self, prefer_cuda: bool) -> None:
        try:
            from uniface.recognition import ArcFace
        except ImportError as e:
            raise ImportError(
                "未安装 UniFace，请执行 pip install -r requirements/face.txt"
            ) from e
        providers = get_onnx_providers(prefer_cuda=prefer_cuda)
        try:
            self._recognizer = ArcFace(providers=providers)
        except TypeError:
            self._recognizer = ArcFace()
        logger.info("ArcFace 识别器已加载, providers=%s", providers)

    def extract(
        self, image: np.ndarray, landmarks: Any
    ) -> np.ndarray:
        """返回 L2 归一化 512 维嵌入。landmarks 为 RetinaFace 输出的 5 点关键点。"""
        emb = self._recognizer.get_normalized_embedding(image, landmarks)
        emb = np.asarray(emb, dtype=np.float32).flatten()
        if emb.size == 0:
            raise ValueError("ArcFace 返回空嵌入")
        return emb

    def extract_from_faces(
        self, image: np.ndarray, faces: List[dict]
    ) -> List[np.ndarray]:
        embeddings = []
        for f in faces:
            landmarks = f.get("landmarks")
            if landmarks is None:
                continue
            embeddings.append(self.extract(image, landmarks))
        return embeddings