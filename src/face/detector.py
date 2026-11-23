from typing import Any, Dict, List

from src.common.device import get_onnx_providers
from src.common.logger import get_logger

logger = get_logger("face.detector")


class BaseDetector:
    def detect(self, frame) -> List[Dict[str, Any]]:
        raise NotImplementedError


class UniFaceDetector(BaseDetector):
    def __init__(self, score_threshold: float = 0.5, prefer_cuda: bool = False):
        self.score_threshold = score_threshold
        self._detector = None
        self._init_model(prefer_cuda)

    def _init_model(self, prefer_cuda: bool):
        try:
            from uniface.detection import RetinaFace
        except ImportError as e:
            raise ImportError(
                "未安装 UniFace。请执行 pip install -r requirements/face.txt，"
                "或临时改用 --backend opencv-haar"
            ) from e

        providers = get_onnx_providers(prefer_cuda=prefer_cuda)
        try:
            self._detector = RetinaFace(providers=providers)
        except TypeError:
            self._detector = RetinaFace()
        logger.info("UniFace RetinaFace 检测器已加载, providers=%s", providers)

    def detect(self, frame) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        results = self._detector.detect(frame)
        faces: List[Dict[str, Any]] = []
        for r in results:
            bbox, score, landmarks = self._parse_result(r)
            if bbox is None or score < self.score_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "score": score,
                    "landmarks": landmarks,
                }
            )
        return faces

    @staticmethod
    def _parse_result(r):
        if isinstance(r, dict):
            return r.get("bbox"), float(r.get("score", 0.0)), r.get("landmarks")
        if hasattr(r, "bbox"):
            return (
                getattr(r, "bbox"),
                float(getattr(r, "score", 1.0)),
                getattr(r, "landmarks", None),
            )
        try:
            import numpy as np

            arr = np.asarray(r).flatten()
            if arr.size >= 4:
                score = float(arr[4]) if arr.size > 4 else 1.0
                return arr[:4], score, None
        except Exception:
            pass
        return None, 0.0, None


class OpenCVHaarDetector(BaseDetector):
    def __init__(self, score_threshold: float = 0.5):
        import cv2

        self._cv2 = cv2
        self.score_threshold = score_threshold
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"无法加载 Haar 级联模型: {cascade_path}")
        logger.info("OpenCV Haar 检测器已加载（仅用于工程骨架验证）")

    def detect(self, frame) -> List[Dict[str, Any]]:
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [
            {
                "bbox": (int(x), int(y), int(x + w), int(y + h)),
                "score": 1.0,
                "landmarks": None,
            }
            for (x, y, w, h) in detections
        ]


def create_detector(
    backend: str,
    score_threshold: float = 0.5,
    prefer_cuda: bool = False,
) -> BaseDetector:
    if backend == "uniface":
        return UniFaceDetector(score_threshold=score_threshold, prefer_cuda=prefer_cuda)
    if backend == "opencv-haar":
        return OpenCVHaarDetector(score_threshold=score_threshold)
    raise ValueError(f"未知后端: {backend}")