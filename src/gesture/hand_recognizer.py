"""MediaPipe Gesture Recognizer 封装。"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.common.config_loader import resolve_path
from src.common.logger import get_logger

logger = get_logger("gesture.hand_recognizer")

DEFAULT_MODEL_PATH = "models/gesture_recognizer.task"


class HandGestureRecognizer:
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._recognizer = None
        self._mp = None
        self._num_hands = num_hands
        self._init_model(model_path, min_detection_confidence, min_tracking_confidence)

    def _init_model(
        self,
        model_path: Optional[str],
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise ImportError(
                "未安装 mediapipe，请执行 pip install mediapipe"
            ) from e

        self._mp = mp
        path = resolve_path(model_path or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"手势模型文件不存在: {path}。请从 MediaPipe 官方下载 "
                f"gesture_recognizer.task 并放到 models/ 目录"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(path))
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self._num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)
        logger.info("MediaPipe 手势识别器已加载: %s", path)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[Dict[str, Any]]:
        """返回每只手的 {label, score, landmarks, handedness}。"""
        if self._recognizer is None:
            return []
        rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)

        hands: List[Dict[str, Any]] = []
        if not result.hand_landmarks:
            return hands

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            label, score = "None", 0.0
            if idx < len(result.gestures) and result.gestures[idx]:
                top = result.gestures[idx][0]
                label, score = top.category_name, float(top.score)

            handedness = "Unknown"
            if idx < len(result.handedness) and result.handedness[idx]:
                handedness = result.handedness[idx][0].category_name

            # hand_landmarks 本身就是 NormalizedLandmark 的列表，直接遍历
            points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            hands.append(
                {
                    "label": label,
                    "score": score,
                    "handedness": handedness,
                    "landmarks": points,
                }
            )
        return hands

    def close(self) -> None:
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None