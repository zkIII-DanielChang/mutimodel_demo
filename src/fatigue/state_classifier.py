"""基于 EAR/MAR 时序的规则疲劳分级。"""

from collections import deque
from typing import Deque, Dict, Tuple


class FatigueStateClassifier:
    def __init__(
        self,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.55,
        window_seconds: float = 2.0,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.window_seconds = window_seconds
        self._history: Deque[Tuple[float, float, float]] = deque()
        self._closed_seconds = 0.0
        self._yawn_seconds = 0.0

    def update(self, ear: float, mar: float, timestamp: float) -> Dict[str, object]:
        self._history.append((timestamp, ear, mar))
        while self._history and timestamp - self._history[0][0] > self.window_seconds:
            self._history.popleft()

        dt = 0.0
        if len(self._history) >= 2:
            dt = self._history[-1][0] - self._history[-2][0]
        dt = max(0.0, min(dt, 0.2))

        if ear < self.ear_threshold:
            self._closed_seconds += dt
        else:
            self._closed_seconds = max(0.0, self._closed_seconds - dt * 2.0)

        if mar > self.mar_threshold:
            self._yawn_seconds += dt
        else:
            self._yawn_seconds = max(0.0, self._yawn_seconds - dt * 1.5)

        # 规则打分
        score = 0.0
        if self._closed_seconds > 0.5:
            score += 1.0
        if self._closed_seconds > 1.2:
            score += 1.0
        if self._yawn_seconds > 0.4:
            score += 0.5
        if self._yawn_seconds > 1.0:
            score += 0.5

        if score >= 2.0:
            state = "severe"
        elif score >= 1.0:
            state = "mild"
        else:
            state = "awake"

        return {
            "state": state,
            "score": score,
            "closed_seconds": self._closed_seconds,
            "yawn_seconds": self._yawn_seconds,
        }