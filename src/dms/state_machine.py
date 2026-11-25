"""DMS 演示用的时序风险状态机。"""

from typing import Dict, Optional


class DMSStateMachine:
    """把单帧视觉特征转换为稳定的驾驶风险状态。"""

    NORMAL = "normal"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    SEVERE_DROWSY = "severe_drowsy"
    UNAVAILABLE = "unavailable"

    MESSAGES = {
        NORMAL: "正常驾驶",
        DISTRACTED: "请注意前方道路",
        DROWSY: "检测到疲劳，请保持清醒",
        SEVERE_DROWSY: "严重疲劳，请立即休息",
        UNAVAILABLE: "DMS 不可用：请调整摄像头或坐姿",
    }

    ALERT_LEVELS = {
        NORMAL: 0,
        DISTRACTED: 1,
        DROWSY: 2,
        SEVERE_DROWSY: 3,
        UNAVAILABLE: 1,
    }

    def __init__(
        self,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.55,
        distraction_yaw: float = 25.0,
        distraction_pitch: float = 20.0,
        distraction_threshold: float = 2.0,
        drowsy_closed_threshold: float = 0.8,
        severe_closed_threshold: float = 1.5,
        yawn_threshold: float = 1.0,
        unavailable_threshold: float = 2.0,
        recovery_threshold: float = 0.8,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.distraction_yaw = distraction_yaw
        self.distraction_pitch = distraction_pitch
        self.distraction_threshold = distraction_threshold
        self.drowsy_closed_threshold = drowsy_closed_threshold
        self.severe_closed_threshold = severe_closed_threshold
        self.yawn_threshold = yawn_threshold
        self.unavailable_threshold = unavailable_threshold
        self.recovery_threshold = recovery_threshold

        self.state = self.NORMAL
        self._last_timestamp: Optional[float] = None
        self._clear_seconds = 0.0
        self.face_lost_seconds = 0.0
        self.quality_bad_seconds = 0.0
        self.closed_seconds = 0.0
        self.yawn_duration = 0.0
        self.distraction_duration = 0.0

    def update(
        self,
        *,
        timestamp: float,
        face_detected: bool,
        quality_ok: bool = True,
        ear: Optional[float] = None,
        mar: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> Dict[str, object]:
        dt = 0.0
        if self._last_timestamp is not None:
            dt = max(0.0, min(timestamp - self._last_timestamp, 0.25))
        self._last_timestamp = timestamp

        if face_detected:
            self.face_lost_seconds = 0.0
        else:
            self.face_lost_seconds += dt

        if quality_ok:
            self.quality_bad_seconds = 0.0
        else:
            self.quality_bad_seconds += dt

        if face_detected and quality_ok:
            if ear is not None and ear < self.ear_threshold:
                self.closed_seconds += dt
            else:
                self.closed_seconds = 0.0

            if mar is not None and mar > self.mar_threshold:
                self.yawn_duration += dt
            else:
                self.yawn_duration = 0.0

            pose_off = (
                yaw is not None
                and pitch is not None
                and (
                    abs(yaw) >= self.distraction_yaw
                    or abs(pitch) >= self.distraction_pitch
                )
            )
            if pose_off:
                self.distraction_duration += dt
            else:
                self.distraction_duration = 0.0
        else:
            self.closed_seconds = 0.0
            self.yawn_duration = 0.0
            self.distraction_duration = 0.0

        candidate = self._candidate_state()
        changed = False
        if candidate == self.state:
            self._clear_seconds = 0.0
        elif candidate == self.NORMAL:
            self._clear_seconds += dt
            if self._clear_seconds >= self.recovery_threshold:
                self.state = candidate
                self._clear_seconds = 0.0
                changed = True
        else:
            self.state = candidate
            self._clear_seconds = 0.0
            changed = True

        return {
            "state": self.state,
            "changed": changed,
            "alert_level": self.ALERT_LEVELS[self.state],
            "message": self.MESSAGES[self.state],
            "face_lost_seconds": self.face_lost_seconds,
            "quality_bad_seconds": self.quality_bad_seconds,
            "closed_seconds": self.closed_seconds,
            "yawn_seconds": self.yawn_duration,
            "distraction_seconds": self.distraction_duration,
        }

    def _candidate_state(self) -> str:
        if (
            self.face_lost_seconds >= self.unavailable_threshold
            or self.quality_bad_seconds >= self.unavailable_threshold
        ):
            return self.UNAVAILABLE
        if self.closed_seconds >= self.severe_closed_threshold:
            return self.SEVERE_DROWSY
        if (
            self.closed_seconds >= self.drowsy_closed_threshold
            or self.yawn_duration >= self.yawn_threshold
        ):
            return self.DROWSY
        if self.distraction_duration >= self.distraction_threshold:
            return self.DISTRACTED
        return self.NORMAL
