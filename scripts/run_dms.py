"""笔记本摄像头版 DMS 演示。

启动：
    python scripts/run_dms.py
    python scripts/run_dms.py --camera 1 --no-sound

按键：q 退出，c 校准当前 pitch 为 0，m 切换镜像显示。
"""

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.common.config_loader import load_config, resolve_path
from src.common.csv_logger import CSVLogger
from src.common.logger import get_logger
from src.dms.state_machine import DMSStateMachine
from src.face.head_pose import HeadPoseEstimator
from src.fatigue.metrics import compute_ear, compute_mar

logger = get_logger("run_dms")

WINDOW_NAME = "DMS Demo"
DISPLAY_STATE = {
    DMSStateMachine.NORMAL: "NORMAL",
    DMSStateMachine.DISTRACTED: "DISTRACTED",
    DMSStateMachine.DROWSY: "DROWSY",
    DMSStateMachine.SEVERE_DROWSY: "SEVERE DROWSINESS",
    DMSStateMachine.UNAVAILABLE: "DMS UNAVAILABLE",
}
STATE_COLORS = {
    DMSStateMachine.NORMAL: (60, 190, 60),
    DMSStateMachine.DISTRACTED: (0, 190, 255),
    DMSStateMachine.DROWSY: (0, 140, 255),
    DMSStateMachine.SEVERE_DROWSY: (0, 0, 255),
    DMSStateMachine.UNAVAILABLE: (110, 110, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="笔记本摄像头 DMS 演示")
    parser.add_argument("--camera", type=int, default=None, help="摄像头 index，默认读取 configs/camera.yaml")
    parser.add_argument("--model", type=str, default="models/face_landmarker.task")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-sound", action="store_true", help="关闭 Windows 告警音")
    parser.add_argument("--no-mirror", action="store_true", help="关闭镜像显示")
    return parser.parse_args()


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(index)


def clamp_box(points, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, min(int(min(xs) * width), width - 1))
    y1 = max(0, min(int(min(ys) * height), height - 1))
    x2 = max(0, min(int(max(xs) * width), width - 1))
    y2 = max(0, min(int(max(ys) * height), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def quality_check(frame, face_box: Optional[Tuple[int, int, int, int]]):
    if face_box is None:
        return False, "no face"
    x1, y1, x2, y2 = face_box
    height, width = frame.shape[:2]
    face_width = x2 - x1
    face_height = y2 - y1
    if face_width < max(80, int(width * 0.12)) or face_height < max(80, int(height * 0.20)):
        return False, "face too small"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 18:
        return False, "too dark"
    if brightness > 245:
        return False, "too bright"
    return True, "ok"


class AlertPlayer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._playing = False
        self._last_alert = 0.0

    def play(self, state: str, now: float, changed: bool):
        if not self.enabled:
            return
        if state == DMSStateMachine.NORMAL:
            return
        cooldown = 2.5 if state == DMSStateMachine.SEVERE_DROWSY else 4.0
        if not changed and now - self._last_alert < cooldown:
            return
        if self._playing:
            return
        self._last_alert = now
        tones = {
            DMSStateMachine.DISTRACTED: (700, 120),
            DMSStateMachine.DROWSY: (900, 220),
            DMSStateMachine.SEVERE_DROWSY: (1200, 350),
            DMSStateMachine.UNAVAILABLE: (500, 180),
        }
        frequency, duration = tones[state]
        thread = threading.Thread(
            target=self._beep, args=(frequency, duration), daemon=True
        )
        thread.start()

    def _beep(self, frequency: int, duration: int):
        with self._lock:
            self._playing = True
            try:
                try:
                    import winsound
                    winsound.Beep(frequency, duration)
                except (ImportError, RuntimeError):
                    pass
            finally:
                self._playing = False


def draw_text(frame, text: str, origin, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    camera_cfg = load_config("camera")
    paths_cfg = load_config("paths")
    threshold_cfg = load_config("thresholds")

    camera_index = args.camera if args.camera is not None else camera_cfg.get("default_index", 0)
    width = int(camera_cfg.get("width", 640))
    height = int(camera_cfg.get("height", 480))
    model_path = resolve_path(args.model)
    if not model_path.exists():
        logger.error("Face Landmarker 模型不存在: %s", model_path)
        logger.error("请先把 face_landmarker.task 放到 models/ 目录")
        return 1

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        logger.error("未安装 mediapipe，请执行 pip install -r requirements/fatigue.txt")
        return 1

    try:
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:
        logger.error("Face Landmarker 初始化失败: %s", exc)
        return 1

    state_machine = DMSStateMachine(
        ear_threshold=float(threshold_cfg.get("dms_ear_threshold", 0.21)),
        mar_threshold=float(threshold_cfg.get("dms_mar_threshold", 0.55)),
        distraction_yaw=float(threshold_cfg.get("dms_distraction_yaw", 25.0)),
        distraction_pitch=float(threshold_cfg.get("dms_distraction_pitch", 20.0)),
        distraction_threshold=float(threshold_cfg.get("dms_distraction_seconds", 2.0)),
        drowsy_closed_threshold=float(threshold_cfg.get("dms_drowsy_closed_seconds", 0.8)),
        severe_closed_threshold=float(threshold_cfg.get("dms_severe_closed_seconds", 1.5)),
        yawn_threshold=float(threshold_cfg.get("dms_yawn_seconds", 1.0)),
        unavailable_threshold=float(threshold_cfg.get("dms_unavailable_seconds", 2.0)),
    )
    head_pose = HeadPoseEstimator()
    alert = AlertPlayer(enabled=not args.no_sound)

    cap = open_camera(camera_index)
    if not cap.isOpened():
        landmarker.close()
        logger.error("无法打开摄像头 index=%d", camera_index)
        logger.error("可尝试 python scripts/run_dms.py --camera 1")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    log_path = resolve_path(paths_cfg.get("logs_dir", "data/logs")) / "dms.csv"
    start_time = time.perf_counter()
    frame_count = 0
    fps = 0.0
    fps_frames = 0
    fps_started = time.perf_counter()
    mirror = not args.no_mirror
    last_pitch = None

    logger.info("DMS 已启动：camera=%d，model=%s", camera_index, model_path)
    logger.info("按 q 退出，c 校准 pitch，m 切换镜像显示")

    try:
        with CSVLogger(log_path) as csv_logger:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.error("摄像头读取失败")
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                timestamp_ms = int((now - start_time) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    logger.warning("Face Landmarker 推理失败: %s", exc)
                    result = None

                face_detected = bool(result and result.face_landmarks)
                ear = mar = yaw = pitch = roll = None
                face_box = None
                quality_reason = "no face"
                if face_detected:
                    points = [(float(p.x), float(p.y)) for p in result.face_landmarks[0]]
                    face_box = clamp_box(points, frame.shape[1], frame.shape[0])
                    quality_ok, quality_reason = quality_check(frame, face_box)
                    if quality_ok:
                        ear_l, ear_r = compute_ear(points)
                        ear = (ear_l + ear_r) / 2.0
                        mar = compute_mar(points)
                        pose_indices = [33, 263, 1, 61, 291]
                        pose_points = np.array(
                            [
                                [points[i][0] * frame.shape[1], points[i][1] * frame.shape[0]]
                                for i in pose_indices
                            ],
                            dtype=np.float64,
                        )
                        angles = head_pose.estimate(pose_points, frame.shape)
                        if angles is not None:
                            yaw, pitch, roll = angles
                            last_pitch = pitch
                else:
                    quality_ok = False

                state_info = state_machine.update(
                    timestamp=now,
                    face_detected=face_detected,
                    quality_ok=quality_ok,
                    ear=ear,
                    mar=mar,
                    yaw=yaw,
                    pitch=pitch,
                )
                state = state_info["state"]
                alert.play(state, now, bool(state_info["changed"]))

                if face_box is not None:
                    x1, y1, x2, y2 = face_box
                    box_color = STATE_COLORS[state] if state != DMSStateMachine.NORMAL else (60, 220, 60)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    for index in [33, 133, 362, 263, 61, 291, 13, 14]:
                        if index < len(points):
                            px = int(points[index][0] * frame.shape[1])
                            py = int(points[index][1] * frame.shape[0])
                            cv2.circle(frame, (px, py), 2, (255, 220, 0), -1)

                height_px, width_px = frame.shape[:2]
                banner_color = STATE_COLORS[state]
                cv2.rectangle(frame, (0, 0), (width_px, 48), banner_color, -1)
                draw_text(frame, f"DMS: {DISPLAY_STATE[state]}", (14, 32), (255, 255, 255), 0.85, 2)
                draw_text(frame, f"FPS {fps:.1f}", (width_px - 115, 30), (255, 255, 255), 0.55, 2)

                y = 75
                if ear is not None and mar is not None:
                    draw_text(frame, f"EAR {ear:.3f}   MAR {mar:.3f}", (12, y))
                    y += 24
                if yaw is not None and pitch is not None:
                    draw_text(frame, f"HEAD yaw {yaw:+.1f}  pitch {pitch:+.1f}  roll {roll:+.1f}", (12, y))
                    y += 24
                draw_text(frame, f"FACE {'OK' if face_detected else 'NOT FOUND'}   IMAGE {quality_reason.upper()}", (12, y))
                y += 24
                draw_text(
                    frame,
                    f"closed {state_info['closed_seconds']:.1f}s  distracted {state_info['distraction_seconds']:.1f}s",
                    (12, y),
                )
                y += 28
                draw_text(frame, f"Hint: {DISPLAY_STATE[state]}", (12, y), banner_color, 0.65, 2)
                draw_text(frame, "q quit | c calibrate pitch | m mirror", (12, height_px - 14), (220, 220, 220), 0.5, 1)

                csv_logger.log("dms", "state", state, state_info["message"])
                csv_logger.log("dms", "face_detected", int(face_detected), quality_reason)
                csv_logger.log("dms", "quality_ok", int(quality_ok), quality_reason)
                if ear is not None:
                    csv_logger.log("dms", "ear", f"{ear:.5f}", "")
                    csv_logger.log("dms", "mar", f"{mar:.5f}", "")
                if yaw is not None:
                    csv_logger.log("dms", "head_yaw", f"{yaw:.3f}", "")
                    csv_logger.log("dms", "head_pitch", f"{pitch:.3f}", "")
                if state_info["changed"]:
                    csv_logger.log("dms", "event", state, state_info["message"])
                    logger.info("DMS 状态：%s", DISPLAY_STATE[state])

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    mirror = not mirror
                if key == ord("c") and last_pitch is not None:
                    raw_pitch = last_pitch + head_pose.get_pitch_bias()
                    head_pose.set_pitch_bias(raw_pitch)
                    logger.info("pitch 校准完成：当前姿态设为 0 度")

                frame_count += 1
                fps_frames += 1
                fps_elapsed = time.perf_counter() - fps_started
                if fps_elapsed >= 1.0:
                    fps = fps_frames / fps_elapsed
                    fps_frames = 0
                    fps_started = time.perf_counter()
                if args.max_frames and frame_count >= args.max_frames:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    logger.info("DMS 已退出，共处理 %d 帧，日志：%s", frame_count, log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
