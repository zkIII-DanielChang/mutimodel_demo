"""疲劳检测 + 表情识别独立窗口。

用法：
    python scripts/run_fatigue.py --camera 0
    python scripts/run_fatigue.py --camera 0 --max-frames 300
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.common.config_loader import load_config, resolve_path
from src.common.csv_logger import CSVLogger
from src.common.logger import get_logger

logger = get_logger("run_fatigue")

DEFAULT_MODEL = "models/face_landmarker.task"


def parse_args():
    p = argparse.ArgumentParser(description="疲劳检测 + 表情识别")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--model", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cam_cfg = load_config("camera")
    paths_cfg = load_config("paths")

    camera_index = args.camera if args.camera is not None else cam_cfg.get("default_index", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)

    model_path = resolve_path(args.model or DEFAULT_MODEL)
    if not model_path.exists():
        logger.error(
            "Face Landmarker 模型不存在: %s\n"
            "请下载 face_landmarker.task 并放到 models/ 目录。\n"
            "下载地址: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            model_path,
        )
        return 1

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        logger.error("未安装 mediapipe，请执行 pip install mediapipe")
        return 1

    from src.fatigue.metrics import BlinkCounter, compute_ear, compute_mar
    from src.fatigue.state_classifier import FatigueStateClassifier
    from src.fatigue.emotion import classify_emotion

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    logger.info("Face Landmarker 已加载: %s", model_path)

    log_path = resolve_path(paths_cfg.get("logs_dir", "data/logs")) / "fatigue.csv"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("无法打开摄像头 index=%d", camera_index)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    WINDOW_NAME = "Fatigue & Emotion"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    blink = BlinkCounter()
    classifier = FatigueStateClassifier()
    start_time = time.time()
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_frames = 0

    try:
        with CSVLogger(log_path) as csv_logger:
            logger.info("CSV 日志: %s", log_path)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = frame[:, :, ::-1].copy()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start_time) * 1000)
                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as e:
                    logger.warning("Face Landmarker 推理失败: %s", e)
                    result = None

                if result and result.face_landmarks:
                    lm = result.face_landmarks[0]
                    pts = [(p.x, p.y) for p in lm]
                    h, w = frame.shape[:2]
                    pix = [(int(x * w), int(y * h)) for x, y in pts]

                    ear_l, ear_r = compute_ear(pts)
                    ear = (ear_l + ear_r) / 2.0
                    mar = compute_mar(pts)
                    blink.update(ear)

                    timestamp = time.time()
                    state = classifier.update(ear, mar, timestamp)
                    emotion = classify_emotion(
                        result.face_blendshapes[0] if result.face_blendshapes else None
                    )

                    for idx in [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]:
                        if idx < len(pix):
                            cv2.circle(frame, pix[idx], 2, (0, 255, 0), -1)
                    for idx in [61, 291, 13, 14, 81, 178, 78, 308]:
                        if idx < len(pix):
                            cv2.circle(frame, pix[idx], 2, (0, 0, 255), -1)

                    lines = [
                        f"EAR {ear:.3f}  MAR {mar:.3f}",
                        f"blink {blink.count}  closed {state['closed_seconds']:.1f}s",
                        f"state: {state['state']} (score {state['score']:.1f})",
                        f"emotion: {emotion['label']} {emotion['score']:.2f}",
                    ]
                    y0 = 60
                    for i, text in enumerate(lines):
                        cv2.putText(
                            frame, text, (10, y0 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        )

                    csv_logger.log("fatigue", "ear", f"{ear:.4f}", "")
                    csv_logger.log("fatigue", "mar", f"{mar:.4f}", "")
                    csv_logger.log("fatigue", "blink_count", str(blink.count), "")
                    csv_logger.log("fatigue", "state", state["state"],
                                   f"{state['score']:.2f}")
                    csv_logger.log("fatigue", "emotion", emotion["label"],
                                   f"{emotion['score']:.2f}")
                else:
                    cv2.putText(
                        frame, "No face detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )

                fps_frames += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps = fps_frames / (now - fps_timer)
                    fps_frames = 0
                    fps_timer = now

                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                )
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_count += 1
                if args.max_frames and frame_count >= args.max_frames:
                    logger.info("达到 max_frames=%d，退出", args.max_frames)
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    logger.info("共处理 %d 帧", frame_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())# 清理残留
