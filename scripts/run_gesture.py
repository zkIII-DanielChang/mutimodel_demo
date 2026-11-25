"""手势识别独立窗口。

用法：
    python scripts/run_gesture.py --camera 0
    python scripts/run_gesture.py --camera 0 --max-frames 300
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.common.config_loader import load_config, resolve_path
from src.common.csv_logger import CSVLogger
from src.common.logger import get_logger

logger = get_logger("run_gesture")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def parse_args():
    p = argparse.ArgumentParser(description="手势识别独立窗口")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--model", type=str, default=None, help="gesture_recognizer.task 路径")
    p.add_argument("--num-hands", type=int, default=2)
    return p.parse_args()


def draw_hands(frame, hands):
    h, w = frame.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand["landmarks"]]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        if pts:
            label = f"{hand['label']} {hand['score']:.2f} ({hand['handedness']})"
            cv2.putText(
                frame, label, (pts[0][0], max(pts[0][1] - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )


def main() -> int:
    args = parse_args()
    cam_cfg = load_config("camera")
    paths_cfg = load_config("paths")

    camera_index = args.camera if args.camera is not None else cam_cfg.get("default_index", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)

    try:
        from src.gesture.hand_recognizer import HandGestureRecognizer

        recognizer = HandGestureRecognizer(
            model_path=args.model,
            num_hands=args.num_hands,
        )
    except Exception as e:
        logger.error("手势识别器初始化失败: %s", e)
        return 1

    log_path = resolve_path(paths_cfg.get("logs_dir", "data/logs")) / "gesture.csv"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("无法打开摄像头 index=%d", camera_index)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    WINDOW_NAME = "Gesture Recognition"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_frames = 0
    start_time = time.time()

    try:
        with CSVLogger(log_path) as csv_logger:
            logger.info("CSV 日志: %s", log_path)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_ms = int((time.time() - start_time) * 1000)
                try:
                    hands = recognizer.process(frame, timestamp_ms)
                except Exception as e:
                    logger.warning("手势识别失败: %s", e)
                    hands = []

                draw_hands(frame, hands)
                for hand in hands:
                    csv_logger.log(
                        "gesture", "label", hand["label"], f"{hand['score']:.3f}"
                    )

                fps_frames += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps = fps_frames / (now - fps_timer)
                    fps_frames = 0
                    fps_timer = now

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | hands: {len(hands)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
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
        recognizer.close()

    logger.info("共处理 %d 帧", frame_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())