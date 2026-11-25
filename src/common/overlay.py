"""人脸检测最小闭环，可选识别、头部姿态、眼动。原生 OpenCV 绘制版。"""

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
from src.common.device import describe_devices, has_cuda
from src.common.logger import get_logger
from src.face.detector import create_detector

logger = get_logger("run_face")


def parse_args():
    parser = argparse.ArgumentParser(description="人脸检测最小闭环（可选识别/姿态/眼动）")
    parser.add_argument(
        "--backend", choices=["uniface", "opencv-haar"], default="uniface",
    )
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prefer-cuda", action="store_true")
    parser.add_argument("--recognize", action="store_true")
    parser.add_argument("--head-pose", action="store_true")
    parser.add_argument("--gaze", action="store_true")
    parser.add_argument("--gaze-interval", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cam_cfg = load_config("camera")
    paths_cfg = load_config("paths")
    threshold_cfg = load_config("thresholds")

    camera_index = args.camera if args.camera is not None else cam_cfg.get("default_index", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)

    logger.info("设备信息: %s", describe_devices())
    logger.info("CUDA 可用: %s", has_cuda())

    try:
        detector = create_detector(
            args.backend,
            score_threshold=threshold_cfg.get("face_detection_score", 0.5),
            prefer_cuda=args.prefer_cuda,
        )
    except Exception as e:
        logger.error("检测器初始化失败: %s", e)
        return 1

    recognizer = None
    gallery = None
    if args.recognize:
        if args.backend != "uniface":
            logger.error("识别模式需要 landmarks，仅支持 --backend uniface")
            return 1
        try:
            from src.face.recognizer import ArcFaceRecognizer
            from src.face.gallery import FaceGallery

            recognizer = ArcFaceRecognizer(prefer_cuda=args.prefer_cuda)
            gallery_path = (
                resolve_path(paths_cfg.get("registered_faces_dir", "data/registered_faces"))
                / "face_gallery.npz"
            )
            gallery = FaceGallery(gallery_path)
            gallery.load()
        except Exception as e:
            logger.error("识别器/注册库初始化失败: %s", e)
            return 1

    head_pose_estimator = None
    if args.head_pose:
        try:
            from src.face.head_pose import HeadPoseEstimator
            head_pose_estimator = HeadPoseEstimator()
            logger.info("头部姿态估计已启用，按 c 键可校准 pitch 零偏")
        except Exception as e:
            logger.error("头部姿态初始化失败: %s", e)
            return 1

    gaze_estimator = None
    if args.gaze:
        try:
            from src.face.gaze import GazeEstimator
            gaze_estimator = GazeEstimator(prefer_cuda=args.prefer_cuda)
            if not gaze_estimator.is_available():
                logger.warning("眼动估计器不可用，窗口将显示 gaze: unavailable")
        except Exception as e:
            logger.error("眼动初始化失败: %s", e)
            gaze_estimator = None

    log_path = resolve_path(paths_cfg.get("logs_dir", "data/logs")) / "face_detection.csv"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("无法打开摄像头 index=%d", camera_index)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    WINDOW_NAME = "Face Detection"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_frames = 0
    similarity_threshold = threshold_cfg.get("face_recognition_similarity", 0.6)
    last_pitch = None
    gaze_cache = {}
    gaze_interval = max(1, args.gaze_interval)

    try:
        with CSVLogger(log_path) as csv_logger:
            logger.info("CSV 日志: %s", log_path)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                faces = detector.detect(frame)
                for f in faces:
                    x1, y1, x2, y2 = f["bbox"]
                    label = f"face {f['score']:.2f}"
                    color = (0, 255, 0)
                    extra_lines = []

                    if (
                        recognizer is not None
                        and gallery is not None
                        and f.get("landmarks") is not None
                    ):
                        try:
                            emb = recognizer.extract(frame, f["landmarks"])
                            r = gallery.search(emb, threshold=similarity_threshold)
                            label = f"{r['name']} {r['similarity']:.2f}"
                            color = (0, 255, 0) if r["matched"] else (0, 165, 255)
                            csv_logger.log(
                                "face", "identity", r["name"],
                                "matched" if r["matched"] else "unknown",
                            )
                            csv_logger.log(
                                "face", "similarity", f"{r['similarity']:.4f}", "",
                            )
                        except Exception as e:
                            logger.warning("识别失败: %s", e)

                    if head_pose_estimator is not None and f.get("landmarks") is not None:
                        angles = head_pose_estimator.estimate(f["landmarks"], frame.shape)
                        if angles is not None:
                            yaw, pitch, roll = angles
                            last_pitch = pitch
                            extra_lines.append(
                                f"pose Y{yaw:+.0f} P{pitch:+.0f} R{roll:+.0f}"
                            )
                            csv_logger.log("face", "head_yaw", f"{yaw:.2f}", "")
                            csv_logger.log("face", "head_pitch", f"{pitch:.2f}", "")
                            csv_logger.log("face", "head_roll", f"{roll:.2f}", "")

                    if gaze_estimator is not None:
                        if gaze_estimator.is_available():
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            key = (cx // 40, cy // 40, (x2 - x1) // 40, (y2 - y1) // 40)
                            cached = gaze_cache.get(key)
                            should_run = (frame_count % gaze_interval == 0) or cached is None
                            if should_run:
                                face_crop = frame[y1:y2, x1:x2]
                                if face_crop.size > 0:
                                    g = gaze_estimator.estimate(face_crop)
                                    if g is not None:
                                        gaze_cache[key] = g
                                        cached = g
                                    else:
                                        cached = None
                            if cached is not None:
                                extra_lines.append(
                                    f"gaze Y{cached['yaw']:+.1f} P{cached['pitch']:+.1f}"
                                )
                                csv_logger.log("face", "gaze_yaw", f"{cached['yaw']:.2f}", "")
                                csv_logger.log("face", "gaze_pitch", f"{cached['pitch']:.2f}", "")
                            else:
                                extra_lines.append("gaze: no result")
                        else:
                            extra_lines.append("gaze: unavailable")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )
                    for i, line in enumerate(extra_lines):
                        cv2.putText(
                            frame, line,
                            (x1, min(y2 + 18 + i * 18, frame.shape[0] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        )

                    csv_logger.log("face", "detection", f"{f['score']:.3f}", "detected")

                fps_frames += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    fps = fps_frames / (now - fps_timer)
                    fps_frames = 0
                    fps_timer = now

                hint = " | c: calibrate pitch" if head_pose_estimator is not None else ""
                cv2.putText(
                    frame, f"FPS: {fps:.1f} | faces: {len(faces)}{hint}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and head_pose_estimator is not None and last_pitch is not None:
                    raw_pitch = last_pitch + head_pose_estimator.get_pitch_bias()
                    head_pose_estimator.set_pitch_bias(raw_pitch)
                    logger.info("pitch 零偏校准完成: %.2f 度（原始 pitch）", raw_pitch)

                frame_count += 1
                if args.max_frames and frame_count >= args.max_frames:
                    logger.info("达到 max_frames=%d，退出", args.max_frames)
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info("共处理 %d 帧", frame_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())