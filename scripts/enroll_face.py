"""人脸注册：摄像头采集多帧，平均嵌入后写入注册库。

用法：
    python scripts/enroll_face.py --name alice --camera 0 --num-samples 15
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.common.config_loader import load_config, resolve_path
from src.common.logger import get_logger
from src.face.detector import create_detector
from src.face.gallery import FaceGallery
from src.face.recognizer import ArcFaceRecognizer

logger = get_logger("enroll_face")


def parse_args():
    p = argparse.ArgumentParser(description="人脸注册")
    p.add_argument("--name", required=True, help="注册姓名")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=15, help="采集帧数，取平均嵌入")
    p.add_argument("--prefer-cuda", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cam_cfg = load_config("camera")
    paths_cfg = load_config("paths")
    threshold_cfg = load_config("thresholds")

    camera_index = args.camera if args.camera is not None else cam_cfg.get("default_index", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)

    try:
        detector = create_detector(
            "uniface",
            score_threshold=threshold_cfg.get("face_detection_score", 0.5),
            prefer_cuda=args.prefer_cuda,
        )
        recognizer = ArcFaceRecognizer(prefer_cuda=args.prefer_cuda)
    except Exception as e:
        logger.error("模型初始化失败: %s", e)
        return 1

    gallery_path = (
        resolve_path(paths_cfg.get("registered_faces_dir", "data/registered_faces"))
        / "face_gallery.npz"
    )
    gallery = FaceGallery(gallery_path)
    gallery.load()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("无法打开摄像头 index=%d", camera_index)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    collected = []
    logger.info("开始采集 %d 帧，请正对摄像头，按 q 提前结束", args.num_samples)
    try:
        while len(collected) < args.num_samples:
            ok, frame = cap.read()
            if not ok:
                break
            faces = detector.detect(frame)
            if len(faces) == 1:
                f = faces[0]
                x1, y1, x2, y2 = f["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"enroll {len(collected)+1}/{args.num_samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                try:
                    emb = recognizer.extract(frame, f["landmarks"])
                    collected.append(emb)
                except Exception as e:
                    logger.warning("本帧嵌入提取失败: %s", e)
            elif len(faces) > 1:
                cv2.putText(
                    frame,
                    "Multiple faces - only one person",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow("Enroll Face", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not collected:
        logger.error("未采集到有效嵌入，注册失败")
        return 1

    mean_emb = np.mean(np.stack(collected, axis=0), axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
    gallery.enroll(args.name, mean_emb)
    gallery.save()
    logger.info("注册完成: %s，使用 %d 帧，库内共 %d 人", args.name, len(collected), len(gallery))
    return 0


if __name__ == "__main__":
    sys.exit(main())