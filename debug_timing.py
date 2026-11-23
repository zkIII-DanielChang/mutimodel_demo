"""分模块耗时诊断：读帧 / 检测 / 眼动。"""

import time

import cv2

from src.face.detector import create_detector
from src.face.gaze import GazeEstimator


def main():
    print("初始化检测器...")
    det = create_detector("uniface")
    print("初始化眼动...")
    gz = GazeEstimator()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    if not cap.isOpened():
        print("摄像头打开失败")
        return

    # 预热
    for _ in range(10):
        cap.read()

    n = 50
    frames = []
    t = time.time()
    for _ in range(n):
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    t_read = time.time() - t
    print(f"读帧: {t_read / max(len(frames), 1) * 1000:.1f} ms/帧, 有效帧 {len(frames)}")

    # 检测
    t = time.time()
    all_faces = [det.detect(fr) for fr in frames]
    t_det = time.time() - t
    total_faces = sum(len(x) for x in all_faces)
    print(f"检测: {t_det / max(len(frames), 1) * 1000:.1f} ms/帧, 共检出 {total_faces} 张脸")

    # 收集裁剪图
    crops = []
    for fr, faces in zip(frames, all_faces):
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            crop = fr[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
    print(f"收集到 {len(crops)} 张人脸裁剪图")

    # 眼动
    if crops and gz.is_available():
        sample = crops[:20]
        t = time.time()
        results = [gz.estimate(c) for c in sample]
        t_gaze = time.time() - t
        valid = sum(1 for r in results if r is not None)
        print(f"眼动: {t_gaze / max(len(sample), 1) * 1000:.1f} ms/次, 有效 {valid}/{len(sample)}")
    else:
        print("眼动不可用或无裁剪图，跳过")

    cap.release()
    print("\n单帧预估（检测 + 每 3 帧一次眼动）:")
    est = t_read / max(len(frames), 1) * 1000 + t_det / max(len(frames), 1) * 1000
    if crops and gz.is_available():
        est += t_gaze / max(len(sample), 1) * 1000 / 3
    print(f"  ≈ {est:.1f} ms/帧 → 理论 FPS ≈ {1000 / est:.1f}")


if __name__ == "__main__":
    main()