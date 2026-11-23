"""声纹注册与实时识别。

用法：
    # 注册：对着麦克风说 3 段话（每段自动检测静音后保存）
    python scripts/run_voiceprint.py --mode enroll --name alice --clips 3

    # 识别：实时说话，打印说话人 + 相似度
    python scripts/run_voiceprint.py --mode identify

按 Ctrl+C 退出。
"""

import argparse
import queue
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.common.config_loader import load_config, resolve_path
from src.common.csv_logger import CSVLogger
from src.common.logger import get_logger

logger = get_logger("run_voiceprint")

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
SILENCE_THRESHOLD = 0.012
SILENCE_SECONDS = 0.8
MAX_CLIP_SECONDS = 6.0


def parse_args():
    p = argparse.ArgumentParser(description="声纹注册与实时识别")
    p.add_argument("--mode", choices=["enroll", "identify"], required=True)
    p.add_argument("--name", type=str, default=None, help="注册模式必填")
    p.add_argument("--clips", type=int, default=3, help="注册时采集段数")
    p.add_argument("--threshold", type=float, default=0.40, help="判定阈值")
    return p.parse_args()


def record_utterances(sd, target_clips: int):
    """从麦克风采集，VAD 切出 target_clips 段语音，返回 float32 数组列表。"""
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE,
        dtype="float32", callback=callback,
    )
    stream.start()
    silence_needed = int(SILENCE_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
    max_frames = int(MAX_CLIP_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

    clips = []
    buffer = []
    in_speech = False
    silence_count = 0

    try:
        while len(clips) < target_clips:
            try:
                block = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(block ** 2)))
            if rms > SILENCE_THRESHOLD:
                in_speech = True
                silence_count = 0
                buffer.append(block)
            elif in_speech:
                silence_count += 1
                buffer.append(block)
                if silence_count >= silence_needed or len(buffer) >= max_frames:
                    clip = np.concatenate(buffer).astype(np.float32)
                    clips.append(clip)
                    logger.info("已采集 %d/%d 段，时长 %.2fs",
                                len(clips), target_clips, len(clip) / SAMPLE_RATE)
                    buffer = []
                    in_speech = False
                    silence_count = 0
    finally:
        stream.stop()
        stream.close()
    return clips


def main() -> int:
    args = parse_args()
    paths_cfg = load_config("paths")

    try:
        import sounddevice as sd
    except ImportError:
        logger.error("未安装 sounddevice")
        return 1

    try:
        from src.voiceprint.engine import VoiceprintEngine
    except Exception as e:
        logger.error("声纹引擎导入失败: %s", e)
        return 1

    try:
        engine = VoiceprintEngine(threshold=args.threshold)
    except Exception as e:
        logger.error("声纹引擎初始化失败: %s", e)
        return 1

    if args.mode == "enroll":
        if not args.name:
            logger.error("enroll 模式必须指定 --name")
            return 1
        logger.info("开始注册 %s，请连续说 %d 段话", args.name, args.clips)
        clips = record_utterances(sd, args.clips)
        n = engine.enroll_from_audio(args.name, clips, SAMPLE_RATE)
        if n == 0:
            logger.error("注册失败，未采集到有效音频")
            return 1
        logger.info("注册完成，使用 %d 段音频", n)
        return 0

    # identify 模式
    log_path = resolve_path(paths_cfg.get("logs_dir", "data/logs")) / "voiceprint.csv"
    logger.info("开始实时识别，请说话。按 Ctrl+C 退出。")

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE,
        dtype="float32", callback=callback,
    )
    stream.start()
    silence_needed = int(SILENCE_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
    max_frames = int(MAX_CLIP_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

    buffer = []
    in_speech = False
    silence_count = 0

    try:
        with CSVLogger(log_path) as csv_logger:
            logger.info("CSV 日志: %s", log_path)
            while True:
                try:
                    block = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(block ** 2)))
                if rms > SILENCE_THRESHOLD:
                    in_speech = True
                    silence_count = 0
                    buffer.append(block)
                elif in_speech:
                    silence_count += 1
                    buffer.append(block)
                    if silence_count >= silence_needed or len(buffer) >= max_frames:
                        audio = np.concatenate(buffer).astype(np.float32)
                        buffer = []
                        in_speech = False
                        silence_count = 0

                        result = engine.identify_audio(audio, SAMPLE_RATE)
                        if result is None:
                            logger.info("[无有效声纹结果]")
                            continue
                        if result["accepted"]:
                            logger.info("说话人: %s | 相似度: %.3f | 判定: matched",
                                        result["name"], result["score"])
                        else:
                            logger.info("说话人: Unknown | 相似度: %.3f | 判定: unknown",
                                        result["score"])
                        csv_logger.log("voiceprint", "speaker", result["name"],
                                       "matched" if result["accepted"] else "unknown")
                        csv_logger.log("voiceprint", "similarity",
                                       f"{result['score']:.4f}", "")
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        stream.stop()
        stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())