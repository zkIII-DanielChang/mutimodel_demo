"""语音识别 + 指令匹配。

用法：
    python scripts/run_asr.py
    python scripts/run_asr.py --threshold 0.5 --max-utterances 10
按 Ctrl+C 退出。
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.common.config_loader import load_config, resolve_path
from src.common.csv_logger import CSVLogger
from src.common.logger import get_logger

logger = get_logger("run_asr")

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
SILENCE_THRESHOLD = 0.012
SILENCE_SECONDS = 0.8
MAX_UTTERANCE_SECONDS = 10.0


def parse_args():
    p = argparse.ArgumentParser(description="语音识别 + 指令匹配")
    p.add_argument("--threshold", type=float, default=0.5, help="指令匹配阈值")
    p.add_argument("--max-utterances", type=int, default=0, help="0 表示不限")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths_cfg = load_config("paths")

    try:
        import sounddevice as sd
    except ImportError:
        logger.error("未安装 sounddevice，请执行 pip install sounddevice")
        return 1

    try:
        from src.asr.recognizer import ASRRecognizer
        from src.asr.command_matcher import CommandMatcher
    except Exception as e:
        logger.error("ASR 模块导入失败: %s", e)
        return 1

    # 加载指令
    try:
        cmd_cfg = load_config("commands")
        commands = cmd_cfg.get("commands", [])
    except Exception as e:
        logger.error("指令配置加载失败: %s", e)
        return 1
    matcher = CommandMatcher(commands)

    try:
        recognizer = ASRRecognizer()
    except Exception as e:
        logger.error("ASR 初始化失败: %s", e)
        return 1

    log_path = resolve_path(paths_cfg.get("logs_dir", "data/logs")) / "asr.csv"

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            logger.debug("音频状态: %s", status)
        audio_q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=callback,
    )
    stream.start()
    logger.info("麦克风已启动，请说话。按 Ctrl+C 退出。")

    silence_frames_needed = int(SILENCE_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
    max_frames = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

    speech_buffer = []
    in_speech = False
    silence_count = 0
    utterance_count = 0

    try:
        with CSVLogger(log_path) as csv_logger:
            while True:
                try:
                    block = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(block ** 2)))
                if rms > SILENCE_THRESHOLD:
                    in_speech = True
                    silence_count = 0
                    speech_buffer.append(block)
                else:
                    if in_speech:
                        silence_count += 1
                        speech_buffer.append(block)
                        if (
                            silence_count >= silence_frames_needed
                            or len(speech_buffer) >= max_frames
                        ):
                            audio = np.concatenate(speech_buffer).astype(np.float32)
                            speech_buffer = []
                            in_speech = False
                            silence_count = 0

                            text = recognizer.recognize(audio, SAMPLE_RATE)
                            if not text:
                                logger.info("[空识别]")
                                continue

                            m = matcher.match(text, threshold=args.threshold)
                            if m["matched"]:
                                logger.info(
                                    "识别: %s | 指令: %s | 相似度: %.3f",
                                    text, m["command"], m["similarity"],
                                )
                            else:
                                logger.info(
                                    "识别: %s | 未匹配（最高 %.3f）",
                                    text, m["similarity"],
                                )

                            csv_logger.log("asr", "text", text, "")
                            csv_logger.log(
                                "asr", "command",
                                m["command"] or "",
                                "matched" if m["matched"] else "unmatched",
                            )
                            csv_logger.log(
                                "asr", "similarity",
                                f"{m['similarity']:.4f}", "",
                            )

                            utterance_count += 1
                            if (
                                args.max_utterances
                                and utterance_count >= args.max_utterances
                            ):
                                logger.info(
                                    "达到 max_utterances=%d，退出", args.max_utterances
                                )
                                return 0
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        stream.stop()
        stream.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())