"""通过麦克风注册与识别本地说话人模板。"""

import argparse
import queue
import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.common.config_loader import load_config, resolve_path
from src.common.csv_logger import CSVLogger
from src.common.logger import get_logger
from src.voiceprint.engine import SAMPLE_RATE, VoiceprintEngine

logger = get_logger("voiceprint_demo")
BLOCK_SIZE = 1024


class UtteranceBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.blocks = []
        self.total = self.voiced = self.silence = 0

    def feed(self, block):
        rms = float(np.std(block))
        voiced = rms >= (0.008 if self.blocks else 0.012)
        if not voiced and not self.blocks:
            return None
        self.blocks.append(block)
        self.total += len(block)
        if voiced:
            self.voiced += len(block)
            self.silence = 0
        else:
            self.silence += len(block)
        if self.silence < 0.6 * SAMPLE_RATE and self.total < 6 * SAMPLE_RATE:
            return None
        audio = np.concatenate(self.blocks)
        if self.silence:
            audio = audio[:-self.silence]
        valid = self.voiced >= 0.65 * SAMPLE_RATE and VoiceprintEngine._prepare_audio(audio, SAMPLE_RATE) is not None
        self.reset()
        return audio if valid else np.empty(0, dtype=np.float32)


class VoiceprintDemo:
    def __init__(self, root, engine, sd, default_clips=3):
        self.root = root
        self.engine = engine
        self.audio_q = queue.Queue(maxsize=32)
        self.sequence = 0
        self.last_block = None
        self.segment = UtteranceBuffer()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.job = None
        self.job_kind = None
        self.confirming_clear = False
        self.enroll_name = None
        self.clips = []
        self.stream = None
        self._build(default_clips)
        self._refresh_names()

        paths = load_config("paths")
        self.csv = CSVLogger(resolve_path(paths.get("logs_dir", "data/logs")) / "voiceprint.csv")
        try:
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE,
                                         dtype="float32", callback=self._audio_callback)
            self.stream.start()
        except Exception:
            self._close()
            raise
        self.root.after(40, self._poll)

    def _build(self, default_clips):
        self.root.title("声纹注册与识别演示")
        self.root.geometry("610x485")
        self.root.minsize(580, 430)
        body = ttk.Frame(self.root, padding=16)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="声纹识别", font=("Microsoft YaHei", 17, "bold")).pack(anchor="w")
        ttk.Label(body, text="本地演示 · 不用于车辆解锁或身份认证").pack(anchor="w", pady=(0, 10))
        self.result = tk.StringVar()
        self.detail = tk.StringVar()
        ttk.Label(body, textvariable=self.result, font=("Microsoft YaHei", 14, "bold")).pack(anchor="w")
        ttk.Label(body, textvariable=self.detail, wraplength=530).pack(anchor="w", pady=(2, 6))

        enroll = ttk.LabelFrame(body, text="多段注册", padding=10)
        enroll.pack(fill="x", pady=(0, 8))
        self.name = tk.StringVar()
        self.count = tk.IntVar(value=default_clips)
        row = ttk.Frame(enroll)
        row.pack(fill="x")
        ttk.Label(row, text="姓名").pack(side="left")
        ttk.Entry(row, textvariable=self.name, width=16).pack(side="left", padx=5)
        ttk.Label(row, text="段数").pack(side="left")
        ttk.Spinbox(row, from_=2, to=10, textvariable=self.count, width=4).pack(side="left", padx=5)
        self.enroll_button = ttk.Button(row, text="开始注册", command=self._start_enroll)
        self.enroll_button.pack(side="left", padx=4)
        self.cancel_button = ttk.Button(row, text="取消", state="disabled", command=self._cancel_enroll)
        self.cancel_button.pack(side="left")
        self.progress = tk.StringVar(value="建议录 3 段不同内容和语速，每段 2–4 秒，段间停顿约 1 秒。")
        ttk.Label(enroll, textvariable=self.progress, wraplength=500).pack(anchor="w", pady=(6, 0))

        self.names = tk.Listbox(body, height=4)
        self.names.pack(fill="both", expand=True)
        tools = ttk.Frame(body)
        tools.pack(fill="x", pady=(8, 0))
        ttk.Label(tools, text="相似度阈值").pack(side="left")
        self.threshold = tk.StringVar(value=f"{self.engine.get_threshold():.3f}")
        self.threshold_entry = ttk.Entry(tools, textvariable=self.threshold, width=6)
        self.threshold_entry.pack(side="left", padx=4)
        self.apply_button = ttk.Button(tools, text="应用", command=self._apply_threshold)
        self.apply_button.pack(side="left")
        self.calibrate_button = ttk.Button(tools, text="样本校准", command=self._calibrate)
        self.calibrate_button.pack(side="left", padx=4)
        self.clear_button = ttk.Button(tools, text="清空模板", command=self._clear)
        self.clear_button.pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _refresh_names(self):
        names = self.engine.list_speakers()
        self.names.delete(0, tk.END)
        self.names.insert(tk.END, *(names or ["（尚未注册）"]))
        self.result.set("等待说话" if names else "请先注册说话人")

    def _set_controls(self):
        collecting = self.enroll_name is not None
        idle = self.job is None and not collecting
        for button in (self.enroll_button, self.apply_button, self.calibrate_button, self.clear_button):
            button.configure(state="normal" if idle else "disabled")
        self.cancel_button.configure(state="normal" if collecting and self.job is None else "disabled")

    def _audio_callback(self, data, frames, time_info, status):
        self.sequence += 1 + bool(status.input_overflow)
        try:
            self.audio_q.put_nowait((self.sequence, data[:, 0].copy()))
        except queue.Full:
            pass

    def _reset_capture(self):
        self.segment.reset()
        self.last_block = None
        for _ in range(self.audio_q.maxsize):
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def _apply_threshold(self):
        try:
            self.engine.set_threshold(self.threshold.get())
        except ValueError as exc:
            self.detail.set(str(exc))
        else:
            self.detail.set("阈值已应用，仅在本次运行生效。")

    def _start_enroll(self):
        name = self.name.get().strip()
        try:
            count = self.count.get()
        except tk.TclError:
            count = 0
        if not name or not 2 <= count <= 10:
            self.progress.set("请输入姓名，采集段数应为 2–10。")
            return
        self.enroll_name = name
        self.target = count
        self.clips = []
        self._reset_capture()
        self.result.set("正在采集注册语音")
        self.detail.set("")
        self.progress.set(f"{name}：0/{count}，请说第一段。")
        self._set_controls()

    def _cancel_enroll(self):
        self.enroll_name = None
        self.clips = []
        self.progress.set("已取消注册，未保存样本。")
        self._reset_capture()
        self._refresh_names()
        self._set_controls()

    def _submit(self, kind, fn, *args):
        self.job_kind = kind
        self.job = self.worker.submit(fn, *args)
        self._set_controls()

    def _calibrate(self):
        self.detail.set("正在使用已注册语音计算演示阈值…")
        self._submit("calibrate", self.engine.calibrate_threshold)

    def _clear(self):
        self.confirming_clear = True
        try:
            confirmed = messagebox.askyesno("清空模板", "确定删除本机所有已注册声纹模板？", parent=self.root)
        finally:
            self.confirming_clear = False
        self._reset_capture()
        if confirmed:
            self._submit("clear", self.engine.clear)

    def _accept_segment(self, audio):
        if not audio.size:
            target = self.progress if self.enroll_name else self.detail
            target.set("有效语音过短、音量过低或削波，请用正常音量说一整句。")
            if not self.enroll_name:
                self.result.set("本段未识别")
                self.csv.log("voiceprint", "speaker", "—", "invalid_audio")
                self.csv.log("voiceprint", "similarity", "", "invalid_audio")
            return
        if self.enroll_name:
            self.clips.append(audio)
            n = len(self.clips)
            if n < self.target:
                self.progress.set(f"已采集 {n}/{self.target} 段，请换一句话。")
            else:
                self.progress.set("正在检查样本一致性并生成模板…")
                self._submit("enroll", self.engine.enroll_from_audio, self.enroll_name, self.clips)
        elif self.engine.speaker_count():
            self.result.set("正在识别…")
            self.detail.set("")
            self._submit("identify", self.engine.identify_audio, audio)

    def _complete_job(self):
        job, kind = self.job, self.job_kind
        self.job = None
        try:
            value = job.result()
            if kind == "identify":
                name, score, reason = "—", "", "no_result"
                if value:
                    name = value["name"] if value["accepted"] else "未知说话人"
                    score = f"{value['score']:.3f}"
                    reason = value["reason"]
                    note = {"matched": "匹配", "below_threshold": "未达阈值", "ambiguous": "候选分数过近"}[reason]
                    self.result.set(name)
                    self.detail.set(f"{note} · 相似度 {score} · 阈值 {value['threshold']:.3f}")
                else:
                    self.result.set("未获得有效声纹")
                    self.detail.set("请正常说一整句后停顿。")
                self.csv.log("voiceprint", "speaker", name, reason)
                self.csv.log("voiceprint", "similarity", score, reason)
            elif kind == "enroll":
                self._refresh_names()
                self.progress.set(f"注册完成，已保存 {value} 段。" if value else
                                  "注册未保存：请确认由同一人独立录制，检查音量后重试。")
            elif kind == "calibrate":
                if value is None:
                    self.detail.set("无法校准：至少需要两位说话人，每人两段有效样本。")
                else:
                    self.threshold.set(f"{self.engine.get_threshold():.3f}")
                    self.detail.set("已应用本次演示阈值；请用未参与注册的语音复测。")
            else:
                self._refresh_names()
                self.detail.set("声纹模板已清空。")
                self.csv.log("voiceprint", "speaker", "—", "cleared")
                self.csv.log("voiceprint", "similarity", "", "cleared")
        except Exception as exc:
            logger.exception("声纹操作失败")
            self.result.set("操作失败")
            self.detail.set(str(exc))
            if kind == "enroll":
                self.progress.set("注册失败，请重试。")
        finally:
            if kind == "enroll":
                self.enroll_name = None
                self.clips = []
            self._reset_capture()
            self._set_controls()

    def _poll(self):
        if self.job is not None and self.job.done():
            self._complete_job()
        if self.job is None and not self.confirming_clear:
            for _ in range(10):
                try:
                    sequence, block = self.audio_q.get_nowait()
                except queue.Empty:
                    break
                if self.last_block is not None and sequence != self.last_block + 1:
                    self.segment.reset()
                self.last_block = sequence
                audio = self.segment.feed(block)
                if audio is not None:
                    self._accept_segment(audio)
                    if self.job is not None:
                        break
        else:
            self._reset_capture()
        self.root.after(40, self._poll)

    def _close(self):
        try:
            if self.stream is not None:
                self.stream.close()
        finally:
            self.worker.shutdown(wait=False, cancel_futures=True)
            self.csv.close()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="声纹注册与识别演示")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--clips", type=int, choices=range(2, 11), default=3)
    args = parser.parse_args()
    try:
        import sounddevice as sd

        engine = VoiceprintEngine(threshold=args.threshold)
        root = tk.Tk()
        VoiceprintDemo(root, engine, sd, default_clips=args.clips)
        root.mainloop()
    except Exception as exc:
        logger.error("声纹启动失败：%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
