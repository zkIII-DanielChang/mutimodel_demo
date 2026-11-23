"""声纹注册与识别统一演示界面（tkinter）。

一个窗口完成：注册、实时识别、已注册列表、阈值调整、清空库。
启动后默认处于识别模式，说话即识别。

用法：
    python scripts/voiceprint_demo.py
    python scripts/voiceprint_demo.py --threshold 0.35 --clips 3
"""

import argparse
import queue
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.common.config_loader import load_config, resolve_path
from src.common.csv_logger import CSVLogger
from src.common.logger import get_logger

logger = get_logger("voiceprint_demo")

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
SILENCE_THRESHOLD = 0.012
SILENCE_SECONDS = 0.8
MAX_CLIP_SECONDS = 6.0
POLL_INTERVAL_MS = 30
MAX_BLOCKS_PER_POLL = 10


class VoiceprintDemo:
    def __init__(self, root, engine, sd, default_clips=3):
        self.root = root
        self.engine = engine
        self.sd = sd
        self.default_clips = default_clips
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None
        self.running = True

        self.mode = "identifying"
        self.buffer: list = []
        self.in_speech = False
        self.silence_count = 0

        self.enroll_name = None
        self.enroll_target = default_clips
        self.enroll_clips: list = []

        paths_cfg = load_config("paths")
        log_path = resolve_path(paths_cfg.get("logs_dir", "data/logs")) / "voiceprint.csv"
        self.csv_logger = CSVLogger(log_path)

        self._build_ui()
        self._refresh_speakers()
        self._start_stream()
        self.root.after(POLL_INTERVAL_MS, self._poll)

    def _build_ui(self):
        self.root.title("声纹注册与识别演示")
        self.root.geometry("620x720")
        self.root.configure(bg="#f7f8fa")

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        self.status_var = tk.StringVar(value="状态：识别中")
        ttk.Label(
            top, textvariable=self.status_var,
            font=("Microsoft YaHei", 13, "bold"),
        ).pack(side="left")

        threshold_frame = ttk.Frame(top)
        threshold_frame.pack(side="right")
        ttk.Label(threshold_frame, text="阈值：").pack(side="left")
        self.threshold_var = tk.DoubleVar(value=self.engine.get_threshold())
        self.threshold_spin = ttk.Spinbox(
            threshold_frame, from_=0.10, to=0.90, increment=0.05,
            textvariable=self.threshold_var, width=5,
            command=self._apply_threshold,
        )
        self.threshold_spin.pack(side="left")
        self.threshold_spin.bind("<Return>", lambda e: self._apply_threshold())
        self.threshold_spin.bind("<FocusOut>", lambda e: self._apply_threshold())

        enroll = ttk.LabelFrame(self.root, text="注册新说话人", padding=10)
        enroll.pack(fill="x", padx=10, pady=5)

        row1 = ttk.Frame(enroll)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="姓名：").pack(side="left")
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(row1, textvariable=self.name_var, width=18)
        self.name_entry.pack(side="left", padx=5)
        ttk.Label(row1, text="采集段数：").pack(side="left", padx=(15, 0))
        self.clips_var = tk.IntVar(value=self.default_clips)
        ttk.Spinbox(
            row1, from_=1, to=10, textvariable=self.clips_var, width=5,
        ).pack(side="left", padx=5)
        self.enroll_btn = ttk.Button(row1, text="开始注册", command=self._start_enroll)
        self.enroll_btn.pack(side="left", padx=10)

        self.progress_var = tk.StringVar(value="输入姓名后点击“开始注册”")
        ttk.Label(
            enroll, textvariable=self.progress_var, foreground="#1a73e8",
        ).pack(anchor="w", pady=4)

        identify = ttk.LabelFrame(self.root, text="实时识别", padding=10)
        identify.pack(fill="x", padx=10, pady=5)

        self.identify_var = tk.StringVar(value="等待说话...")
        ttk.Label(
            identify, textvariable=self.identify_var,
            font=("Microsoft YaHei", 20, "bold"), foreground="#0f9d58",
        ).pack(anchor="w", pady=6)

        self.similarity_var = tk.StringVar(value="")
        ttk.Label(
            identify, textvariable=self.similarity_var,
            font=("Microsoft YaHei", 11), foreground="#555555",
        ).pack(anchor="w")

        list_frame = ttk.LabelFrame(self.root, text="已注册说话人", padding=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        list_inner = ttk.Frame(list_frame)
        list_inner.pack(fill="both", expand=True)

        self.speaker_list = tk.Listbox(
            list_inner, height=6, font=("Microsoft YaHei", 12),
        )
        self.speaker_list.pack(fill="both", expand=True, side="left")
        scrollbar = ttk.Scrollbar(
            list_inner, orient="vertical", command=self.speaker_list.yview,
        )
        scrollbar.pack(side="right", fill="y")
        self.speaker_list.configure(yscrollcommand=scrollbar.set)

        btn_row = ttk.Frame(list_frame)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="刷新列表", command=self._refresh_speakers).pack(
            side="left"
        )
        ttk.Button(btn_row, text="清空全部", command=self._clear_all).pack(
            side="left", padx=6
        )

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="退出", command=self._on_close).pack(side="right")
        self.hint_var = tk.StringVar(
            value="提示：注册时建议用不同语气各说一段（正常/快/慢），可提升跨语气识别鲁棒性。"
        )
        ttk.Label(
            bottom, textvariable=self.hint_var, foreground="#888888",
            wraplength=580, justify="left",
        ).pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_threshold(self):
        try:
            v = float(self.threshold_var.get())
        except (tk.TclError, ValueError):
            return
        v = max(0.10, min(0.90, v))
        self.engine.set_threshold(v)
        self.status_var.set(f"状态：识别中（阈值 {v:.2f}）")

    def _refresh_speakers(self):
        self.speaker_list.delete(0, tk.END)
        names = self.engine.list_speakers()
        if names:
            for n in names:
                self.speaker_list.insert(tk.END, n)
        else:
            self.speaker_list.insert(tk.END, "（尚未注册）")

    def _clear_all(self):
        if not messagebox.askyesno("确认", "确定清空所有已注册说话人？"):
            return
        self.engine.clear()
        self._refresh_speakers()
        self.identify_var.set("等待说话...")
        self.similarity_var.set("")
        self.progress_var.set("说话人库已清空")
        self.status_var.set("状态：识别中")

    def _start_stream(self):
        def callback(indata, frames, time_info, status):
            self.audio_q.put(indata[:, 0].copy())

        self.stream = self.sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE,
            dtype="float32", callback=callback,
        )
        self.stream.start()

    def _clear_queue(self):
        while True:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def _start_enroll(self):
        if self.mode == "enrolling":
            messagebox.showinfo("提示", "正在注册中，请等待完成")
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("提示", "请输入姓名")
            return

        self.enroll_name = name
        self.enroll_target = max(1, int(self.clips_var.get()))
        self.enroll_clips = []
        self.buffer = []
        self.in_speech = False
        self.silence_count = 0

        self.mode = "enrolling"
        self.status_var.set(f"状态：正在注册 {name}（0/{self.enroll_target}）")
        self.progress_var.set(
            f"请说 {self.enroll_target} 段话，每段 1–3 秒；建议用不同语气各说一段。"
        )
        self.enroll_btn.config(state="disabled")
        self._clear_queue()

    def _back_to_identify(self):
        self.mode = "identifying"
        self.buffer = []
        self.in_speech = False
        self.silence_count = 0
        self.status_var.set(f"状态：识别中（阈值 {self.engine.get_threshold():.2f}）")
        self.enroll_btn.config(state="normal")
        self._clear_queue()

    def _poll(self):
        if not self.running:
            return

        silence_needed = int(SILENCE_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
        max_blocks = int(MAX_CLIP_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

        for _ in range(MAX_BLOCKS_PER_POLL):
            try:
                block = self.audio_q.get_nowait()
            except queue.Empty:
                break

            rms = float(np.sqrt(np.mean(block ** 2)))
            if rms > SILENCE_THRESHOLD:
                self.in_speech = True
                self.silence_count = 0
                self.buffer.append(block)
            elif self.in_speech:
                self.silence_count += 1
                self.buffer.append(block)
                if (
                    self.silence_count >= silence_needed
                    or len(self.buffer) >= max_blocks
                ):
                    audio = np.concatenate(self.buffer).astype(np.float32)
                    self.buffer = []
                    self.in_speech = False
                    self.silence_count = 0
                    self._handle_utterance(audio)

        self.root.after(POLL_INTERVAL_MS, self._poll)

    def _handle_utterance(self, audio: np.ndarray):
        if self.mode == "identifying":
            result = self.engine.identify_audio(audio, SAMPLE_RATE)
            if result is None:
                self.identify_var.set("未获得声纹结果")
                self.similarity_var.set("")
                return
            if result["accepted"]:
                self.identify_var.set(f"说话人：{result['name']}")
                self.similarity_var.set(
                    f"相似度 {result['score']:.3f} · matched"
                )
            else:
                self.identify_var.set("说话人：Unknown")
                self.similarity_var.set(
                    f"相似度 {result['score']:.3f} · unknown"
                )
            self.csv_logger.log(
                "voiceprint", "speaker", result["name"],
                "matched" if result["accepted"] else "unknown",
            )
            self.csv_logger.log(
                "voiceprint", "similarity", f"{result['score']:.4f}", "",
            )

        elif self.mode == "enrolling":
            self.enroll_clips.append(audio)
            n = len(self.enroll_clips)
            self.status_var.set(
                f"状态：正在注册 {self.enroll_name}（{n}/{self.enroll_target}）"
            )
            self.progress_var.set(
                f"已采集 {n}/{self.enroll_target} 段，时长 {len(audio) / SAMPLE_RATE:.2f}s"
            )
            if n >= self.enroll_target:
                self._finish_enroll()

    def _finish_enroll(self):
        self.progress_var.set("正在生成声纹，请稍候...")
        self.root.update_idletasks()

        name = self.enroll_name
        clips = self.enroll_clips
        try:
            n = self.engine.enroll_from_audio(name, clips, SAMPLE_RATE)
        except Exception as e:
            n = 0
            logger.error("注册异常: %s", e)

        if n > 0:
            self._refresh_speakers()
            self.progress_var.set(f"注册完成：{name}，使用 {n} 段音频")
            messagebox.showinfo("注册成功", f"{name} 注册完成，使用 {n} 段音频")
        else:
            self.progress_var.set("注册失败，请重试")
            messagebox.showerror("注册失败", "未生成有效声纹，请检查麦克风或重试")

        self.enroll_name = None
        self.enroll_clips = []
        self.name_var.set("")
        self._back_to_identify()

    def _on_close(self):
        self.running = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        try:
            self.csv_logger.close()
        except Exception:
            pass
        self.root.destroy()


def parse_args():
    p = argparse.ArgumentParser(description="声纹注册与识别统一演示")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--clips", type=int, default=3, help="默认注册采集段数")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print("未安装 sounddevice，请执行 pip install sounddevice")
        return 1

    try:
        from src.voiceprint.engine import VoiceprintEngine

        engine = VoiceprintEngine(threshold=args.threshold)
    except Exception as e:
        print(f"声纹引擎初始化失败: {e}")
        return 1

    root = tk.Tk()
    VoiceprintDemo(root, engine, sd, default_clips=args.clips)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())