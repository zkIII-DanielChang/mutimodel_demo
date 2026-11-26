"""车载多模态演示控制中心。"""

import os
import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config, resolve_path
from src.console import theme
from src.console.csv_tailer import estimate_fps_from_csv, latest_metrics
from src.console.process_manager import ProcessManager

CARD_GROUPS = (
    ("驾驶监测", ("dms", "head_pose", "gaze", "fatigue", "emotion", "gesture")),
    ("语音与身份", ("asr", "voiceprint", "face_recognize", "fingerprint")),
)
METRICS = {
    "dms": {"state": "驾驶状态", "face_detected": "人脸", "quality_ok": "画质",
            "ear": "EAR", "mar": "MAR", "head_yaw": "Yaw", "head_pitch": "Pitch"},
    "face_recognize": {"identity": "身份", "similarity": "相似度"},
    "head_pose": {"head_yaw": "Yaw", "head_pitch": "Pitch", "head_roll": "Roll"},
    "gaze": {"gaze_yaw": "视线 Y", "gaze_pitch": "视线 P"},
    "gesture": {"label": "手势"},
    "fatigue": {"state": "疲劳", "ear": "EAR", "mar": "MAR"},
    "emotion": {"emotion": "表情"},
    "asr": {"text": "识别文本", "command": "匹配指令"},
    "voiceprint": {"speaker": "说话人", "similarity": "相似度"},
    "fingerprint": {"decision": "判定", "good_matches": "匹配数"},
}
DMS_STATES = {
    "normal": ("正常", theme.ACCENT_GREEN),
    "distracted": ("分心提醒", theme.ACCENT_YELLOW),
    "drowsy": ("疲劳提醒", theme.ACCENT_YELLOW),
    "severe_drowsy": ("严重疲劳", theme.ACCENT_RED),
    "unavailable": ("监控不可用", theme.ACCENT_RED),
}


class DemoConsole:
    def __init__(self, root):
        self.root = root
        root.title("车载多模态 · 演示控制中心")
        root.configure(bg=theme.BG_ROOT)
        root.geometry(f"{min(1180, root.winfo_screenwidth() - 40)}x"
                      f"{min(680, root.winfo_screenheight() - 95)}")
        root.minsize(840, 520)
        cfg = load_config("demo_scenes")
        self.scenes = cfg["scenes"]
        self.config = cfg["modalities"]
        self.logs_path = resolve_path(load_config("paths").get("logs_dir", "data/logs"))
        self.manager = ProcessManager(self.config, PROJECT_ROOT, self.logs_path)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.job = None
        self.stop_requested = False
        self.closing = False
        self.scene_index = 0
        self.focus_key = None
        self.cards = {}
        self.statuses = {}
        self._build()
        self._poll()

    def _build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Console.TButton", padding=(8, 6), relief="flat", borderwidth=1,
                        background=theme.BG_CARD, foreground=theme.FG_PRIMARY,
                        bordercolor=theme.BG_CARD_ACTIVE)
        style.map("Console.TButton", background=[("selected", theme.BG_CARD_ACTIVE),
                                                  ("active", theme.BG_CARD_HOVER)],
                  bordercolor=[("selected", theme.ACCENT_BLUE), ("focus", theme.ACCENT_BLUE)])
        header = tk.Frame(self.root, bg=theme.BG_PANEL, padx=18, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="车载多模态  /  演示控制中心", bg=theme.BG_PANEL,
                 fg=theme.FG_PRIMARY, font=theme.FONT_TITLE).pack(side="left")
        for text, command in (("停止全部", self._stop_all), ("打开日志", self._open_logs)):
            ttk.Button(header, text=text, command=command,
                       style="Console.TButton").pack(side="right", padx=(6, 0))

        navigation = tk.Frame(self.root, bg=theme.BG_ROOT, padx=14, pady=8)
        navigation.pack(fill="x")
        self.scene_buttons = []
        for i, title in enumerate(["手动控制"] + [f"{s['id']}  {s['name']}" for s in self.scenes]):
            button = ttk.Button(navigation, text=title, style="Console.TButton", width=0,
                                command=lambda index=i: self._select_scene(index))
            button.pack(side="left", padx=3)
            self.scene_buttons.append(button)
        self.overview = tk.StringVar()
        tk.Label(navigation, textvariable=self.overview, bg=theme.BG_ROOT,
                 fg=theme.FG_PRIMARY, font=theme.FONT_HINT).pack(side="right", padx=4)

        footer = tk.Label(self.root, text="1–5 切换场景     空格 下一幕     Esc 停止全部     L 打开日志",
                          anchor="w", padx=16, pady=7, bg=theme.BG_PANEL,
                          fg=theme.FG_SECONDARY, font=theme.FONT_HINT)
        footer.pack(side="bottom", fill="x")
        body = tk.Frame(self.root, bg=theme.BG_ROOT)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        body.columnconfigure(0, weight=5, minsize=450)
        body.columnconfigure(1, weight=4, minsize=330)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=theme.BG_PANEL, padx=12, pady=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = tk.Frame(body, bg=theme.BG_PANEL, padx=13, pady=10)
        right.grid(row=0, column=1, sticky="nsew")

        self.scene_title = self._label(left, "", theme.FONT_TITLE)
        self.scene_title.pack(fill="x")
        self.narrator = self._label(left, "", theme.FONT_HINT, fg=theme.FG_SECONDARY,
                                    justify="left", wraplength=420)
        self.narrator.pack(fill="x", pady=(4, 6))
        left.bind("<Configure>", lambda event: self.narrator.configure(wraplength=event.width - 24))
        for heading, keys in CARD_GROUPS:
            self._label(left, heading, theme.FONT_SUBTITLE).pack(fill="x", pady=(5, 2))
            grid = tk.Frame(left, bg=theme.BG_PANEL)
            grid.pack(fill="x")
            for i, key in enumerate(keys):
                grid.columnconfigure(i % 3, weight=1, uniform="card")
                button = tk.Button(grid, anchor="w", justify="left", padx=8, pady=5,
                                   font=theme.FONT_CARD_STATUS, relief="flat", bd=0,
                                   activebackground=theme.BG_CARD_HOVER, activeforeground=theme.FG_PRIMARY,
                                   command=lambda k=key: self._toggle(k))
                button.grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=2)
                self.cards[key] = button

        tools = tk.Frame(right, bg=theme.BG_PANEL)
        tools.pack(fill="x")
        self._label(tools, "实时指标", theme.FONT_TITLE).pack(side="left")
        self.metric_choice = ttk.Combobox(tools, state="readonly", width=17)
        self.metric_choice.pack(side="right")
        self.metric_choice.bind("<<ComboboxSelected>>", self._choose_metrics)
        self.metric_keys = []
        self.headline = self._label(right, "", theme.FONT_TITLE)
        self.headline.pack(fill="x", pady=(10, 6))
        table = tk.Frame(right, bg=theme.BG_PANEL)
        table.pack(fill="x")
        table.columnconfigure((0, 1), weight=1, uniform="metric")
        self.metric_cells = []
        for i in range(6):
            cell = self._label(table, "", theme.FONT_METRIC_LABEL, bg=theme.BG_CARD, padx=8, pady=4)
            cell.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            self.metric_cells.append(cell)
        self._label(right, "操作与告警", theme.FONT_SUBTITLE).pack(fill="x", pady=(10, 4))
        self.log = tk.Text(right, height=3, wrap="word", bg=theme.BG_CARD, fg=theme.FG_PRIMARY,
                           font=theme.FONT_HINT, relief="flat", padx=8, pady=6, state="disabled")
        self.log.pack(fill="both", expand=True)
        self.last_dms = None
        self._show_scene(0)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda _: self._stop_all())
        self.root.bind("<KeyPress-l>", lambda _: self._open_logs())
        self.root.bind("<space>", lambda event: self._next_scene(event))
        for index, scene in enumerate(self.scenes, 1):
            self.root.bind(str(scene["id"]), lambda _, i=index: self._select_scene(i))

    @staticmethod
    def _label(parent, text, font, **kwargs):
        return tk.Label(parent, text=text, font=font, anchor="w",
                        bg=kwargs.pop("bg", theme.BG_PANEL),
                        fg=kwargs.pop("fg", theme.FG_PRIMARY), **kwargs)

    def _show_scene(self, index):
        self.scene_index = index
        for i, button in enumerate(self.scene_buttons):
            button.state(["selected" if i == index else "!selected"])
        if index:
            scene = self.scenes[index - 1]
            self.scene_title.configure(text=f"{scene['id']}  {scene['name']}  ·  {scene['duration']}")
            self.narrator.configure(text=scene["narrator"])
        else:
            self.scene_title.configure(text="手动控制")
            self.narrator.configure(text="点击卡片启停模态。视频在独立窗口显示，切换时会先释放摄像头。")

    def _select_scene(self, index):
        if self.job is not None or self.closing:
            return
        self._show_scene(index)
        if not index:
            return
        scene = self.scenes[index - 1]
        target = [key for key in (scene.get("video_modality"), *scene["parallel_modalities"]) if key]
        self.focus_key = target[0] if target else None
        self._log(f"切换场景：{scene['name']}")
        self.job = self.worker.submit(self.manager.activate, target)

    def _next_scene(self, event):
        if event.widget is not self.metric_choice and not isinstance(event.widget, (tk.Button, ttk.Button)):
            self._select_scene(self.scene_index % len(self.scenes) + 1)

    def _toggle(self, key):
        if self.job is not None or self.closing:
            return
        running = self.statuses.get(key) == "running"
        self.focus_key = key
        self._log(f"{'停止' if running else '启动'} {self.config[key]['label']}")
        self.job = self.worker.submit(self.manager.stop if running else self.manager.start, key)

    def _choose_metrics(self, _):
        index = self.metric_choice.current()
        if 0 <= index < len(self.metric_keys):
            self.focus_key = self.metric_keys[index]

    def _poll(self):
        if self.job is not None and self.job.done():
            try:
                if self.job.result() is False:
                    self._log("部分操作未完成，请查看模态状态和日志。")
            except Exception as exc:
                self._log(f"操作失败：{exc}")
            self.job = None
        if self.job is None and self.stop_requested:
            self.stop_requested = False
            self.job = self.worker.submit(self.manager.stop_all)
        if self.job is None:
            if self.closing and not self.manager.running_modalities():
                self.worker.shutdown(wait=False)
                self.root.destroy()
                return
            if self.closing:
                self.closing = False
            self._refresh()
        else:
            self.overview.set("正在切换模态…")
        self.root.after(200, self._poll)

    def _refresh(self):
        statuses = self.manager.get_all_status()
        running = [key for key in self.cards if statuses[key] == "running"]
        self.overview.set(f"运行 {len(running)}  ·  异常 {sum(s == 'error' for s in statuses.values())}")
        for key, button in self.cards.items():
            status = statuses[key]
            if self.statuses.get(key) == status:
                continue
            color = theme.STATUS_COLORS[status]
            action = "点击停止" if status == "running" else "点击启动"
            button.configure(text=f"{self.config[key]['label']}\n● {theme.STATUS_LABELS[status]} · {action}",
                             fg=color if status == "error" else theme.FG_PRIMARY,
                             bg=theme.BG_CARD_ACTIVE if status == "running" else theme.BG_CARD)
            if status == "error":
                self._log(f"{self.config[key]['label']}异常，详见 {key}.log")
        self.statuses = statuses
        if self.metric_keys != running:
            self.metric_keys = running
            self.metric_choice.configure(values=[self.config[k]["label"] for k in running])
        active = self.focus_key if self.focus_key in running else next(iter(running), None)
        if active:
            self.metric_choice.current(running.index(active))
        else:
            self.metric_choice.set("")

        dms = self._metrics("dms") if "dms" in running else {}
        dms_state = dms.get("state")
        if dms_state in DMS_STATES and dms_state != self.last_dms:
            self._log(f"DMS：{DMS_STATES[dms_state][0]}")
        self.last_dms = dms_state
        self._render_metrics(active, dms if active == "dms" else self._metrics(active))

    def _metrics(self, key):
        if not key:
            return {}
        modality = self.manager.modalities[key]
        return latest_metrics(self.logs_path / modality.csv, list(METRICS[key]),
                              since=modality.started_at, max_age=2 if modality.type == "video" else None)

    def _render_metrics(self, key, values):
        headline = "暂无运行中的模态"
        color = theme.FG_SECONDARY
        detail = []
        if key == "dms":
            headline, color = DMS_STATES.get(values.get("state"), ("等待 DMS 数据", theme.FG_SECONDARY))
            if values.get("quality_ok") != "1":
                values = {k: v for k, v in values.items() if k in ("state", "face_detected", "quality_ok")}
            modality = self.manager.modalities[key]
            fps = estimate_fps_from_csv(self.logs_path / modality.csv, since=modality.started_at, metric="state")
            if fps is not None:
                headline += f"  ·  {fps:.0f} FPS"
        elif key:
            headline = self.config[key]["label"] if values else "等待新数据"
        for field, label in METRICS.get(key, {}).items():
            if key == "dms" and field == "state":
                continue
            value = values.get(field) or "—"
            if field in ("face_detected", "quality_ok"):
                options = ("未检测", "已检测") if field == "face_detected" else ("不良", "正常")
                value = {"0": options[0], "1": options[1]}.get(value, value)
            detail.append(f"{label}  {value[:28]}")
        self.headline.configure(text=headline, fg=color)
        for i, cell in enumerate(self.metric_cells):
            cell.configure(text=detail[i] if i < len(detail) else "")

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", f"{datetime.now():%H:%M:%S}  {text}\n")
        if int(self.log.index("end-1c").split(".")[0]) > 100:
            self.log.delete("1.0", "2.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _stop_all(self):
        self._log("停止全部模态")
        self.stop_requested = True

    def _open_logs(self):
        self.logs_path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.logs_path))
        except (OSError, AttributeError) as exc:
            self._log(f"无法打开日志目录：{exc}")

    def _on_close(self):
        if self.closing:
            return
        if (self.job is not None or "running" in self.statuses.values()) and not messagebox.askyesno(
            "确认退出", "停止所有模态并退出？", parent=self.root
        ):
            return
        self.closing = True
        self.stop_requested = True


def main():
    root = tk.Tk()
    DemoConsole(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
