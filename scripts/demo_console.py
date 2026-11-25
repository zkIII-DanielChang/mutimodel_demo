"""九模态车载总控界面。

深色主题，九张模态卡片，点击启停子进程，右栏实时读 CSV 显示指标。
视频模态保持 OpenCV 独立窗口，不嵌入 tkinter，避免帧率下降。

本版重点修复：右栏指标不再销毁重建，改为预创建 + 只更新 text，
消除每 500ms 的页面闪动。
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.common.config_loader import load_config, resolve_path
from src.common.logger import get_logger
from src.console import theme
from src.console.csv_tailer import estimate_fps_from_csv, latest_metrics
from src.console.process_manager import ProcessManager

logger = get_logger("demo_console")

CARD_LAYOUT = [
    ["face_recognize", "head_pose", "gaze"],
    ["gesture", "fatigue", "emotion"],
    ["asr", "voiceprint", "fingerprint"],
]

CARD_ICONS = {
    "face_recognize": "👤",
    "head_pose": "🧭",
    "gaze": "👁",
    "gesture": "✋",
    "fatigue": "😴",
    "emotion": "🙂",
    "asr": "🎤",
    "voiceprint": "🔊",
    "fingerprint": "🖐",
}

METRIC_MAP = {
    "face_recognize": ["identity", "similarity"],
    "head_pose": ["head_yaw", "head_pitch", "head_roll"],
    "gaze": ["gaze_yaw", "gaze_pitch"],
    "gesture": ["label"],
    "fatigue": ["ear", "mar", "state"],
    "emotion": ["emotion"],
    "asr": ["text", "command"],
    "voiceprint": ["speaker", "similarity"],
    "fingerprint": ["decision", "good_matches"],
}

METRIC_LABELS = {
    "identity": "身份",
    "similarity": "相似度",
    "head_yaw": "Yaw",
    "head_pitch": "Pitch",
    "head_roll": "Roll",
    "gaze_yaw": "视线 Y",
    "gaze_pitch": "视线 P",
    "label": "手势",
    "ear": "EAR",
    "mar": "MAR",
    "state": "疲劳",
    "emotion": "表情",
    "text": "识别文本",
    "command": "匹配指令",
    "speaker": "说话人",
    "decision": "判定",
    "good_matches": "匹配数",
}

# 预创建所有可能的 metric 控件
ALL_METRICS = set()
for _m in METRIC_MAP.values():
    ALL_METRICS.update(_m)

POLL_INTERVAL_MS = 500


class ModalityCard:
    def __init__(self, parent, key: str, label: str, on_click):
        self.key = key
        self.label = label
        self.on_click = on_click

        self.frame = tk.Frame(
            parent, bg=theme.BG_CARD, highlightthickness=1,
            highlightbackground="#2a3038", cursor="hand2",
        )
        self.frame.pack_propagate(False)

        inner = tk.Frame(self.frame, bg=theme.BG_CARD)
        inner.pack(fill="both", expand=True, padx=10, pady=8)

        top = tk.Frame(inner, bg=theme.BG_CARD)
        top.pack(fill="x")
        self.icon_label = tk.Label(
            top, text=CARD_ICONS.get(key, "◆"),
            font=("Segoe UI Emoji", 16),
            bg=theme.BG_CARD, fg=theme.FG_PRIMARY,
        )
        self.icon_label.pack(side="left")
        self.dot = tk.Label(
            top, text="●", font=("Arial", 12),
            bg=theme.BG_CARD, fg=theme.STATUS_COLORS["idle"],
        )
        self.dot.pack(side="right")

        self.title_label = tk.Label(
            inner, text=label, font=theme.FONT_CARD_TITLE,
            bg=theme.BG_CARD, fg=theme.FG_PRIMARY, anchor="w",
        )
        self.title_label.pack(fill="x", pady=(4, 0))

        self.status_label = tk.Label(
            inner, text="待机", font=theme.FONT_CARD_STATUS,
            bg=theme.BG_CARD, fg=theme.FG_SECONDARY, anchor="w",
        )
        self.status_label.pack(fill="x")

        self.fps_label = tk.Label(
            inner, text="", font=theme.FONT_CARD_STATUS,
            bg=theme.BG_CARD, fg=theme.ACCENT_BLUE, anchor="w",
        )
        self.fps_label.pack(fill="x")

        for w in [self.frame, inner, top, self.icon_label, self.title_label,
                  self.status_label, self.fps_label, self.dot]:
            w.bind("<Button-1>", lambda e: self.on_click(self.key))
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)

        self._last_status = "idle"
        self._last_fps_text = ""

    def _on_enter(self, _):
        self.frame.configure(bg=theme.BG_CARD_HOVER, highlightbackground=theme.ACCENT_BLUE)
        for w in [self.icon_label, self.title_label, self.status_label, self.fps_label, self.dot]:
            try:
                w.configure(bg=theme.BG_CARD_HOVER)
            except Exception:
                pass

    def _on_leave(self, _):
        self.frame.configure(bg=theme.BG_CARD, highlightbackground="#2a3038")
        for w in [self.icon_label, self.title_label, self.status_label, self.fps_label, self.dot]:
            try:
                w.configure(bg=theme.BG_CARD)
            except Exception:
                pass

    def update_status(self, status: str, fps: float = None):
        # 状态未变时跳过颜色更新
        if status != self._last_status:
            color = theme.STATUS_COLORS.get(status, theme.ACCENT_GRAY)
            label = theme.STATUS_LABELS.get(status, status)
            self.dot.configure(fg=color)
            self.status_label.configure(text=label, fg=color)
            self._last_status = status

        # FPS 文本未变时跳过
        new_fps_text = f"{fps:.1f} FPS" if (fps is not None and status == "running") else ""
        if new_fps_text != self._last_fps_text:
            self.fps_label.configure(text=new_fps_text)
            self._last_fps_text = new_fps_text


class DemoConsole:
    def __init__(self, root):
        self.root = root
        self.root.title("车载多模态驾驶辅助系统 · 总控台")
        self.root.geometry("1280x800")
        self.root.configure(bg=theme.BG_ROOT)
        self.root.minsize(1100, 700)

        config_path = resolve_path("configs/demo_scenes.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.modalities_cfg = cfg.get("modalities", {})
        self.scenes = cfg.get("scenes", [])

        self.logs_path = resolve_path(load_config("paths").get("logs_dir", "data/logs"))
        self.manager = ProcessManager(self.modalities_cfg, PROJECT_ROOT, self.logs_path)
        self.cards = {}

        # 用于避免重复刷新
        self._last_running_text = ""
        self._last_active_key = None
        self._last_metric_values: dict = {}

        self._build_ui()
        self._poll()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = tk.Frame(self.root, bg=theme.BG_PANEL, height=56)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        tk.Label(
            top, text="车载多模态驾驶辅助系统",
            font=theme.FONT_TITLE, bg=theme.BG_PANEL, fg=theme.FG_PRIMARY,
        ).pack(side="left", padx=20)

        tk.Label(
            top, text="总控台",
            font=theme.FONT_SUBTITLE, bg=theme.BG_PANEL, fg=theme.FG_SECONDARY,
        ).pack(side="left", padx=(6, 0))

        scene_frame = tk.Frame(top, bg=theme.BG_PANEL)
        scene_frame.pack(side="right", padx=20)

        tk.Label(
            scene_frame, text="场景：", font=theme.FONT_SUBTITLE,
            bg=theme.BG_PANEL, fg=theme.FG_SECONDARY,
        ).pack(side="left")

        self.scene_var = tk.StringVar(value="手动控制")
        scene_names = ["手动控制"] + [f"{s['id']}. {s['name']}" for s in self.scenes]
        self.scene_combo = ttk.Combobox(
            scene_frame, textvariable=self.scene_var, values=scene_names,
            state="readonly", width=14,
        )
        self.scene_combo.pack(side="left", padx=6)
        self.scene_combo.bind("<<ComboboxSelected>>", self._on_scene_change)

        tk.Button(
            top, text="退出", command=self._on_close,
            bg=theme.BG_CARD, fg=theme.FG_PRIMARY, bd=0,
            font=theme.FONT_SUBTITLE, padx=16, pady=6, cursor="hand2",
        ).pack(side="right", padx=(0, 20))

        body = tk.Frame(self.root, bg=theme.BG_ROOT)
        body.pack(fill="both", expand=True, padx=12, pady=12)

        # 左栏：九卡片
        left = tk.Frame(body, bg=theme.BG_ROOT, width=440)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(
            left, text="模态控制", font=theme.FONT_SUBTITLE,
            bg=theme.BG_ROOT, fg=theme.FG_SECONDARY, anchor="w",
        ).pack(fill="x", pady=(0, 8))

        grid = tk.Frame(left, bg=theme.BG_ROOT)
        grid.pack(fill="both", expand=True)

        for r, row in enumerate(CARD_LAYOUT):
            for c, key in enumerate(row):
                cfg = self.modalities_cfg.get(key, {})
                label = cfg.get("label", key)
                card = ModalityCard(grid, key, label, self._on_card_click)
                card.frame.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                self.cards[key] = card

        for i in range(3):
            grid.rowconfigure(i, weight=1)
            grid.columnconfigure(i, weight=1)

        # 中栏
        center = tk.Frame(body, bg=theme.BG_PANEL)
        center.pack(side="left", fill="both", expand=True, padx=(0, 12))

        tk.Label(
            center, text="当前场景", font=theme.FONT_SUBTITLE,
            bg=theme.BG_PANEL, fg=theme.FG_SECONDARY, anchor="w",
        ).pack(fill="x", padx=20, pady=(16, 4))

        self.scene_title = tk.Label(
            center, text="手动控制", font=("Microsoft YaHei", 22, "bold"),
            bg=theme.BG_PANEL, fg=theme.FG_PRIMARY, anchor="w",
        )
        self.scene_title.pack(fill="x", padx=20)

        self.scene_narrator = tk.Label(
            center, text="点击左侧卡片启动模态，视频窗口在独立窗口中打开。",
            font=theme.FONT_SUBTITLE, bg=theme.BG_PANEL, fg=theme.FG_SECONDARY,
            anchor="w", wraplength=460, justify="left",
        )
        self.scene_narrator.pack(fill="x", padx=20, pady=(8, 16))

        tk.Label(
            center, text="运行中的模态", font=theme.FONT_SUBTITLE,
            bg=theme.BG_PANEL, fg=theme.FG_SECONDARY, anchor="w",
        ).pack(fill="x", padx=20)

        self.running_list = tk.Text(
            center, height=8, bg=theme.BG_CARD, fg=theme.FG_PRIMARY,
            font=theme.FONT_LOG, bd=0, padx=10, pady=8,
            insertbackground=theme.FG_PRIMARY, state="disabled",
        )
        self.running_list.pack(fill="x", padx=20, pady=(6, 16))

        # 右栏：指标 + 日志
        right = tk.Frame(body, bg=theme.BG_PANEL, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(
            right, text="实时指标", font=theme.FONT_SUBTITLE,
            bg=theme.BG_PANEL, fg=theme.FG_SECONDARY, anchor="w",
        ).pack(fill="x", padx=16, pady=(16, 8))

        self.metrics_frame = tk.Frame(right, bg=theme.BG_PANEL)
        self.metrics_frame.pack(fill="x", padx=16)

        # 预创建所有指标行，初始隐藏
        self.metric_widgets = {}
        for key in sorted(ALL_METRICS):
            row = tk.Frame(self.metrics_frame, bg=theme.BG_PANEL)
            label = tk.Label(
                row, text=METRIC_LABELS.get(key, key), width=10,
                font=theme.FONT_METRIC_LABEL, bg=theme.BG_PANEL,
                fg=theme.FG_SECONDARY, anchor="w",
            )
            label.pack(side="left")
            value = tk.Label(
                row, text="", font=theme.FONT_METRIC_VALUE,
                bg=theme.BG_PANEL, fg=theme.ACCENT_BLUE, anchor="w",
            )
            value.pack(side="left", padx=6)
            self.metric_widgets[key] = {
                "row": row, "value": value, "visible": False,
            }

        self.metric_placeholder = tk.Label(
            self.metrics_frame, text="（无运行中的模态）",
            font=theme.FONT_METRIC_LABEL, bg=theme.BG_PANEL, fg=theme.FG_MUTED,
        )
        self.metric_placeholder.pack(anchor="w")

        tk.Label(
            right, text="日志", font=theme.FONT_SUBTITLE,
            bg=theme.BG_PANEL, fg=theme.FG_SECONDARY, anchor="w",
        ).pack(fill="x", padx=16, pady=(16, 4))

        self.log_text = tk.Text(
            right, height=14, bg=theme.BG_CARD, fg=theme.FG_SECONDARY,
            font=theme.FONT_LOG, bd=0, padx=10, pady=8, state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._log_lines = []

        # 底栏
        bottom = tk.Frame(self.root, bg=theme.BG_PANEL, height=36)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        self.device_var = tk.StringVar(value="设备状态：摄像头 · 麦克风 · 模型缓存")
        tk.Label(
            bottom, textvariable=self.device_var, font=theme.FONT_HINT,
            bg=theme.BG_PANEL, fg=theme.FG_SECONDARY,
        ).pack(side="left", padx=16)

        tk.Label(
            bottom, text="空格：下一幕  1-5：跳幕  Esc：停止全部  L：打开日志",
            font=theme.FONT_HINT, bg=theme.BG_PANEL, fg=theme.FG_MUTED,
        ).pack(side="right", padx=16)

        self.root.bind("<space>", lambda e: self._next_scene())
        self.root.bind("<Escape>", lambda e: self._stop_all())
        self.root.bind("<Key-l>", lambda e: self._open_logs())
        for i in range(1, 6):
            self.root.bind(str(i), lambda e, idx=i: self._jump_scene(idx))

    # ---------------- 卡片点击 ----------------
    def _on_card_click(self, key: str):
        status = self.manager.get_status(key)
        if status in ("running", "starting"):
            self._log(f"停止 {self.modalities_cfg[key]['label']}")
            self.manager.stop(key)
        else:
            self._log(f"启动 {self.modalities_cfg[key]['label']}")
            self.manager.start(key)

    # ---------------- 场景 ----------------
    def _on_scene_change(self, _):
        val = self.scene_var.get()
        if val == "手动控制":
            self.scene_title.configure(text="手动控制")
            self.scene_narrator.configure(
                text="点击左侧卡片启动模态，视频窗口在独立窗口中打开。"
            )
            return
        try:
            idx = int(val.split(".")[0])
        except Exception:
            return
        self._activate_scene(idx)

    def _activate_scene(self, scene_id: int):
        scene = next((s for s in self.scenes if s["id"] == scene_id), None)
        if scene is None:
            return
        self._log(f"切换场景：{scene['name']}")
        self.scene_title.configure(
            text=f"{scene['id']}. {scene['name']}  ·  {scene['duration']}"
        )
        self.scene_narrator.configure(text=scene.get("narrator", ""))

        self.manager.stop_all_video()

        vid = scene.get("video_modality")
        if vid:
            self.manager.start(vid)

        for mod in scene.get("parallel_modalities", []):
            self.manager.start(mod)

    def _next_scene(self):
        cur = self.scene_var.get()
        try:
            cur_id = int(cur.split(".")[0]) if cur != "手动控制" else 0
        except Exception:
            cur_id = 0
        nxt = (cur_id % len(self.scenes)) + 1
        scene = next((s for s in self.scenes if s["id"] == nxt), None)
        if scene:
            self.scene_var.set(f"{scene['id']}. {scene['name']}")
            self._activate_scene(nxt)

    def _jump_scene(self, idx: int):
        scene = next((s for s in self.scenes if s["id"] == idx), None)
        if scene:
            self.scene_var.set(f"{scene['id']}. {scene['name']}")
            self._activate_scene(idx)

    # ---------------- 轮询 ----------------
    def _poll(self):
        # 更新卡片状态
        for key, card in self.cards.items():
            status = self.manager.get_status(key)
            csv_name = self.modalities_cfg[key].get("csv", "")
            fps = None
            if status == "running" and csv_name:
                csv_path = self.logs_path / csv_name
                fps = estimate_fps_from_csv(csv_path)
            card.update_status(status, fps)

        self._update_metrics()
        self._update_running_list()

        self.root.after(POLL_INTERVAL_MS, self._poll)

    def _update_running_list(self):
        running = self.manager.running_modalities()
        if running:
            new_text = "".join(
                f"● {self.modalities_cfg[key]['label']}\n" for key in running
            )
        else:
            new_text = "（无运行中的模态）\n"
        if new_text == self._last_running_text:
            return
        self.running_list.configure(state="normal")
        self.running_list.delete("1.0", "end")
        self.running_list.insert("end", new_text)
        self.running_list.configure(state="disabled")
        self._last_running_text = new_text

    def _update_metrics(self):
        running = self.manager.running_modalities()
        active_key = None
        for key in ["face_recognize", "head_pose", "gaze", "gesture", "fatigue",
                    "emotion", "asr", "voiceprint", "fingerprint"]:
            if key in running:
                active_key = key
                break

        # 无运行模态：隐藏所有指标，显示占位
        if not active_key:
            for w in self.metric_widgets.values():
                if w["visible"]:
                    w["row"].pack_forget()
                    w["visible"] = False
            if not self.metric_placeholder.winfo_ismapped():
                self.metric_placeholder.pack(anchor="w")
            self._last_active_key = None
            self._last_metric_values = {}
            return

        # 有运行模态：隐藏占位
        if self.metric_placeholder.winfo_ismapped():
            self.metric_placeholder.pack_forget()

        csv_name = self.modalities_cfg[active_key].get("csv", "")
        if not csv_name:
            return
        csv_path = self.logs_path / csv_name
        wanted = METRIC_MAP.get(active_key, [])
        metrics = latest_metrics(csv_path, wanted)

        # 切换模态时，隐藏不在新 wanted 中的
        if active_key != self._last_active_key:
            for key, w in self.metric_widgets.items():
                if key not in wanted and w["visible"]:
                    w["row"].pack_forget()
                    w["visible"] = False
            self._last_metric_values = {}
            self._last_active_key = active_key

        # 按 wanted 顺序显示/更新
        for key in wanted:
            if key not in self.metric_widgets:
                continue
            w = self.metric_widgets[key]
            if key in metrics:
                new_val = metrics[key]
                if self._last_metric_values.get(key) != new_val:
                    w["value"].configure(text=new_val)
                    self._last_metric_values[key] = new_val
                if not w["visible"]:
                    w["row"].pack(fill="x", pady=2)
                    w["visible"] = True
            elif w["visible"]:
                w["row"].pack_forget()
                w["visible"] = False

    # ---------------- 辅助 ----------------
    def _log(self, text: str):
        self._log_lines.append(text)
        if len(self._log_lines) > 20:
            self._log_lines = self._log_lines[-20:]
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "\n".join(self._log_lines) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _stop_all(self):
        self._log("停止全部模态")
        self.manager.stop_all()

    def _open_logs(self):
        import os
        try:
            os.startfile(str(self.logs_path))
        except Exception as e:
            logger.warning("打开日志目录失败: %s", e)

    def _on_close(self):
        if self.manager.running_modalities():
            if not messagebox.askyesno("确认退出", "还有模态在运行，确定退出？"):
                return
        self.manager.stop_all()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground=theme.BG_CARD,
            background=theme.BG_CARD,
            foreground=theme.FG_PRIMARY,
            arrowcolor=theme.FG_PRIMARY,
            bordercolor="#2a3038",
        )
    except Exception:
        pass
    DemoConsole(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())