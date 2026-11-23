"""指纹算法侧比对演示（图像级 SIFT，非实时传感器）。

用法：
    # 默认在 data/fingerprint_images/ 下分别选 registered.bmp 和 probe.bmp
    python scripts/fingerprint_demo.py

    # 显式指定两张图像
    python scripts/fingerprint_demo.py --registered data/fingerprint_images/a.bmp --probe data/fingerprint_images/b.bmp
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.common.config_loader import load_config, resolve_path
from src.common.csv_logger import CSVLogger
from src.common.logger import get_logger
from src.fingerprint.matcher import FingerprintMatcher

logger = get_logger("fingerprint_demo")


class FingerprintDemo:
    def __init__(self, root, matcher, csv_logger):
        self.root = root
        self.matcher = matcher
        self.csv_logger = csv_logger
        self.registered_path = None
        self.probe_path = None
        self._canvas_img = None
        self._build_ui()

    def _build_ui(self):
        self.root.title("指纹算法侧比对演示")
        self.root.geometry("980x760")
        self.root.configure(bg="#f7f8fa")

        # 顶部按钮区
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Button(
            top, text="选择注册指纹", command=self._pick_registered,
        ).pack(side="left", padx=5)
        self.reg_label = ttk.Label(top, text="未选择", foreground="#888888")
        self.reg_label.pack(side="left", padx=5)

        ttk.Button(
            top, text="选择待测指纹", command=self._pick_probe,
        ).pack(side="left", padx=(20, 5))
        self.probe_label = ttk.Label(top, text="未选择", foreground="#888888")
        self.probe_label.pack(side="left", padx=5)

        ttk.Button(
            top, text="开始比对", command=self._compare,
        ).pack(side="right", padx=5)

        # 结果区
        result_frame = ttk.LabelFrame(self.root, text="比对结果", padding=10)
        result_frame.pack(fill="x", padx=10, pady=5)

        self.decision_var = tk.StringVar(value="等待比对")
        ttk.Label(
            result_frame, textvariable=self.decision_var,
            font=("Microsoft YaHei", 22, "bold"), foreground="#555555",
        ).pack(anchor="w", pady=4)

        self.detail_var = tk.StringVar(value="")
        ttk.Label(
            result_frame, textvariable=self.detail_var,
            font=("Microsoft YaHei", 11), foreground="#666666",
        ).pack(anchor="w")

        # 图像区
        image_frame = ttk.Frame(self.root, padding=10)
        image_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(
            image_frame, bg="#ffffff", highlightthickness=1,
            highlightbackground="#dddddd",
        )
        self.canvas.pack(fill="both", expand=True)

        # 底部标注
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        ttk.Label(
            bottom,
            text="算法侧演示：图像级 SIFT 特征比对，非实时指纹传感器链路。上板卡后通过 USB 指纹传感器恢复完整采集。",
            foreground="#b26a00",
            wraplength=920,
        ).pack(side="left")

    def _pick_registered(self):
        path = filedialog.askopenfilename(
            title="选择注册指纹图像",
            initialdir=str(resolve_path("data/fingerprint_images")),
            filetypes=[("图像", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"), ("所有", "*.*")],
        )
        if path:
            self.registered_path = path
            self.reg_label.config(text=Path(path).name, foreground="#333333")

    def _pick_probe(self):
        path = filedialog.askopenfilename(
            title="选择待测指纹图像",
            initialdir=str(resolve_path("data/fingerprint_images")),
            filetypes=[("图像", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"), ("所有", "*.*")],
        )
        if path:
            self.probe_path = path
            self.probe_label.config(text=Path(path).name, foreground="#333333")

    def _compare(self):
        if not self.registered_path or not self.probe_path:
            messagebox.showwarning("提示", "请先选择注册指纹和待测指纹")
            return

        result = self.matcher.compare(self.registered_path, self.probe_path)
        if result is None:
            self.decision_var.set("无法读取图像")
            self.detail_var.set("请确认两张图像都是有效的指纹图片")
            self.canvas.delete("all")
            return

        if result["decision"] == "SAME":
            self.decision_var.set("判定：同一指纹")
            self.decision_var.set("判定：同一指纹")
        else:
            self.decision_var.set("判定：不同指纹")

        # 根据判定改颜色
        color = "#0f9d58" if result["decision"] == "SAME" else "#d93025"
        for child in self.root.winfo_children():
            pass  # 颜色在 Label 创建时固定，这里用 detail 文字颜色区分

        self.detail_var.set(
            f"有效匹配 {result['good_matches']} / 总匹配 {result['total_matches']}  "
            f"| 匹配率 {result['ratio']:.2%}  "
            f"| 注册关键点 {result['registered_keypoints']}  "
            f"| 待测关键点 {result['probe_keypoints']}  "
            f"| 判定阈值 {self.matcher.min_good_matches}"
        )

        vis = self.matcher.visualize(self.registered_path, self.probe_path)
        self._show_image(vis)

        self.csv_logger.log(
            "fingerprint", "decision", result["decision"],
            f"good={result['good_matches']}",
        )
        self.csv_logger.log(
            "fingerprint", "good_matches", str(result["good_matches"]), "",
        )
        self.csv_logger.log(
            "fingerprint", "match_ratio", f"{result['ratio']:.4f}", "",
        )

    def _show_image(self, bgr: np.ndarray):
        if bgr is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                450, 200, text="无法生成匹配可视化", font=("Microsoft YaHei", 14),
            )
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        canvas_w = max(self.canvas.winfo_width(), 900)
        canvas_h = max(self.canvas.winfo_height(), 420)
        pil.thumbnail((canvas_w, canvas_h), Image.LANCZOS)

        self._canvas_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self._canvas_img, anchor="center",
        )


def parse_args():
    p = argparse.ArgumentParser(description="指纹算法侧比对演示")
    p.add_argument("--registered", type=str, default=None, help="注册指纹图像路径")
    p.add_argument("--probe", type=str, default=None, help="待测指纹图像路径")
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio 阈值")
    p.add_argument("--min-good", type=int, default=50, help="判定同一指纹的最少有效匹配数")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths_cfg = load_config("paths")
    log_path = resolve_path(paths_cfg.get("logs_dir", "data/logs")) / "fingerprint.csv"

    try:
        matcher = FingerprintMatcher(
            ratio_threshold=args.ratio,
            min_good_matches=args.min_good,
        )
    except Exception as e:
        print(f"指纹比对器初始化失败: {e}")
        return 1

    csv_logger = CSVLogger(log_path)

    root = tk.Tk()
    demo = FingerprintDemo(root, matcher, csv_logger)
    if args.registered:
        demo.registered_path = args.registered
        demo.reg_label.config(text=Path(args.registered).name, foreground="#333333")
    if args.probe:
        demo.probe_path = args.probe
        demo.probe_label.config(text=Path(args.probe).name, foreground="#333333")
    if args.registered and args.probe:
        root.after(300, demo._compare)

    def on_close():
        csv_logger.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())