"""环境自检：逐项检查 9 种模态所需依赖是否安装、关键 API 是否可用。

用法：
    python check_env.py
"""

import importlib
import platform
import subprocess
import sys
from pathlib import Path


RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" | {detail}"
    print(line)


def check_import(module_name, attr=None, display_name=None, required=True):
    display = display_name or module_name
    try:
        mod = importlib.import_module(module_name)
        if attr is not None:
            if not hasattr(mod, attr):
                record(display, False, f"已导入但缺少属性 {attr}")
                return None
        version = getattr(mod, "__version__", "unknown")
        record(display, True, f"version={version}")
        return mod
    except Exception as e:
        record(display, not required, f"{type(e).__name__}: {e}")
        return None


def check_python():
    v = sys.version_info
    record(
        "Python 版本",
        v.major == 3 and v.minor == 11,
        f"{platform.python_version()} ({'venv' if sys.prefix != sys.base_prefix else 'global'})",
    )
    record("平台", True, f"{platform.system()} {platform.machine()}")


def check_base():
    check_import("numpy")
    check_import("cv2", display_name="opencv")
    check_import("yaml", display_name="pyyaml")
    check_import("scipy")
    check_import("skimage", display_name="scikit-image")
    check_import("tqdm")
    check_import("requests")

    # opencv contrib 检查
    try:
        import cv2
        has_sift = hasattr(cv2, "SIFT_create")
        record("opencv contrib (SIFT)", has_sift, "SIFT_create 可用" if has_sift else "缺少 SIFT，指纹比对不可用")
        # 检查是否同时装了 opencv-python 与 contrib
        try:
            from importlib.metadata import distributions
            names = {d.metadata["Name"].lower() for d in distributions() if d.metadata["Name"]}
            both = "opencv-python" in names and "opencv-contrib-python" in names
            record(
                "opencv 安装唯一性",
                not both,
                "同时存在 opencv-python 和 opencv-contrib-python，建议卸载前者" if both else "无冲突",
            )
        except Exception as e:
            record("opencv 安装唯一性", True, f"检查跳过: {e}")
    except Exception as e:
        record("opencv contrib (SIFT)", False, str(e))


def check_face():
    uniface = check_import("uniface", display_name="uniface")
    if uniface is not None:
        for attr, label in [
            ("RetinaFace", "uniface.RetinaFace"),
            ("ArcFace", "uniface.ArcFace"),
        ]:
            record(label, hasattr(uniface, attr), "可用" if hasattr(uniface, attr) else "缺少，需确认导入路径")

    try:
        from uniface.detection import RetinaFace  # noqa: F401
        from uniface.recognition import ArcFace  # noqa: F401
        record("uniface 子模块导入", True, "detection.RetinaFace / recognition.ArcFace")
    except Exception as e:
        record("uniface 子模块导入", False, f"{type(e).__name__}: {e}")

    ort = check_import("onnxruntime", display_name="onnxruntime")
    if ort is not None:
        providers = ort.get_available_providers()
        record("onnxruntime providers", "CPUExecutionProvider" in providers, str(providers))


def check_torch():
    torch = check_import("torch")
    if torch is not None:
        cuda = torch.cuda.is_available()
        record("torch CUDA", cuda, f"cuda={cuda}, device_count={torch.cuda.device_count() if cuda else 0}")
    check_import("torchvision", required=False)
    check_import("torchaudio", required=False)


def check_mediapipe():
    mp = check_import("mediapipe", display_name="mediapipe")
    if mp is not None:
        try:
            from mediapipe.tasks import python  # noqa: F401
            from mediapipe.tasks.python import vision  # noqa: F401
            record("mediapipe tasks API", True, "python.tasks / vision 可用")
        except Exception as e:
            record("mediapipe tasks API", False, f"{type(e).__name__}: {e}")

    model_path = Path("models") / "gesture_recognizer.task"
    record(
        "手势模型文件",
        model_path.exists(),
        str(model_path) if model_path.exists() else f"缺少 {model_path}，需从 MediaPipe 官方下载",
    )


def check_asr():
    check_import("funasr")
    check_import("modelscope")
    # 麦克风后端二选一
    sd = check_import("sounddevice", required=False)
    pa = check_import("pyaudio", required=False)
    record(
        "麦克风采集后端",
        sd is not None or pa is not None,
        "sounddevice 可用" if sd is not None else ("pyaudio 可用" if pa is not None else "两者均未安装"),
    )


def check_voiceprint():
    vf = check_import("voicefingerprint", display_name="voice-fingerprint", required=False)
    ws = check_import("wespeaker", display_name="wespeaker", required=False)
    record(
        "声纹后端",
        vf is not None or ws is not None,
        "voice-fingerprint 可用" if vf is not None else ("wespeaker 可用" if ws is not None else "两者均未安装"),
    )


def check_camera():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            record("摄像头 index=0", False, "无法打开，可能被占用或索引不对")
        else:
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                record("摄像头 index=0", True, f"分辨率 {w}x{h}")
            else:
                record("摄像头 index=0", False, "打开成功但读帧失败")
        cap.release()
    except Exception as e:
        record("摄像头 index=0", False, f"{type(e).__name__}: {e}")


def check_microphone():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        record("麦克风设备", len(inputs) > 0, f"输入设备数={len(inputs)}")
    except Exception:
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            count = p.get_device_count()
            p.terminate()
            record("麦克风设备", count > 0, f"PyAudio 设备数={count}")
        except Exception as e:
            record("麦克风设备", False, f"{type(e).__name__}: {e}")


def check_nvidia():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            record("nvidia-smi", True, out.stdout.strip().splitlines()[0])
        else:
            record("nvidia-smi", False, out.stderr.strip() or "无输出")
    except FileNotFoundError:
        record("nvidia-smi", False, "未找到 nvidia-smi，可能无 NVIDIA GPU 或驱动未装")
    except Exception as e:
        record("nvidia-smi", False, f"{type(e).__name__}: {e}")


def summarize():
    total = len(RESULTS)
    failed = [r for r in RESULTS if not r[1]]
    optional_failed = [r for r in failed if "可选" in r[0] or r[0] in ("sounddevice", "pyaudio", "wespeaker", "torchvision", "torchaudio")]
    print("\n" + "=" * 60)
    print(f"总计 {total} 项检查，失败 {len(failed)} 项")
    if failed:
        print("失败项：")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
    else:
        print("全部通过。")
    print("=" * 60)


def main():
    print("=" * 60)
    print("环境自检开始")
    print("=" * 60)
    check_python()
    print("-" * 60 + " 基础依赖")
    check_base()
    print("-" * 60 + " 人脸 / 眼动 / 头部姿态 / 表情")
    check_face()
    print("-" * 60 + " PyTorch")
    check_torch()
    print("-" * 60 + " 手势 / 疲劳")
    check_mediapipe()
    print("-" * 60 + " 语音识别")
    check_asr()
    print("-" * 60 + " 声纹识别")
    check_voiceprint()
    print("-" * 60 + " 硬件")
    check_camera()
    check_microphone()
    check_nvidia()
    summarize()


if __name__ == "__main__":
    main()