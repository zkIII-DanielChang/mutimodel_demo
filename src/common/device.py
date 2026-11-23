"""设备探测与 provider 选择。

关键修复：
1. 启动时自动把 pip 安装的 nvidia/*/bin 目录注册到 Windows DLL 搜索路径，
   解决 cublasLt64_12.dll、cudnn_ops64_9.dll 等找不到导致静默回退 CPU 的问题。
2. 优先调用 onnxruntime 自带的 preload_dlls()（1.26 版本有），
   它会自动处理 CUDA/cuDNN 的预加载。
3. provider 优先级：CUDA → DirectML → CPU，可在 prefer_cuda=False 时强制 CPU。
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

# ---------------- NVIDIA DLL 路径注册 ----------------
def _iter_nvidia_bin_dirs():
    """枚举 .venv/Lib/site-packages/nvidia/*/bin 下所有存在的目录。"""
    try:
        site_packages = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            return
        for pkg_dir in nvidia_root.iterdir():
            if not pkg_dir.is_dir():
                continue
            bin_dir = pkg_dir / "bin"
            if bin_dir.exists():
                yield bin_dir
    except Exception:
        return


def _register_nvidia_dll_dirs() -> List[str]:
    """把 nvidia/*/bin 注册到 Windows DLL 搜索路径。返回注册成功的目录列表。"""
    registered: List[str] = []
    for bin_dir in _iter_nvidia_bin_dirs():
        try:
            os.add_dll_directory(str(bin_dir))
            registered.append(str(bin_dir))
        except OSError:
            pass
        except Exception:
            pass
    return registered


def _try_preload_onnx_dlls() -> bool:
    """调用 onnxruntime.preload_dlls()（1.26+ 支持），失败返回 False。"""
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
            return True
    except Exception:
        pass
    return False


# 模块加载时立即执行：先注册目录，再尝试 preload
_REGISTERED_DLL_DIRS = _register_nvidia_dll_dirs()
_PRELOAD_OK = _try_preload_onnx_dlls()


# ---------------- Provider 选择 ----------------
def get_onnx_providers(prefer_cuda: bool = True) -> List[str]:
    """返回 provider 优先级列表。

    prefer_cuda=True：优先 CUDA，其次 DirectML，最后 CPU。
    prefer_cuda=False：只用 CPU。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    providers: List[str] = []

    if prefer_cuda:
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        elif "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")

    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    return providers or ["CPUExecutionProvider"]


def has_gpu() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    p = ort.get_available_providers()
    return "CUDAExecutionProvider" in p or "DmlExecutionProvider" in p


def has_cuda() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def describe_devices() -> Dict[str, object]:
    """返回 onnxruntime 版本、可用 provider、已注册 DLL 目录、preload 状态。"""
    info: Dict[str, object] = {}
    try:
        import onnxruntime as ort
        info["onnxruntime_version"] = ort.__version__
        info["available_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime_version"] = None
        info["available_providers"] = []
    info["registered_dll_dirs"] = _REGISTERED_DLL_DIRS
    info["preload_dlls_ok"] = _PRELOAD_OK
    return info