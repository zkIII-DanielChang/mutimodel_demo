# 多模态人机交互

## 协作者需要知道

1. 问题写在about/problems里
2. 1. 为避免环境配置复杂，再由于模型很大，github仓库不包含.venv. 我的建议是，把原本的.venv打包拷贝，在下面安装让你解压的时候你给解压了。
   2. github仓库的.venv.rar没有模型，仅用于示例。
3. 下一步任务：以后再写

## 项目简介

基于 UniFace、MediaPipe、FunASR 的车载多模态驾驶辅助演示系统，在普通 Windows 笔记本上运行 8 个交互模态（声纹暂停）。视频模态使用 DirectML 加速，独立窗口展示；语音和指纹使用独立 GUI。所有分析结果写入统一 CSV 日志。

---

## 环境要求

- Windows 10/11
- Python 3.11
- NVIDIA GPU（可选，用于 DirectML 加速）
- 摄像头 index=0，麦克风

---

## 安装

需要放到D:/mutimodel_demo下面

```powershell
# 创建 venv
python -m venv .venv
.venv\Scripts\activate

# 这个时候可以把.venv.rar解压了，不覆盖任何文件

# 基础依赖
pip install numpy opencv-contrib-python pyyaml scipy scikit-image sounddevice pillow pytest

# 人脸/姿态/眼动
pip install uniface

# 手势/疲劳/表情
pip install mediapipe

# 语音识别

## 原本是：
## pip install --upgrade funasr modelscope torch torchaudio --index-url https://download.pytorch.org/whl/cpu
## 这时候报错： WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='download.pytorch.org', port=443): Read timed out. (read timeout=15)")': /whl/cpu/funasr/

## 所以改成官方源：
pip install --upgrade funasr modelscope torch torchaudio

# 声纹
pip install voice-fingerprint

# GPU 加速（DirectML，不需要 CUDA）
pip install onnxruntime-directml
```

**注意**：`onnxruntime` 和 `onnxruntime-directml` 不能共存。如果之前装过 `onnxruntime`，先 `pip uninstall onnxruntime onnxruntime-gpu -y`，再装 directml。

验证 provider：

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

应输出 `['DmlExecutionProvider', 'CPUExecutionProvider']`。

---

## 模型文件

| 模型 | 位置 | 大小 | 来源 |
| --- | --- | --- | --- |
| RetinaFace | `~/.uniface/models/retinaface_mnet_v2.onnx` | 12.5 MB | UniFace 自动下载或手动放置 |
| ArcFace | `~/.uniface/models/arcface_mnet.onnx` | 13.6 MB | 同上 |
| MobileGaze | `~/.uniface/models/gaze_mobileone_s0.onnx` | 5 MB | 同上 |
| 手势 | `models/gesture_recognizer.task` | 2.4 MB | MediaPipe 官方 |
| 面部 Landmarker | `models/face_landmarker.task` | 3.7 MB | MediaPipe 官方 |
| FunASR | `~/.cache/modelscope/` | 约 2 GB | 首次运行自动下载 |
| WeSpeaker | `~/.cache/huggingface/` | 26.5 MB | 首次运行自动下载 |

MediaPipe 两个模型需手动下载：

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" -OutFile "models\gesture_recognizer.task"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models\face_landmarker.task"
```

---

## 运行

### 人脸注册（首次必须做）

```powershell
python scripts\enroll_face.py --name alice --camera 0 --num-samples 20
```

采集时正对摄像头、光照均匀、距离 0.5–1 米。采集完成后 `data/registered_faces/face_gallery.npz` 生成。

### 各模态启动命令

| 模态 | 命令 | FPS |
| --- | --- | --- |
| 人脸识别 | `python scripts\run_face.py --backend uniface --camera 0 --recognize --prefer-cuda` | 18–19 |
| 头部姿态 | `python scripts\run_face.py --backend uniface --camera 0 --head-pose --prefer-cuda` | 23 |
| 眼动 | `python scripts\run_face.py --backend uniface --camera 0 --gaze --gaze-interval 5 --prefer-cuda` | 16–17 |
| 手势 | `python scripts\run_gesture.py --camera 0` | 15–25 |
| 疲劳+表情 | `python scripts\run_fatigue.py --camera 0` | 27 |
| 语音识别 | `python scripts\run_asr.py` | — |
| 指纹比对 | `python scripts\fingerprint_demo.py` | — |
| 声纹 | `python scripts\voiceprint_demo.py` | 暂停 |

**摄像头独占**：人脸、手势、疲劳三个视频模态不能同时开，必须分时切换。语音和指纹不占摄像头，可与人脸窗口并行。

**关闭窗口**：视频窗口按 `q`，GUI 点右上角 X，终端程序按 `Ctrl+C`。如果卡住，`Get-Process python*` 查 PID，`Stop-Process -Id <Id> -Force`。

### 第一版 DMS 演示（笔记本摄像头）

第一版 DMS 将笔记本摄像头作为驾驶员监控摄像头，在一个窗口中完成实时闭环：

- 正常驾驶状态
- 持续转头或低头的分心提醒
- 持续闭眼、打哈欠的疲劳提醒
- 人脸丢失、脸部过小、画面过暗或过亮时的 DMS 不可用提示
- Windows 声音告警和 `data/logs/dms.csv` 事件日志

确保 `models/face_landmarker.task` 存在后运行：

```powershell
python scripts\\run_dms.py
```

如果笔记本摄像头不是 index 0：

```powershell
python scripts\\run_dms.py --camera 1
```

窗口按键：

- `q`：退出
- `c`：将当前头部 pitch 校准为 0 度
- `m`：切换镜像显示

演示时先正视摄像头确认“正常驾驶”，再持续转头约 2 秒触发分心提醒，闭眼约 1–2 秒触发疲劳分级告警，最后遮挡摄像头验证“DMS 不可用”。当前阈值是演示初值，需要根据笔记本摄像头位置、光照和坐姿调整 `configs/thresholds.yaml`。

九张卡片，点击启停对应模态。人脸类模态自动带 `--prefer-cuda`。按 `Esc` 停止全部，空格切换场景。

---

## 目录结构

```text
mutimodel_demo/
├── configs/                  # YAML 配置
│   ├── camera.yaml          # 摄像头索引和分辨率
│   ├── paths.yaml           # 数据和模型路径
│   ├── thresholds.yaml      # 检测/识别阈值
│   ├── commands.yaml        # 语音指令集
│   └── demo_scenes.yaml     # 五幕演示配置
├── src/
│   ├── common/              # 配置加载、日志、CSV、设备探测
│   ├── face/                # 检测、识别、姿态、眼动
│   ├── gesture/             # MediaPipe 手势
│   ├── asr/                 # FunASR + 指令匹配
│   ├── voiceprint/          # 声纹引擎
│   ├── fatigue/             # EAR/MAR、状态分类、表情
│   ├── fingerprint/         # 细节点 + SIFT
│   └── console/             # 总控台进程管理和主题
├── scripts/                  # 各模态入口脚本
├── tests/                    # pytest 用例
├── models/                   # MediaPipe 模型
├── data/
│   ├── registered_faces/    # 人脸注册库
│   ├── registered_voices/   # 声纹库
│   ├── fingerprint_images/  # 指纹图像
│   └── logs/                # CSV 日志
└── docs/
    └── demo_drive_assist.md # 五幕演示手册
```

---

## 已知限制

| 项 | 说明 |
| --- | --- |
| 声纹跨语气 | 同一人换语气识别不稳，暂停纳入演示。改善方案：多语气注册 + 阈值校准，见后续 |
| 表情类别 | 基于 MediaPipe blendshapes 的演示级规则，5 类，不是文档的 8 类 ViT 模型 |
| 指纹 | 图像级 SIFT 比对，非实时传感器链路。窗口标注“算法侧演示” |
| 语音错别字 | 端到端 ASR 正常表现，指令匹配层有容忍 |
| 摄像头独占 | 三个视频模态不能同屏，只能分时演示 |
| GPU | DirectML 加速人脸类推理，手势和疲劳走 MediaPipe XNNPACK，不经过 DirectML |
| 测试 | 30 个用例，28 通过，2 个 gaze 测试待修（旧接口） |

---

## 演示动线

按 `docs/demo_drive_assist.md` 五幕：

1. 上车认证：人脸识别 + 指纹
2. 驾驶监测：疲劳 + 表情
3. 手势控制：拇指向上/握拳/张开手掌
4. 语音指令：导航/音乐/空调
5. 复盘：5 个 CSV 同时展示

---

## 故障排查

**摄像头打不开**：`Get-Process python*` 清残留进程，或换 `--camera 1`。

**模型下载失败**：UniFace 模型从 GitHub/HuggingFace 拉取，国内可能超时。手动下载放到 `~/.uniface/models/`，文件名与报错一致即可。

**onnxruntime provider 只有 CPU**：确认装的是 `onnxruntime-directml` 而不是 `onnxruntime`；`src/common/device.py` 已自动注册 nvidia DLL 目录，无需手动改 PATH。

**总控台子进程窗口不出现**：`data/logs/<modality>.log` 有子进程完整输出，查日志定位。

---

## 参考

- UniFace: `pip install uniface`，[文档](https://github.com/yakhyo/uniface)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [FunASR](https://github.com/modelscope/FunASR)
- voice-fingerprint: `pip install voice-fingerprint`# mutimodel_demo
