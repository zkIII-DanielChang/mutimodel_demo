"""FunASR 语音识别封装，输入 float32 单声道音频，输出文本。"""

from typing import Optional

import numpy as np

from src.common.logger import get_logger

logger = get_logger("asr.recognizer")


class ASRRecognizer:
    def __init__(
        self,
        model_name: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        device: str = "cpu",
    ):
        self._model = None
        self._init_model(model_name, vad_model, punc_model, device)

    def _init_model(self, model_name, vad_model, punc_model, device) -> None:
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "未安装 FunASR，请执行 pip install --upgrade funasr modelscope"
            ) from e
        logger.info("正在加载 FunASR 模型（首次运行会下载模型，请耐心等待）...")
        self._model = AutoModel(
            model=model_name,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
            disable_update=True,
        )
        logger.info("FunASR 模型已加载: %s", model_name)

    def recognize(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """输入 float32 [-1,1] 单声道音频，返回识别文本。失败返回空串。"""
        if self._model is None or audio is None or audio.size == 0:
            return ""
        try:
            res = self._model.generate(input=audio, fs=sample_rate)
        except Exception as e:
            logger.warning("ASR 识别失败: %s", e)
            return ""
        if not res:
            return ""
        text = res[0].get("text", "") if isinstance(res[0], dict) else ""
        return text.strip()