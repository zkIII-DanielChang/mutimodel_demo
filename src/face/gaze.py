"""眼动追踪封装：UniFace MobileGaze，默认 MobileOne-S0 轻量骨干。"""

from typing import Dict, Optional

import numpy as np

from src.common.device import get_onnx_providers
from src.common.logger import get_logger

logger = get_logger("face.gaze")

# 可选：RESNET18 / RESNET34 / RESNET50 / MOBILENET_V2 / MOBILEONE_S0
# MobileOne-S0 体积最小（4.8MB），面向实时边缘应用
DEFAULT_GAZE_MODEL = "MOBILEONE_S0"


class GazeEstimator:
    def __init__(self, prefer_cuda: bool = False, model_name: str = DEFAULT_GAZE_MODEL):
        self._estimator = None
        self._available = False
        self._model_name = model_name
        self._init_model(prefer_cuda)

    def _init_model(self, prefer_cuda: bool) -> None:
        try:
            from uniface.gaze import MobileGaze
        except ImportError:
            logger.warning("UniFace gaze 模块不可用，眼动功能禁用")
            return
        except Exception as e:
            logger.warning("UniFace gaze 导入异常，眼动功能禁用: %s", e)
            return

        providers = get_onnx_providers(prefer_cuda=prefer_cuda)

        # 优先用 GazeWeights 枚举，兼容不同版本的 model_name 参数
        try:
            from uniface.constants import GazeWeights

            weights = getattr(GazeWeights, self._model_name.upper())
            self._estimator = MobileGaze(model_name=weights, providers=providers)
        except (ImportError, AttributeError, TypeError):
            # 回退：直接传字符串
            try:
                self._estimator = MobileGaze(model_name=self._model_name, providers=providers)
            except TypeError:
                try:
                    self._estimator = MobileGaze()
                except Exception as e:
                    logger.warning("MobileGaze 初始化失败，眼动功能禁用: %s", e)
                    return
            except Exception as e:
                logger.warning("MobileGaze 初始化失败，眼动功能禁用: %s", e)
                return
        except Exception as e:
            logger.warning("MobileGaze 初始化失败，眼动功能禁用: %s", e)
            return

        self._available = True
        logger.info("MobileGaze 眼动估计器已加载, model=%s", self._model_name)

    def is_available(self) -> bool:
        return self._available

    def estimate(self, face_crop: np.ndarray) -> Optional[Dict[str, float]]:
        """输入人脸裁剪图，返回 {'yaw': 度, 'pitch': 度}，失败返回 None。"""
        if not self._available or self._estimator is None:
            return None
        if face_crop is None or face_crop.size == 0:
            return None
        try:
            result = self._estimator.estimate(face_crop)
        except Exception as e:
            logger.debug("gaze 估计失败: %s", e)
            return None
        if result is None:
            return None
        try:
            yaw = float(np.degrees(result.yaw))
            pitch = float(np.degrees(result.pitch))
        except Exception as e:
            logger.debug("gaze 结果解析失败: %s", e)
            return None
        return {"yaw": yaw, "pitch": pitch}