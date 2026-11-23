"""指令模糊匹配：字符 bigram 余弦相似度。"""

from typing import Dict, List

import numpy as np

from src.common.logger import get_logger

logger = get_logger("asr.command_matcher")


def _char_bigrams(text: str) -> Dict[str, int]:
    text = text.replace(" ", "")
    if len(text) < 2:
        return {text: 1} if text else {}
    grams: Dict[str, int] = {}
    for i in range(len(text) - 1):
        g = text[i : i + 2]
        grams[g] = grams.get(g, 0) + 1
    return grams


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], dtype=np.float64)
    vb = np.array([b.get(k, 0) for k in keys], dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class CommandMatcher:
    def __init__(self, commands: List[str]):
        self.commands = list(commands)
        self._command_grams = [_char_bigrams(c) for c in self.commands]
        logger.info("指令匹配器已加载 %d 条指令", len(self.commands))

    def match(self, text: str, threshold: float = 0.5) -> Dict[str, object]:
        """返回 {'command': 最佳指令或 None, 'similarity': 最高分, 'matched': bool}。"""
        if not text or not self.commands:
            return {"command": None, "similarity": 0.0, "matched": False}
        text_grams = _char_bigrams(text)
        sims = [_cosine(text_grams, cg) for cg in self._command_grams]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        matched = best_sim >= threshold
        return {
            "command": self.commands[best_idx] if matched else None,
            "similarity": best_sim,
            "matched": matched,
        }