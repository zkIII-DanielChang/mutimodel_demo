"""基于 MediaPipe Face Landmarker blendshapes 的演示级表情分类。"""

from typing import Dict, List, Optional


def _get_score(blendshapes: List, name: str) -> float:
    for b in blendshapes:
        if b.category_name == name:
            return float(b.score)
    return 0.0


def classify_emotion(blendshapes: Optional[List]) -> Dict[str, object]:
    """返回 {'label': str, 'score': float}。无法判断时返回 neutral 0.0。"""
    if not blendshapes:
        return {"label": "neutral", "score": 0.0}

    smile = max(
        _get_score(blendshapes, "mouthSmileLeft"),
        _get_score(blendshapes, "mouthSmileRight"),
    )
    frown = max(
        _get_score(blendshapes, "mouthFrownLeft"),
        _get_score(blendshapes, "mouthFrownRight"),
    )
    brow_down = max(
        _get_score(blendshapes, "browDownLeft"),
        _get_score(blendshapes, "browDownRight"),
    )
    brow_up = max(
        _get_score(blendshapes, "browOuterUpLeft"),
        _get_score(blendshapes, "browOuterUpRight"),
    )
    jaw_open = _get_score(blendshapes, "jawOpen")
    eye_squint = max(
        _get_score(blendshapes, "eyeSquintLeft"),
        _get_score(blendshapes, "eyeSquintRight"),
    )

    candidates = {
        "happy": max(smile, eye_squint * 0.8),
        "sad": max(frown, brow_down * 0.5),
        "angry": max(brow_down, frown * 0.8),
        "surprised": max(jaw_open, brow_up),
        "neutral": 0.35,
    }
    label = max(candidates, key=candidates.get)
    score = float(candidates[label])
    if score < 0.25:
        return {"label": "neutral", "score": 0.0}
    return {"label": label, "score": score}