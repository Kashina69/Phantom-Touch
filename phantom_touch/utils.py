"""
utils.py
========
Shared math helpers and simple hand-shape heuristics used across modules.
"""

from __future__ import annotations

import math
from typing import Tuple


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def is_finger_extended(lm, tip_idx: int, pip_idx: int) -> bool:
    # y increases downward; tip above pip -> extended
    return lm[tip_idx].y < lm[pip_idx].y


def norm_dist3(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def is_open_hand(lm_norm) -> bool:
    # Simple heuristic: index and middle extended, or index and ring
    try:
        idx_ext = lm_norm[8].y < lm_norm[6].y
        mid_ext = lm_norm[12].y < lm_norm[10].y
        ring_ext = lm_norm[16].y < lm_norm[14].y
        return (idx_ext and mid_ext) or (idx_ext and ring_ext)
    except Exception:
        return False


