"""
visual.py
=========
Debug overlay drawing for mode, FPS, pinch value, motion intensity, hold state,
and confidence, plus basic hand landmark points.
"""

from __future__ import annotations

import cv2


def draw_overlay(frame, info_lines, lm_px=None):
    if lm_px is not None:
        try:
            cv2.circle(frame, (lm_px[8].x, lm_px[8].y), 6, (0, 128, 255), -1)
            cv2.circle(frame, (lm_px[4].x, lm_px[4].y), 6, (255, 128, 0), -1)
        except Exception:
            pass
    y0 = 24
    for i, txt in enumerate(info_lines):
        cv2.putText(frame, txt, (10, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)


