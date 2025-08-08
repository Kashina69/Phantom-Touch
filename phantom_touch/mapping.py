"""
mapping.py
=========
Cursor mapping utilities for Normal (absolute) and Draw (relative/clutch) modes,
including a simple velocity gate to suppress jitter spikes.
"""

from __future__ import annotations

import time
from typing import Tuple
import numpy as np
import pyautogui

from .filters import OneEuroFilter


class CursorMapper:
    def __init__(self, screen_w: int, screen_h: int, min_cutoff: float, beta: float, d_cutoff: float, max_speed_px_s: float, edge_db_px: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.max_speed = max_speed_px_s
        self.edge_db = edge_db_px
        self.last_screen_pos: Tuple[float, float] | None = None
        self.move_last_t = time.time()

    def move_absolute(self, x_img: int, y_img: int, margin_x: int, margin_y: int, img_w: int, img_h: int) -> None:
        abs_x = np.interp(x_img, (margin_x, img_w - margin_x), (0, self.screen_w))
        abs_y = np.interp(y_img, (margin_y, img_h - margin_y), (0, self.screen_h))
        tnow = time.time()
        fx = self.filter_x.filter(float(abs_x), tnow)
        fy = self.filter_y.filter(float(abs_y), tnow)
        if self.last_screen_pos is None:
            self.last_screen_pos = (fx, fy)
        dt = max(1e-3, tnow - self.move_last_t)
        max_delta = self.max_speed * dt + 5
        if abs(fx - self.last_screen_pos[0]) <= max_delta and abs(fy - self.last_screen_pos[1]) <= max_delta:
            tx = int(max(self.edge_db, min(self.screen_w - self.edge_db, fx)))
            ty = int(max(self.edge_db, min(self.screen_h - self.edge_db, fy)))
            pyautogui.moveTo(tx, ty, duration=0, _pause=False)
            self.last_screen_pos = (fx, fy)
            self.move_last_t = tnow

    def move_relative(self, dx_img: float, dy_img: float) -> None:
        cx, cy = pyautogui.position()
        nx = max(0, min(self.screen_w - 1, int(cx + dx_img)))
        ny = max(0, min(self.screen_h - 1, int(cy + dy_img)))
        pyautogui.moveTo(nx, ny, duration=0, _pause=False)


