"""
filters.py
==========
Low-latency smoothing utilities. Implements the One Euro filter which provides
good temporal smoothing with minimal lag by adapting cutoff based on motion.

Exported:
- OneEuroFilter: class with `filter(value, timestamp_s)` → float
"""

from __future__ import annotations

import math


class LowPass:
    def __init__(self, alpha: float, init_value: float | None = None):
        self.alpha = alpha
        self.initialized = init_value is not None
        self.last = init_value if init_value is not None else 0.0

    def apply(self, value: float, alpha: float | None = None) -> float:
        a = self.alpha if alpha is None else alpha
        if not self.initialized:
            self.last = value
            self.initialized = True
        self.last = a * value + (1.0 - a) * self.last
        return self.last


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_hat = LowPass(alpha=0.0)
        self.dx_hat = LowPass(alpha=0.0)
        self.last_time = None

    def filter(self, value: float, timestamp_s: float) -> float:
        if self.last_time is None:
            self.last_time = timestamp_s
            self.x_hat = LowPass(alpha=1.0, init_value=value)
            self.dx_hat = LowPass(alpha=1.0, init_value=0.0)
            return value
        dt = max(1e-6, timestamp_s - self.last_time)
        self.last_time = timestamp_s
        dx = (value - self.x_hat.last) / dt
        alpha_d = _smoothing_factor(dt, self.d_cutoff)
        edx = self.dx_hat.apply(dx, alpha=alpha_d)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = _smoothing_factor(dt, cutoff)
        return self.x_hat.apply(value, alpha=alpha)


