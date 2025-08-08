"""
gestures.py
===========
Gesture detection and state machines for clicks, double-click, holds, right
click, backup click heuristics, and mode toggles (3D intensity and triple-tap).

This module is pure logic. It takes MediaPipe landmarks and timing and emits
cursor actions or mode toggles. State is carried in a Gestures object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time

from .utils import is_open_hand, is_finger_extended, norm_dist3


@dataclass
class GestureState:
    pinch_is_on: bool = False
    pinch_on_time: float = 0.0
    last_tap_time: float = 0.0
    hold_active: bool = False
    toggle_state: str = "idle"
    toggle_last_time: float = 0.0
    recent_wrist_3d: List[Tuple[float, float, float, float]] = field(default_factory=list)
    index_y_history: List[Tuple[float, float]] = field(default_factory=list)
    middle_y_history: List[Tuple[float, float]] = field(default_factory=list)
    open_tap_times: List[float] = field(default_factory=list)
    last_synth_click_time: float = 0.0
    im_touch_on: bool = False

    # telemetry
    last_pinch_val_3d: Optional[float] = None
    last_motion_intensity: float = 0.0
    last_hand_conf: float = 0.0


def motion_intensity_3d(samples: List[Tuple[float, float, float, float]], xw: float, yw: float, zw: float) -> float:
    if not samples:
        return 0.0
    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    zs = [s[3] for s in samples]
    dx = (max(xs) - min(xs)) * xw
    dy = (max(ys) - min(ys)) * yw
    dz = (max(zs) - min(zs)) * zw
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def update_mode_toggle(state: GestureState, now: float, lm_norm, hand_conf: float, cfg) -> bool:
    # push wrist sample
    wrist = lm_norm[0]
    state.recent_wrist_3d.append((now, wrist.x, wrist.y, wrist.z))
    window = cfg.toggle_window_s
    while state.recent_wrist_3d and now - state.recent_wrist_3d[0][0] > window:
        state.recent_wrist_3d.pop(0)
    intensity = motion_intensity_3d(state.recent_wrist_3d, cfg.toggle_x_weight, cfg.toggle_y_weight, cfg.toggle_z_weight)
    toggled = False
    if hand_conf >= cfg.hand_conf_min and is_open_hand(lm_norm):
        # 1) intensity toggle
        if intensity >= cfg.toggle_intensity_min:
            if state.toggle_state == "idle":
                state.toggle_state = "armed"
                state.toggle_last_time = now
            elif state.toggle_state == "armed" and (now - state.toggle_last_time) > 0.12:
                state.toggle_state = "cooldown"
                state.toggle_last_time = now
                toggled = True
        # 2) triple open-hand taps
        if intensity >= cfg.open_tap_intensity and (not state.open_tap_times or (now - state.open_tap_times[-1]) >= cfg.open_tap_min_interval_s):
            state.open_tap_times.append(now)
            state.open_tap_times[:] = [t for t in state.open_tap_times if now - t <= cfg.triple_tap_window_s]
            if len(state.open_tap_times) >= 3:
                toggled = True
                state.open_tap_times.clear()
    if state.toggle_state == "cooldown" and (now - state.toggle_last_time) > 0.6:
        state.toggle_state = "idle"
    if not is_open_hand(lm_norm) and state.toggle_state == "armed":
        state.toggle_state = "idle"
    state.last_motion_intensity = intensity
    return toggled


def normalized_pinch_3d(lm_norm) -> float:
    wrist = lm_norm[0]
    index_mcp = lm_norm[5]
    index_tip = lm_norm[8]
    thumb_tip = lm_norm[4]
    hand_scale = max(1e-6, norm_dist3(wrist, index_mcp))
    return norm_dist3(index_tip, thumb_tip) / hand_scale


def update_pinch_hold_clicks(state: GestureState, now: float, lm_norm, cfg) -> Tuple[Optional[str], Optional[str]]:
    """Update core pinch/hold/double-click. Returns (action, click_type)
    action: "HOLD"/"POINTER" when state changes. click_type: "CLICK"/"DOUBLE" if fired.
    """
    pinch_val = normalized_pinch_3d(lm_norm)
    state.last_pinch_val_3d = pinch_val
    action = None
    click_type = None

    if state.pinch_is_on:
        if pinch_val > cfg.pinch3d_off:
            state.pinch_is_on = False
            pinch_off_time = now
            if (pinch_off_time - state.pinch_on_time) <= cfg.tap_max_s and state.last_motion_intensity <= cfg.pinch_stability_intensity_max:
                if (pinch_off_time - state.last_tap_time) <= cfg.double_tap_window_s:
                    click_type = "DOUBLE"
                    state.last_tap_time = 0.0
                else:
                    state.last_tap_time = pinch_off_time
            if state.hold_active:
                action = "POINTER"
                state.hold_active = False
    else:
        if pinch_val < cfg.pinch3d_on and state.last_motion_intensity <= cfg.pinch_stability_intensity_max:
            state.pinch_is_on = True
            state.pinch_on_time = now

    if state.pinch_is_on and not state.hold_active and (now - state.pinch_on_time) >= cfg.hold_delay_s:
        state.hold_active = True
        action = "HOLD"

    return action, click_type


def update_backup_clicks(state: GestureState, now: float, h: int, lm_px, cfg) -> Optional[str]:
    """Rapid finger motions and super-quick pinch release → synthetic clicks.
    Returns "LEFT"/"RIGHT" or None.
    """
    # index rapid motion
    state.index_y_history.append((now, lm_px[8].y))
    cut = now - cfg.index_click_window_s
    while state.index_y_history and state.index_y_history[0][0] < cut:
        state.index_y_history.pop(0)
    if state.index_y_history:
        ys = [y for (_, y) in state.index_y_history]
        if (max(ys) - min(ys)) >= (cfg.index_click_min_amp_frac * h) and (now - state.last_synth_click_time) >= cfg.click_cooldown_s:
            state.last_synth_click_time = now
            return "LEFT"

    # middle rapid motion (only if extended)
    state.middle_y_history.append((now, lm_px[12].y))
    cutm = now - cfg.middle_click_window_s
    while state.middle_y_history and state.middle_y_history[0][0] < cutm:
        state.middle_y_history.pop(0)
    middle_extended = is_finger_extended(lm_norm=None, tip_idx=0, pip_idx=0)  # placeholder; caller should gate by extension
    # We don't know extension here; caller should check.
    return None


def index_middle_hold(state: GestureState, now: float, lm_norm, cfg) -> Optional[str]:
    """Index+Middle touch toggles HOLD while touching, release when apart.
    Returns action ("HOLD"/"POINTER") when state changes.
    """
    im_dist = norm_dist3(lm_norm[8], lm_norm[12]) / max(1e-6, norm_dist3(lm_norm[0], lm_norm[5]))
    if state.im_touch_on:
        if im_dist > cfg.im_touch_off:
            state.im_touch_on = False
            if state.hold_active:
                state.hold_active = False
                return "POINTER"
    else:
        if im_dist < cfg.im_touch_on:
            state.im_touch_on = True
            state.hold_active = True
            return "HOLD"
    return None


