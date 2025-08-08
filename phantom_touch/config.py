"""
config.py
=========
Centralized configuration and CLI argument parsing for PhantomTouch.

This module defines default values and exposes a function `parse_args()` that
returns a populated namespace. The rest of the app should import from here to
avoid scattering configuration across modules.

Key groups:
- Camera & processing: capture size, downscaled processing width, backend.
- Filtering & speed: One Euro filter params, velocity gating to suppress spikes.
- Gestures: pinch thresholds, timings, 3D motion toggle, triple-tap toggle,
  backup click thresholds, index+middle (iron-man) hold.
- Debug: landmark overlay and metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Defaults:
    # Camera & performance
    camera_width: int = 640
    camera_height: int = 480
    process_input_width: int = 320
    active_area_ratio: float = 0.85
    mediapipe_complexity: int = 0  # 0 fast, 1 accurate

    # Filtering (One Euro) and velocity gating
    min_cutoff: float = 1.7
    beta: float = 0.3
    d_cutoff: float = 1.0
    max_speed_px_per_s: float = 5000.0
    edge_deadband_px: int = 2

    # Pinch thresholds and timings (normalized 3D by hand scale)
    pinch3d_on: float = 0.055
    pinch3d_off: float = 0.075
    hold_delay_s: float = 0.18
    tap_max_s: float = 0.14
    double_tap_window_s: float = 0.30
    pinch_stability_intensity_max: float = 0.12

    # Drawing mode relative movement
    draw_gain: float = 1.25

    # Mode toggle by 3D motion intensity
    toggle_window_s: float = 0.6
    toggle_intensity_min: float = 0.22
    toggle_x_weight: float = 0.3
    toggle_y_weight: float = 1.0
    toggle_z_weight: float = 1.1
    hand_confidence_min: float = 0.5

    # Backup gestures & triple tap
    index_click_window_s: float = 0.25
    index_click_min_amp_frac: float = 0.06
    middle_click_window_s: float = 0.25
    middle_click_min_amp_frac: float = 0.06
    click_cooldown_s: float = 0.25
    backup_pinch_tap_max_s: float = 0.08
    im_touch_on: float = 0.060
    im_touch_off: float = 0.090
    triple_tap_window_s: float = 0.8
    open_tap_intensity: float = 0.18
    open_tap_min_interval_s: float = 0.12

    # Gesture cadence
    gesture_every: int = 0

    # Debug
    vis_debug: bool = True


def add_args(parser: argparse.ArgumentParser, d: Defaults) -> None:
    # Camera and processing
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0,1,2,...) to open")
    parser.add_argument("--width", type=int, default=d.camera_width, help="Camera capture width")
    parser.add_argument("--height", type=int, default=d.camera_height, help="Camera capture height")
    parser.add_argument("--proc_width", type=int, default=d.process_input_width, help="Downscaled width for processing (lower=faster)")
    parser.add_argument("--active_area", type=float, default=d.active_area_ratio, help="Active box size fraction (0.0-1.0); affects clutch in Draw Mode")
    parser.add_argument("--mediapipe_complexity", type=int, choices=[0, 1], default=d.mediapipe_complexity, help="MediaPipe model complexity: 0 fast, 1 accurate")
    parser.add_argument("--gesture_every", type=int, default=d.gesture_every, help="Run gesture detection every N frames (0=every frame)")
    parser.add_argument("--list_cameras", action="store_true", help="List available camera indices and exit")
    parser.add_argument("--probe", type=int, default=6, help="When listing cameras, probe indices [0..N-1]")

    # Filtering & speed
    parser.add_argument("--min_cutoff", type=float, default=d.min_cutoff, help="One Euro min cutoff (lower=smoother, more lag)")
    parser.add_argument("--beta", type=float, default=d.beta, help="One Euro beta parameter (higher=snappier, less smooth)")
    parser.add_argument("--d_cutoff", type=float, default=d.d_cutoff, help="One Euro derivative cutoff")
    parser.add_argument("--max_speed", type=float, default=d.max_speed_px_per_s, help="Max cursor speed px/s (spike gate)")

    # Gestures core
    parser.add_argument("--pinch_on", type=float, default=d.pinch3d_on, help="3D pinch ON threshold")
    parser.add_argument("--pinch_off", type=float, default=d.pinch3d_off, help="3D pinch OFF threshold")
    parser.add_argument("--hold_delay", type=float, default=d.hold_delay_s, help="Hold activation delay seconds")
    parser.add_argument("--tap_max", type=float, default=d.tap_max_s, help="Tap max duration seconds")
    parser.add_argument("--double_window", type=float, default=d.double_tap_window_s, help="Double-tap window seconds")
    parser.add_argument("--stability_intensity", type=float, default=d.pinch_stability_intensity_max, help="Max 3D motion intensity to accept tap/press")
    parser.add_argument("--hand_conf_min", type=float, default=d.hand_confidence_min, help="Minimum hand confidence to accept gestures")

    # Mode toggle (3D motion + triple-tap)
    parser.add_argument("--toggle_intensity", type=float, default=d.toggle_intensity_min, help="3D motion intensity threshold to toggle draw mode")
    parser.add_argument("--x_weight", type=float, default=d.toggle_x_weight, help="3D intensity weight on X for mode toggle")
    parser.add_argument("--y_weight", type=float, default=d.toggle_y_weight, help="3D intensity weight on Y for mode toggle")
    parser.add_argument("--z_weight", type=float, default=d.toggle_z_weight, help="3D intensity weight on Z for mode toggle")
    parser.add_argument("--triple_window", type=float, default=d.triple_tap_window_s, help="Time window for triple open-hand taps")
    parser.add_argument("--open_tap_intensity", type=float, default=d.open_tap_intensity, help="Intensity threshold for an open-hand tap")
    parser.add_argument("--open_tap_min_gap", type=float, default=d.open_tap_min_interval_s, help="Minimum gap between open-hand taps")

    # Backup gestures
    parser.add_argument("--index_click_window", type=float, default=d.index_click_window_s, help="Index rapid motion window (s)")
    parser.add_argument("--index_click_amp", type=float, default=d.index_click_min_amp_frac, help="Index rapid motion min amplitude (fraction of frame height)")
    parser.add_argument("--middle_click_window", type=float, default=d.middle_click_window_s, help="Middle rapid motion window (s)")
    parser.add_argument("--middle_click_amp", type=float, default=d.middle_click_min_amp_frac, help="Middle rapid motion min amplitude (fraction of frame height)")
    parser.add_argument("--click_cooldown", type=float, default=d.click_cooldown_s, help="Cooldown between synthetic clicks (s)")
    parser.add_argument("--backup_pinch_tap_max", type=float, default=d.backup_pinch_tap_max_s, help="Super-quick pinch-release counts as click if <= this (s)")
    parser.add_argument("--im_on", type=float, default=d.im_touch_on, help="Index+Middle touch ON threshold (3D normalized)")
    parser.add_argument("--im_off", type=float, default=d.im_touch_off, help="Index+Middle touch OFF threshold (3D normalized)")

    # Draw mode params
    parser.add_argument("--draw_gain", type=float, default=d.draw_gain, help="Gain for relative motion in Draw Mode (higher=faster)")

    # Debug
    dbg = parser.add_mutually_exclusive_group()
    dbg.add_argument("--draw_debug", action="store_true", help="Show debug overlay and landmarks")
    dbg.add_argument("--no_debug", action="store_true", help="Disable debug overlay")


def parse_args() -> argparse.Namespace:
    d = Defaults()
    parser = argparse.ArgumentParser(description="PhantomTouch hand tracking controller")
    add_args(parser, d)
    args = parser.parse_args()
    return args


