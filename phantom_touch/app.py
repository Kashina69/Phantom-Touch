"""
app.py
======
Main application orchestrator: parses config, opens camera, runs the frame loop,
feeds landmarks into gesture logic, updates cursor, and draws optional overlays.

High-level flow:
1) Parse CLI args into a config namespace
2) Open camera with preferred backend and start a FrameGrabber thread
3) For each latest frame:
   - Draw active-area box
   - Downscale for MediaPipe processing and run hand tracking
   - Update mode toggle (3D intensity or triple-tap) when hand is open
   - Update pinch/hold/double-click state; fire clicks via pyautogui
   - Backup gestures (rapid finger motion) and iron-man index+middle hold
   - Map cursor: absolute in Normal mode; relative (clutch) in Draw Mode
   - Draw overlay (mode, FPS, pinch, intensity, confidence) if enabled
4) Gracefully close on 'q' key
"""

from __future__ import annotations

import time
import cv2
import numpy as np
import pyautogui
import mediapipe as mp

from .config import parse_args, Defaults
from .camera import open_camera, FrameGrabber
from .filters import OneEuroFilter
from .gestures import GestureState, update_mode_toggle, update_pinch_hold_clicks, index_middle_hold, normalized_pinch_3d
from .mapping import CursorMapper
from .utils import is_open_hand, is_finger_extended, norm_dist3
from .visual import draw_overlay


def main() -> None:
    args = parse_args()
    d = Defaults()

    # apply args → config values
    cam_w = args.width
    cam_h = args.height
    proc_w = args.proc_width
    active_area = args.active_area
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    if args.list_cameras:
        import sys
        for i in range(args.probe):
            ok = False
            for be in [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0]:
                tmp = cv2.VideoCapture(i, be)
                if tmp.isOpened():
                    ok = True
                    tmp.release()
                    break
            print(f"Camera {i}: {'OK' if ok else 'Not available'}")
        return

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=args.mediapipe_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    # Camera & grabber
    cap = open_camera(args.camera, cam_w, cam_h)
    grabber = FrameGrabber(cap)
    grabber.start()
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    # Screen & cursor mapping
    screen_w, screen_h = pyautogui.size()
    mapper = CursorMapper(screen_w, screen_h, args.min_cutoff, args.beta, args.d_cutoff, args.max_speed, d.edge_deadband_px)

    # State
    state = GestureState()
    drawing_mode = False
    clutch_active = False
    last_in_area_finger = None
    fps_last_t = time.time()
    fps_est = 0.0
    fps_alpha = 0.9

    # Adapter: map CLI arg names to gesture config attribute names expected by gesture logic
    class GestureCfg:
        pass

    gcfg = GestureCfg()
    # Mode toggle + triple tap
    gcfg.toggle_window_s = getattr(args, 'toggle_window', d.toggle_window_s)
    gcfg.toggle_x_weight = getattr(args, 'x_weight', d.toggle_x_weight)
    gcfg.toggle_y_weight = getattr(args, 'y_weight', d.toggle_y_weight)
    gcfg.toggle_z_weight = getattr(args, 'z_weight', d.toggle_z_weight)
    gcfg.toggle_intensity_min = getattr(args, 'toggle_intensity', d.toggle_intensity_min)
    gcfg.triple_tap_window_s = getattr(args, 'triple_window', d.triple_tap_window_s)
    gcfg.open_tap_intensity = getattr(args, 'open_tap_intensity', d.open_tap_intensity)
    gcfg.open_tap_min_interval_s = getattr(args, 'open_tap_min_gap', d.open_tap_min_interval_s)
    gcfg.hand_conf_min = getattr(args, 'hand_conf_min', d.hand_confidence_min)
    # Pinch/hold/double-click
    gcfg.pinch3d_on = getattr(args, 'pinch_on', d.pinch3d_on)
    gcfg.pinch3d_off = getattr(args, 'pinch_off', d.pinch3d_off)
    gcfg.hold_delay_s = getattr(args, 'hold_delay', d.hold_delay_s)
    gcfg.tap_max_s = getattr(args, 'tap_max', d.tap_max_s)
    gcfg.double_tap_window_s = getattr(args, 'double_window', d.double_tap_window_s)
    gcfg.pinch_stability_intensity_max = getattr(args, 'stability_intensity', d.pinch_stability_intensity_max)
    # Index+Middle hold
    gcfg.im_touch_on = getattr(args, 'im_on', d.im_touch_on)
    gcfg.im_touch_off = getattr(args, 'im_off', d.im_touch_off)

    try:
        while True:
            frame = grabber.read_latest()
            if frame is None:
                time.sleep(0.001)
                continue
            h, w, _ = frame.shape

            # Active area box
            margin_x = int(w * (1 - active_area) / 2)
            margin_y = int(h * (1 - active_area) / 2)
            cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), (0, 255, 0), 1)

            # Downscale for processing
            proc = frame
            if proc_w and proc_w < w:
                new_w = proc_w
                new_h = int(h * (proc_w / w))
                proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)

            # Gesture cadence (optional)
            use_this_frame = True
            if args.gesture_every > 0:
                frame_idx = int(time.time() * 1000)
                use_this_frame = (frame_idx % (args.gesture_every + 1) == 0)

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                lm_norm = hand_lm.landmark
                if args.draw_debug:
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                # Pixel coords for overlay and mapping
                lm_px = [type("P", (), {"x": int(l.x * w), "y": int(l.y * h)}) for l in lm_norm]
                index_tip = (lm_px[8].x, lm_px[8].y)
                now = time.time()

                # Hand confidence (approx from handedness)
                try:
                    hand_conf = result.multi_handedness[0].classification[0].score
                except Exception:
                    hand_conf = 1.0

                # Mode toggle
                if use_this_frame and update_mode_toggle(state, now, lm_norm, hand_conf, gcfg):
                    drawing_mode = not drawing_mode
                    clutch_active = False
                    last_in_area_finger = None
                    print(f"Drawing mode: {'ON' if drawing_mode else 'OFF'}")

                # Core pinch/hold/double-click
                if use_this_frame:
                    action, click_type = update_pinch_hold_clicks(state, now, lm_norm, gcfg)
                    if click_type == "DOUBLE":
                        pyautogui.doubleClick(_pause=False)
                    if action == "HOLD" and not state.hold_active:
                        pyautogui.mouseDown(_pause=False)
                        state.hold_active = True
                    elif action == "POINTER" and state.hold_active:
                        pyautogui.mouseUp(_pause=False)
                        state.hold_active = False

                # Iron-man index+middle hold (overrides)
                im_action = index_middle_hold(state, now, lm_norm, gcfg)
                if im_action == "HOLD" and not state.hold_active:
                    pyautogui.mouseDown(_pause=False)
                    state.hold_active = True
                elif im_action == "POINTER" and state.hold_active:
                    pyautogui.mouseUp(_pause=False)
                    state.hold_active = False

                # Mapping
                inside = (margin_x < index_tip[0] < w - margin_x and margin_y < index_tip[1] < h - margin_y)
                if drawing_mode:
                    if not inside:
                        clutch_active = True
                    else:
                        if clutch_active or last_in_area_finger is None:
                            last_in_area_finger = index_tip
                            clutch_active = False
                        dx = (index_tip[0] - last_in_area_finger[0]) * args.draw_gain
                        dy = (index_tip[1] - last_in_area_finger[1]) * args.draw_gain
                        if dx != 0 or dy != 0:
                            mapper.move_relative(dx, dy)
                            last_in_area_finger = index_tip
                else:
                    mapper.move_absolute(index_tip[0], index_tip[1], margin_x, margin_y, w, h)

                # Backup clicks
                if use_this_frame:
                    # Rapid index up/down → left click
                    state.index_y_history.append((now, lm_px[8].y))
                    cut = now - args.index_click_window
                    while state.index_y_history and state.index_y_history[0][0] < cut:
                        state.index_y_history.pop(0)
                    if state.index_y_history:
                        ys = [y for (_, y) in state.index_y_history]
                        if (max(ys) - min(ys)) >= (args.index_click_amp * h) and (now - state.last_synth_click_time) >= args.click_cooldown:
                            pyautogui.click(_pause=False)
                            state.last_synth_click_time = now

                    # Middle extended + rapid motion → right click
                    state.middle_y_history.append((now, lm_px[12].y))
                    cutm = now - args.middle_click_window
                    while state.middle_y_history and state.middle_y_history[0][0] < cutm:
                        state.middle_y_history.pop(0)
                    middle_extended = is_finger_extended(lm_norm, 12, 10)
                    if middle_extended and state.middle_y_history:
                        mys = [y for (_, y) in state.middle_y_history]
                        if (max(mys) - min(mys)) >= (args.middle_click_amp * h) and (now - state.last_synth_click_time) >= args.click_cooldown:
                            pyautogui.click(button='right', _pause=False)
                            state.last_synth_click_time = now

                    # Backup: super-quick pinch release → click
                    pv = normalized_pinch_3d(lm_norm)
                    if state.pinch_is_on:
                        # set on-time already in update_pinch...; here we just look for quick off
                        pass
                    else:
                        # if we just turned off very fast this frame, conservative fallback: handled in main pinch logic
                        pass

            else:
                if state.hold_active:
                    pyautogui.mouseUp(_pause=False)
                    state.hold_active = False

            # Overlay
            now2 = time.time()
            dt = max(1e-6, now2 - fps_last_t)
            fps_last_t = now2
            inst = 1.0 / dt
            fps_est = fps_alpha * fps_est + (1 - fps_alpha) * inst if fps_est > 0 else inst
            if args.draw_debug:
                info = [
                    f"Mode: {'DRAW' if drawing_mode else 'NORMAL'}",
                    f"FPS: {int(fps_est)}",
                    f"Pinch3D: {state.last_pinch_val_3d or 0.0:.3f}",
                    f"Intensity3D: {state.last_motion_intensity:.3f}",
                    f"Hold: {state.hold_active}",
                ]
                draw_overlay(frame, info, lm_px if result.multi_hand_landmarks else None)

            cv2.imshow("PhantomTouch", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        pass
    finally:
        grabber.stop()
        # Release resources; catch BaseException to absorb Ctrl+C during cleanup
        try:
            hands.close()
        except BaseException:
            pass
        try:
            cap.release()
        except BaseException:
            pass
        try:
            cv2.destroyAllWindows()
        except BaseException:
            pass


