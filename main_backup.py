import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import threading
import argparse
import sys

"""
Low-latency hand tracking with two modes:
- Normal mode: absolute mapping of index finger to cursor with low-latency smoothing
- Drawing mode: relative (clutch) mapping with adjustable gain; moving finger out of active area acts like lifting a mouse

Robust pinch/hold/double-click with hysteresis and debouncing.
Toggle drawing mode by performing an open-hand quick down-up "click" motion.
"""

VIS_DEBUG = True  # draw landmarks and metrics overlay

# Camera & performance settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_INPUT_WIDTH = 320  # downscaled width for Mediapipe processing
ACTIVE_AREA_RATIO = 0.85
USE_LANDMARK_DRAW = False  # disable to reduce overhead

# Cursor smoothing (One Euro filter, low latency)
MIN_CUTOFF = 1.7
BETA = 0.3
D_CUTOFF = 1.0

# Jitter/velocity gating
MAX_SPEED_PX_PER_S = 5000.0
EDGE_DEADBAND_PX = 2

# Pinch thresholds and timings (normalized by hand scale)
PINCH3D_ON = 0.055
PINCH3D_OFF = 0.075
HOLD_DELAY_S = 0.18
TAP_MAX_S = 0.14
DOUBLE_TAP_WINDOW_S = 0.30
PINCH_STABILITY_INTENSITY_MAX = 0.12  # suppress taps while moving fast

# Drawing mode relative movement gain
DRAW_GAIN = 1.25  # multiplier for relative motion

# Mode toggle via 3D motion intensity (open hand + strong motion)
TOGGLE_WINDOW_S = 0.6
TOGGLE_INTENSITY_MIN = 0.22
TOGGLE_X_WEIGHT = 0.3
TOGGLE_Y_WEIGHT = 1.0
TOGGLE_Z_WEIGHT = 1.1
HAND_CONFIDENCE_MIN = 0.5

# Additional gesture thresholds
# Index/Middle rapid motion click windows (seconds) and amplitude (fraction of frame height)
INDEX_CLICK_WINDOW_S = 0.25     # higher = consider a longer time for the up/down motion
INDEX_CLICK_MIN_AMP_FRAC = 0.06 # higher = require bigger up/down travel → fewer false clicks
MIDDLE_CLICK_WINDOW_S = 0.25
MIDDLE_CLICK_MIN_AMP_FRAC = 0.06
CLICK_COOLDOWN_S = 0.25         # minimum gap between synthetic clicks

# Backup pinch tap even if moving fast
BACKUP_PINCH_TAP_MAX_S = 0.08   # very quick pinch-release always counts as left click

# Index-Middle touch hold thresholds (normalized 3D by hand scale)
IM_TOUCH_ON = 0.060
IM_TOUCH_OFF = 0.090

# Triple tap mode toggle (open hand taps)
TRIPLE_TAP_WINDOW_S = 0.8       # 3 taps must occur within this window
OPEN_TAP_INTENSITY = 0.18       # intensity threshold to count an open-hand tap
OPEN_TAP_MIN_INTERVAL_S = 0.12  # min gap between open-hand taps to de-bounce

# Init
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # fastest
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

def open_camera(cam_index: int = 0):
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0]
    for be in backends:
        cap = cv2.VideoCapture(cam_index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 60)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
    raise RuntimeError("Unable to open camera with MSMF/DSHOW")

def parse_args():
    parser = argparse.ArgumentParser(description="PhantomTouch hand tracking controller")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0,1,2,...) to open")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH, help="Camera capture width")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT, help="Camera capture height")
    parser.add_argument("--proc_width", type=int, default=PROCESS_INPUT_WIDTH, help="Downscaled width for processing")
    parser.add_argument("--draw_debug", action="store_true", help="Show debug overlay and landmarks")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug overlay")
    parser.add_argument("--toggle_intensity", type=float, default=TOGGLE_INTENSITY_MIN, help="3D motion intensity threshold to toggle draw mode")
    parser.add_argument("--pinch_on", type=float, default=PINCH3D_ON, help="3D pinch ON threshold")
    parser.add_argument("--pinch_off", type=float, default=PINCH3D_OFF, help="3D pinch OFF threshold")
    parser.add_argument("--hold_delay", type=float, default=HOLD_DELAY_S, help="Hold activation delay seconds")
    parser.add_argument("--tap_max", type=float, default=TAP_MAX_S, help="Tap max duration seconds")
    parser.add_argument("--double_window", type=float, default=DOUBLE_TAP_WINDOW_S, help="Double-tap window seconds")
    parser.add_argument("--max_speed", type=float, default=MAX_SPEED_PX_PER_S, help="Max cursor speed px/s")
    parser.add_argument("--beta", type=float, default=BETA, help="One Euro beta parameter")
    parser.add_argument("--min_cutoff", type=float, default=MIN_CUTOFF, help="One Euro min cutoff")
    parser.add_argument("--d_cutoff", type=float, default=D_CUTOFF, help="One Euro derivative cutoff")
    parser.add_argument("--mediapipe_complexity", type=int, choices=[0,1], default=0, help="MediaPipe model complexity (0 fast, 1 accurate)")
    parser.add_argument("--skip_detect", type=int, default=0, help="DEPRECATED. Use --gesture_every. Run detection every N frames (0=every frame)")
    parser.add_argument("--gesture_every", type=int, default=0, help="Run gesture detection every N frames (0=every frame)")
    parser.add_argument("--active_area", type=float, default=ACTIVE_AREA_RATIO, help="Active box size fraction (0.0-1.0). Lower = smaller box. Affects clutch in Draw Mode.")
    parser.add_argument("--draw_gain", type=float, default=DRAW_GAIN, help="Gain for relative motion in Draw Mode. Higher = faster cursor movement.")
    parser.add_argument("--x_weight", type=float, default=TOGGLE_X_WEIGHT, help="3D intensity weight on X for mode toggle. Impacts sensitivity.")
    parser.add_argument("--y_weight", type=float, default=TOGGLE_Y_WEIGHT, help="3D intensity weight on Y for mode toggle. Impacts sensitivity.")
    parser.add_argument("--z_weight", type=float, default=TOGGLE_Z_WEIGHT, help="3D intensity weight on Z for mode toggle. Impacts sensitivity.")
    parser.add_argument("--hand_conf_min", type=float, default=HAND_CONFIDENCE_MIN, help="Minimum hand confidence to accept gestures. Higher = fewer false positives, may ignore some frames.")
    parser.add_argument("--list_cameras", action="store_true", help="List available camera indices and exit")
    parser.add_argument("--probe", type=int, default=6, help="When listing cameras, probe indices [0..N-1]")
    return parser.parse_args()

args = parse_args()

# apply args to globals
CAMERA_WIDTH = args.width
CAMERA_HEIGHT = args.height
PROCESS_INPUT_WIDTH = args.proc_width
TOGGLE_INTENSITY_MIN = args.toggle_intensity
VIS_DEBUG = args.draw_debug or (VIS_DEBUG and not args.no_debug)
PINCH3D_ON = args.pinch_on
PINCH3D_OFF = args.pinch_off
HOLD_DELAY_S = args.hold_delay
TAP_MAX_S = args.tap_max
DOUBLE_TAP_WINDOW_S = args.double_window
MAX_SPEED_PX_PER_S = args.max_speed
BETA = args.beta
MIN_CUTOFF = args.min_cutoff
D_CUTOFF = args.d_cutoff
ACTIVE_AREA_RATIO = args.active_area
DRAW_GAIN = args.draw_gain
TOGGLE_X_WEIGHT = args.x_weight
TOGGLE_Y_WEIGHT = args.y_weight
TOGGLE_Z_WEIGHT = args.z_weight
HAND_CONFIDENCE_MIN = args.hand_conf_min

def probe_cameras(max_index: int):
    found = []
    for i in range(max_index):
        ok = False
        for be in [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0]:
            tmp = cv2.VideoCapture(i, be)
            if tmp.isOpened():
                ok = True
                tmp.release()
                break
        print(f"Camera {i}: {'OK' if ok else 'Not available'}")
        if ok:
            found.append(i)
    if not found:
        print("No cameras detected up to index", max_index - 1)
    return found

if args.list_cameras:
    probe_cameras(args.probe)
    sys.exit(0)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=args.mediapipe_complexity,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = open_camera(args.camera)
print(f"Using camera index {args.camera} @ {CAMERA_WIDTH}x{CAMERA_HEIGHT}, proc_width={PROCESS_INPUT_WIDTH}, complexity={args.mediapipe_complexity}")
cv2.setUseOptimized(True)
cv2.setNumThreads(0)

screen_w, screen_h = pyautogui.size()

# One Euro filter implementation (low latency smoothing)

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

def smoothing_factor(t_e: float, cutoff: float) -> float:
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
        # estimate derivative
        dx = (value - self.x_hat.last) / dt
        alpha_d = smoothing_factor(dt, self.d_cutoff)
        edx = self.dx_hat.apply(dx, alpha=alpha_d)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = smoothing_factor(dt, cutoff)
        return self.x_hat.apply(value, alpha=alpha)

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def within_active_area(pt, w, h, margin_x, margin_y):
    return margin_x < pt[0] < w - margin_x and margin_y < pt[1] < h - margin_y

def is_open_hand(lm_norm) -> bool:
    # Simple heuristic: index and middle extended, not pinching
    # y increases downward; tip above pip -> extended
    try:
        idx_ext = lm_norm[8].y < lm_norm[6].y
        mid_ext = lm_norm[12].y < lm_norm[10].y
        ring_ext = lm_norm[16].y < lm_norm[14].y
        return (idx_ext and mid_ext) or (idx_ext and ring_ext)
    except Exception:
        return False

def normalized_pinch(lm_px):
    # Normalize thumb-index distance by hand scale (wrist to index_mcp)
    wrist = (lm_px[0].x, lm_px[0].y)
    index_mcp = (lm_px[5].x, lm_px[5].y)
    index_tip = (lm_px[8].x, lm_px[8].y)
    thumb_tip = (lm_px[4].x, lm_px[4].y)
    scale = max(1e-6, distance(wrist, index_mcp))
    return distance(index_tip, thumb_tip) / scale

def normalized_pinch_3d(lm_norm):
    def dist3(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
    wrist = lm_norm[0]
    index_mcp = lm_norm[5]
    index_tip = lm_norm[8]
    thumb_tip = lm_norm[4]
    hand_scale = max(1e-6, dist3(wrist, index_mcp))
    return dist3(index_tip, thumb_tip) / hand_scale

def is_finger_extended(lm, tip_idx, pip_idx) -> bool:
    return lm[tip_idx].y < lm[pip_idx].y

def norm_dist3(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


# Frame grabber thread to minimize capture latency
latest_frame = None
frame_lock = threading.Lock()
grabber_running = True

def frame_grabber():
    global latest_frame
    while grabber_running:
        ret, f = cap.read()
        if not ret:
            time.sleep(0.001)
            continue
        f = cv2.flip(f, 1)
        with frame_lock:
            latest_frame = f

grab_thread = threading.Thread(target=frame_grabber, daemon=True)
grab_thread.start()

last_action = None
last_tap_time = 0.0
pinch_on_time = 0.0
pinch_is_on = False
hold_active = False

drawing_mode = False
clutch_active = False
last_in_area_finger = None

filter_x = OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF)
filter_y = OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF)

toggle_state = "idle"
toggle_last_time = 0.0
recent_wrist_3d = []  # (t, x, y, z)
fps_last_t = time.time()
fps_alpha = 0.9
fps_est = 0.0
MAX_SPEED_PX_PER_S = 5000.0
EDGE_DEADBAND_PX = 2
last_screen_pos = None
move_last_t = time.time()
last_pinch_val_3d = None
last_motion_intensity = 0.0
last_hand_conf = 0.0

# Gesture history buffers
index_y_history = []   # (t, y)
middle_y_history = []  # (t, y)
open_tap_times = []    # times of detected open-hand taps
last_synthetic_click_time = 0.0
im_touch_on = False

try:
    while True:
        # Get the most recent frame
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.001)
            continue

        h, w, _ = frame.shape

        # Draw active area box
        margin_x = int(w * (1 - ACTIVE_AREA_RATIO) / 2)
        margin_y = int(h * (1 - ACTIVE_AREA_RATIO) / 2)
        cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), (0, 255, 0), 1)

        # Downscale for processing to reduce CPU latency
        if PROCESS_INPUT_WIDTH and PROCESS_INPUT_WIDTH < w:
            new_w = PROCESS_INPUT_WIDTH
            new_h = int(h * (PROCESS_INPUT_WIDTH / w))
            proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc = frame
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)

        # Optional frame skip for gesture logic to save CPU
        use_this_frame = True
        cadence = args.gesture_every if args.gesture_every > 0 else args.skip_detect
        if cadence > 0:
            frame_idx = int(time.time() * 1000)  # ms timestamp as a cheap counter
            use_this_frame = (frame_idx % (cadence + 1) == 0)

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            lm_norm = hand_lm.landmark
            if USE_LANDMARK_DRAW or VIS_DEBUG:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            # approximate hand confidence if available
            try:
                handedness = result.multi_handedness[0].classification[0]
                hand_conf = handedness.score
            except Exception:
                hand_conf = 1.0

            # Pixel landmarks for convenience
            lm_px = [
                type("P", (), {"x": int(l.x * w), "y": int(l.y * h)})
                for l in lm_norm
            ]

            index_tip = (lm_px[8].x, lm_px[8].y)
            # Mode toggle detection (open hand + strong 3D motion intensity or triple open-hand taps)
            now = time.time()
            wrist_norm = lm_norm[0]
            recent_wrist_3d.append((now, wrist_norm.x, wrist_norm.y, wrist_norm.z))
            while recent_wrist_3d and now - recent_wrist_3d[0][0] > TOGGLE_WINDOW_S:
                recent_wrist_3d.pop(0)

            def calc_intensity(samples):
                if not samples:
                    return 0.0
                xs = [s[1] for s in samples]
                ys = [s[2] for s in samples]
                zs = [s[3] for s in samples]
                dx = (max(xs) - min(xs)) * TOGGLE_X_WEIGHT
                dy = (max(ys) - min(ys)) * TOGGLE_Y_WEIGHT
                dz = (max(zs) - min(zs)) * TOGGLE_Z_WEIGHT
                return math.sqrt(dx * dx + dy * dy + dz * dz)

            motion_intensity = calc_intensity(recent_wrist_3d)
            if use_this_frame:
                if hand_conf >= HAND_CONFIDENCE_MIN and is_open_hand(lm_norm):
                    # 1) Open-hand strong motion toggles
                    if motion_intensity >= TOGGLE_INTENSITY_MIN:
                        if toggle_state == "idle":
                            toggle_state = "armed"
                            toggle_last_time = now
                        elif toggle_state == "armed" and (now - toggle_last_time) > 0.12:
                            drawing_mode = not drawing_mode
                            clutch_active = False
                            last_in_area_finger = None
                            toggle_state = "cooldown"
                            toggle_last_time = now
                            print(f"Drawing mode: {'ON' if drawing_mode else 'OFF'}")
                    # 2) Triple open-hand taps to toggle
                    amp = motion_intensity
                    if amp >= OPEN_TAP_INTENSITY and (not open_tap_times or (now - open_tap_times[-1]) >= OPEN_TAP_MIN_INTERVAL_S):
                        open_tap_times.append(now)
                        # keep only window
                        open_tap_times[:] = [t for t in open_tap_times if now - t <= TRIPLE_TAP_WINDOW_S]
                        if len(open_tap_times) >= 3:
                            drawing_mode = not drawing_mode
                            clutch_active = False
                            last_in_area_finger = None
                            open_tap_times.clear()
                            print(f"Drawing mode (triple tap): {'ON' if drawing_mode else 'OFF'}")
                if toggle_state == "cooldown" and (now - toggle_last_time) > 0.6:
                    toggle_state = "idle"
                if not is_open_hand(lm_norm) and toggle_state == "armed":
                    toggle_state = "idle"

            # Compute normalized pinch metric
            pinch_val_3d = normalized_pinch_3d(lm_norm)
            if use_this_frame:
                # hysteresis
                if pinch_is_on:
                    if pinch_val_3d > PINCH3D_OFF:
                        pinch_is_on = False
                        pinch_off_time = now
                        # tap detection
                        if (pinch_off_time - pinch_on_time) <= TAP_MAX_S and motion_intensity <= PINCH_STABILITY_INTENSITY_MAX:
                            # quick tap
                            if (pinch_off_time - last_tap_time) <= DOUBLE_TAP_WINDOW_S:
                                pyautogui.doubleClick(_pause=False)
                                last_action = "DOUBLE_CLICK"
                                last_tap_time = 0.0
                            else:
                                last_tap_time = pinch_off_time
                        if hold_active:
                            pyautogui.mouseUp(_pause=False)
                            hold_active = False
                            last_action = "POINTER"
                else:
                    if pinch_val_3d < PINCH3D_ON and motion_intensity <= PINCH_STABILITY_INTENSITY_MAX:
                        pinch_is_on = True
                        pinch_on_time = now

                # hold activation
                if pinch_is_on and not hold_active and (now - pinch_on_time) >= HOLD_DELAY_S:
                    pyautogui.mouseDown(_pause=False)
                    hold_active = True
                    last_action = "HOLD"

            # Update last-known metrics for overlay
            last_pinch_val_3d = pinch_val_3d
            last_motion_intensity = motion_intensity
            last_hand_conf = hand_conf

            # Backup gesture: synthetic left click via rapid index up/down OR super-quick pinch
            index_y_history.append((now, lm_px[8].y))
            cutoff_t = now - INDEX_CLICK_WINDOW_S
            while index_y_history and index_y_history[0][0] < cutoff_t:
                index_y_history.pop(0)
            if index_y_history:
                ys = [v for (_, v) in index_y_history]
                if (max(ys) - min(ys)) >= (INDEX_CLICK_MIN_AMP_FRAC * h):
                    if now - last_synthetic_click_time >= CLICK_COOLDOWN_S:
                        pyautogui.click(_pause=False)
                        last_synthetic_click_time = now

            # Backup: super-quick pinch-release always counts as left click
            if 'pinch_off_time' in locals():
                if (pinch_off_time - pinch_on_time) <= BACKUP_PINCH_TAP_MAX_S:
                    if now - last_synthetic_click_time >= CLICK_COOLDOWN_S:
                        pyautogui.click(_pause=False)
                        last_synthetic_click_time = now

            # Active area check for clutch (drawing mode) and alternative gestures
            inside = within_active_area(index_tip, w, h, margin_x, margin_y)
            if drawing_mode:
                if not inside:
                    clutch_active = True
                else:
                    if clutch_active or last_in_area_finger is None:
                        last_in_area_finger = index_tip
                        clutch_active = False
                    # relative movement
                    dx = (index_tip[0] - last_in_area_finger[0]) * DRAW_GAIN
                    dy = (index_tip[1] - last_in_area_finger[1]) * DRAW_GAIN
                    if dx != 0 or dy != 0:
                        cx, cy = pyautogui.position()
                        nx = max(0, min(screen_w - 1, int(cx + dx)))
                        ny = max(0, min(screen_h - 1, int(cy + dy)))
                        pyautogui.moveTo(nx, ny, duration=0, _pause=False)
                        last_in_area_finger = index_tip
            else:
                # Normal mode: absolute mapping with low-latency smoothing
                # Map to screen
                abs_x = np.interp(index_tip[0], (margin_x, w - margin_x), (0, screen_w))
                abs_y = np.interp(index_tip[1], (margin_y, h - margin_y), (0, screen_h))
                tnow = time.time()
                fx = filter_x.filter(float(abs_x), tnow)
                fy = filter_y.filter(float(abs_y), tnow)
                # velocity gate to reduce jitter and spikes
                global_pos = pyautogui.position()
                if last_screen_pos is None:
                    last_screen_pos = (global_pos[0], global_pos[1])
                dt_move = max(1e-3, tnow - move_last_t)
                max_delta = MAX_SPEED_PX_PER_S * dt_move + 5
                if abs(fx - last_screen_pos[0]) <= max_delta and abs(fy - last_screen_pos[1]) <= max_delta:
                    tx = int(max(EDGE_DEADBAND_PX, min(screen_w - EDGE_DEADBAND_PX, fx)))
                    ty = int(max(EDGE_DEADBAND_PX, min(screen_h - EDGE_DEADBAND_PX, fy)))
                    pyautogui.moveTo(tx, ty, duration=0, _pause=False)
                    last_screen_pos = (fx, fy)
                    move_last_t = tnow

            # Right click via middle finger rapid motion when extended
            middle_y_history.append((now, lm_px[12].y))
            cutoff_m = now - MIDDLE_CLICK_WINDOW_S
            while middle_y_history and middle_y_history[0][0] < cutoff_m:
                middle_y_history.pop(0)
            middle_extended = is_finger_extended(lm_norm, 12, 10)
            if middle_extended and middle_y_history:
                mys = [v for (_, v) in middle_y_history]
                if (max(mys) - min(mys)) >= (MIDDLE_CLICK_MIN_AMP_FRAC * h):
                    if now - last_synthetic_click_time >= CLICK_COOLDOWN_S:
                        pyautogui.click(button='right', _pause=False)
                        last_synthetic_click_time = now

            # Index+Middle touch = hold/drag mode (like iron-man gesture)
            im_dist = norm_dist3(lm_norm[8], lm_norm[12]) / max(1e-6, norm_dist3(lm_norm[0], lm_norm[5]))
            if im_touch_on:
                if im_dist > IM_TOUCH_OFF:
                    im_touch_on = False
                    if hold_active:
                        pyautogui.mouseUp(_pause=False)
                        hold_active = False
                        last_action = "POINTER"
            else:
                if im_dist < IM_TOUCH_ON:
                    im_touch_on = True
                    pyautogui.mouseDown(_pause=False)
                    hold_active = True
                    last_action = "HOLD"

        else:
            # No hand detected → release mouse, reset
            if hold_active:
                pyautogui.mouseUp(_pause=False)
                hold_active = False
            pinch_is_on = False
            last_action = None
            last_in_area_finger = None

        # FPS and debug overlay
        now2 = time.time()
        dt = max(1e-6, now2 - fps_last_t)
        fps_last_t = now2
        inst_fps = 1.0 / dt
        fps_est = fps_alpha * fps_est + (1 - fps_alpha) * inst_fps if fps_est > 0 else inst_fps

        if VIS_DEBUG:
            if result.multi_hand_landmarks:
                cv2.circle(frame, (lm_px[8].x, lm_px[8].y), 6, (0, 128, 255), -1)
                cv2.circle(frame, (lm_px[4].x, lm_px[4].y), 6, (255, 128, 0), -1)
            pv = last_pinch_val_3d if last_pinch_val_3d is not None else 0.0
            mi = last_motion_intensity if last_motion_intensity is not None else 0.0
            hc = last_hand_conf if last_hand_conf is not None else 0.0
            info_lines = [
                f"Mode: {'DRAW' if drawing_mode else 'NORMAL'}",
                f"FPS: {int(fps_est)}",
                f"Pinch3D: {pv:.3f}",
                f"Intensity3D: {mi:.3f}",
                f"Hold: {hold_active}",
                f"Conf: {hc:.2f}",
            ]
            y0 = 24
            for i, txt in enumerate(info_lines):
                cv2.putText(frame, txt, (10, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            mode_text = f"Mode: {'DRAW' if drawing_mode else 'NORMAL'}"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("PhantomTouch", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    grabber_running = False
    try:
        grab_thread.join(timeout=0.5)
    except Exception:
        pass
cap.release()
cv2.destroyAllWindows()
