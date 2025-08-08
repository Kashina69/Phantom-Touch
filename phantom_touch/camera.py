"""
camera.py
=========
Camera opening helpers and a frame-grabber thread that always keeps the latest
frame in memory to reduce capture latency.
"""

from __future__ import annotations

import time
import threading
import cv2


def open_camera(cam_index: int, width: int, height: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0]
    for be in backends:
        cap = cv2.VideoCapture(cam_index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 60)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
    raise RuntimeError("Unable to open camera with MSMF/DSHOW")


class FrameGrabber:
    """Background thread that reads frames as fast as possible.

    latest: most recent BGR frame (h, w, 3), horizontally flipped to mirror.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.latest = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self, join_timeout: float = 0.5) -> None:
        self._running = False
        try:
            self._thread.join(timeout=join_timeout)
        except Exception:
            pass

    def read_latest(self):
        with self._lock:
            return None if self.latest is None else self.latest.copy()

    def _loop(self) -> None:
        while self._running:
            ret, f = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            f = cv2.flip(f, 1)
            with self._lock:
                self.latest = f


