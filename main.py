# ============================================================
# TABLE OF CONTENTS
# ------------------------------------------------------------
# 1. DEBUG PRINTS
# 2. LOGGING
# 3. DYNAMIC PERFORMANCE TUNING
# 4. CAMERA RESCAN (HOT-PLUG SUPPORT)
# 5. CAMERA CAPTURE WORKER
# 6. FULLSCREEN OVERLAY
# 7. CAMERA WIDGET
# 8. GRID LAYOUT HELPERS
# 9. SYSTEM / PROCESS HELPERS
# 10. CAMERA DISCOVERY
# 11. CLEANUP + PROFILE SELECTION
# 12. MAIN ENTRYPOINT
# ============================================================
import atexit
import configparser
import glob
import logging
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot

# ============================================================
# DEBUG PRINTS (disabled by default)
# ------------------------------------------------------------
# Simple toggle for extra print logs without changing code.
# ============================================================
# DEBUG_PRINTS = True
DEBUG_PRINTS = False
# UI_FPS_LOGGING = True  # Enable for FPS diagnostics
UI_FPS_LOGGING = False


def dprint(*args, **kwargs):
    """Lightweight debug print wrapper."""
    if DEBUG_PRINTS:
        print(*args, **kwargs)


# ============================================================
# LOGGING + CONFIG
# ------------------------------------------------------------
# Logging is configured at runtime from config/env.
# ============================================================
LOG_LEVEL = "INFO"
LOG_FILE = "./logs/camera_dashboard.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
LOG_TO_STDOUT = True

CONFIG_PATH = os.environ.get("CAMERA_DASHBOARD_CONFIG", "./config.ini")
LOG_FILE_ENV = os.environ.get("CAMERA_DASHBOARD_LOG_FILE")

# ============================================================
# PERFORMANCE + RECOVERY TUNING
# ------------------------------------------------------------
# Controls FPS adaptation and stale-frame recovery behavior.
# ============================================================
DYNAMIC_FPS_ENABLED = True
PERF_CHECK_INTERVAL_MS = 2000
MIN_DYNAMIC_FPS = 5
MIN_DYNAMIC_UI_FPS = 10
UI_FPS_STEP = 2
CPU_LOAD_THRESHOLD = 0.75  # 75% avg load
CPU_TEMP_THRESHOLD_C = 70.0  # Celsius
STRESS_HOLD_COUNT = 2  # consecutive checks before reducing fps
RECOVER_HOLD_COUNT = 3  # consecutive checks before increasing fps

# Stale frame detection + bounded auto-restart policy.
STALE_FRAME_TIMEOUT_SEC = 1.5
RESTART_COOLDOWN_SEC = 5.0
MAX_RESTARTS_PER_WINDOW = 3
RESTART_WINDOW_SEC = 30.0

# ============================================================
# CAMERA RESCAN (HOT-PLUG SUPPORT)
# ------------------------------------------------------------
RESCAN_INTERVAL_MS = 15000  # 15 seconds (reduced CPU usage vs 5s)
FAILED_CAMERA_COOLDOWN_SEC = 30.0

# ============================================================
# APP SETTINGS
# ------------------------------------------------------------
CAMERA_SLOT_COUNT = 3
HEALTH_LOG_INTERVAL_SEC = 30.0
KILL_DEVICE_HOLDERS = True

PROFILE_CAPTURE_WIDTH = 640
PROFILE_CAPTURE_HEIGHT = 480
PROFILE_CAPTURE_FPS = 20
PROFILE_UI_FPS = 15  # Target UI refresh rate (capture runs at 20 FPS)

# GStreamer pipeline support (more efficient on Pi)
USE_GSTREAMER = True  # Try GStreamer first, fallback to V4L2

# Render overhead compensation (ms) - subtract from timer interval to hit target FPS
# Qt timer fires, then render takes ~2-3ms, so actual cycle = interval + render_time
# Example: 15 FPS (66.67ms cycle) → set interval to 66.67 - 3 = 63.67ms
RENDER_OVERHEAD_MS = 3


def _as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _as_int(value, default, min_value=None, max_value=None):
    try:
        if value is None:
            return default
        parsed = int(value)
    except Exception:
        return default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _as_float(value, default, min_value=None, max_value=None):
    try:
        if value is None:
            return default
        parsed = float(value)
    except Exception:
        return default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _load_config(path):
    parser = configparser.ConfigParser()
    if not path:
        return parser
    if os.path.exists(path):
        parser.read(path)
    return parser


def apply_config(parser):
    global LOG_LEVEL
    global LOG_FILE
    global LOG_MAX_BYTES
    global LOG_BACKUP_COUNT
    global LOG_TO_STDOUT

    global DYNAMIC_FPS_ENABLED
    global PERF_CHECK_INTERVAL_MS
    global MIN_DYNAMIC_FPS
    global MIN_DYNAMIC_UI_FPS
    global UI_FPS_STEP
    global CPU_LOAD_THRESHOLD
    global CPU_TEMP_THRESHOLD_C
    global STRESS_HOLD_COUNT
    global RECOVER_HOLD_COUNT
    global STALE_FRAME_TIMEOUT_SEC
    global RESTART_COOLDOWN_SEC
    global MAX_RESTARTS_PER_WINDOW
    global RESTART_WINDOW_SEC
    global RESCAN_INTERVAL_MS
    global FAILED_CAMERA_COOLDOWN_SEC
    global CAMERA_SLOT_COUNT
    global HEALTH_LOG_INTERVAL_SEC
    global KILL_DEVICE_HOLDERS
    global PROFILE_CAPTURE_WIDTH
    global PROFILE_CAPTURE_HEIGHT
    global PROFILE_CAPTURE_FPS
    global PROFILE_UI_FPS

    if parser.has_section("logging"):
        LOG_LEVEL = parser.get("logging", "level", fallback=LOG_LEVEL)
        LOG_FILE = parser.get("logging", "file", fallback=LOG_FILE)
        LOG_MAX_BYTES = _as_int(
            parser.get("logging", "max_bytes", fallback=LOG_MAX_BYTES),
            LOG_MAX_BYTES,
            min_value=1024,
        )
        LOG_BACKUP_COUNT = _as_int(
            parser.get("logging", "backup_count", fallback=LOG_BACKUP_COUNT),
            LOG_BACKUP_COUNT,
            min_value=1,
        )
        LOG_TO_STDOUT = _as_bool(
            parser.get("logging", "stdout", fallback=LOG_TO_STDOUT), LOG_TO_STDOUT
        )

    if parser.has_section("performance"):
        DYNAMIC_FPS_ENABLED = _as_bool(
            parser.get("performance", "dynamic_fps", fallback=DYNAMIC_FPS_ENABLED),
            DYNAMIC_FPS_ENABLED,
        )
        PERF_CHECK_INTERVAL_MS = _as_int(
            parser.get(
                "performance", "perf_check_interval_ms", fallback=PERF_CHECK_INTERVAL_MS
            ),
            PERF_CHECK_INTERVAL_MS,
            min_value=250,
        )
        MIN_DYNAMIC_FPS = _as_int(
            parser.get("performance", "min_dynamic_fps", fallback=MIN_DYNAMIC_FPS),
            MIN_DYNAMIC_FPS,
            min_value=1,
        )
        MIN_DYNAMIC_UI_FPS = _as_int(
            parser.get(
                "performance", "min_dynamic_ui_fps", fallback=MIN_DYNAMIC_UI_FPS
            ),
            MIN_DYNAMIC_UI_FPS,
            min_value=1,
        )
        UI_FPS_STEP = _as_int(
            parser.get("performance", "ui_fps_step", fallback=UI_FPS_STEP),
            UI_FPS_STEP,
            min_value=1,
        )
        CPU_LOAD_THRESHOLD = _as_float(
            parser.get(
                "performance", "cpu_load_threshold", fallback=CPU_LOAD_THRESHOLD
            ),
            CPU_LOAD_THRESHOLD,
            min_value=0.1,
            max_value=1.0,
        )
        CPU_TEMP_THRESHOLD_C = _as_float(
            parser.get(
                "performance", "cpu_temp_threshold_c", fallback=CPU_TEMP_THRESHOLD_C
            ),
            CPU_TEMP_THRESHOLD_C,
            min_value=30.0,
            max_value=100.0,
        )
        STRESS_HOLD_COUNT = _as_int(
            parser.get("performance", "stress_hold_count", fallback=STRESS_HOLD_COUNT),
            STRESS_HOLD_COUNT,
            min_value=1,
        )
        RECOVER_HOLD_COUNT = _as_int(
            parser.get(
                "performance", "recover_hold_count", fallback=RECOVER_HOLD_COUNT
            ),
            RECOVER_HOLD_COUNT,
            min_value=1,
        )
        STALE_FRAME_TIMEOUT_SEC = _as_float(
            parser.get(
                "performance",
                "stale_frame_timeout_sec",
                fallback=STALE_FRAME_TIMEOUT_SEC,
            ),
            STALE_FRAME_TIMEOUT_SEC,
            min_value=0.5,
        )
        RESTART_COOLDOWN_SEC = _as_float(
            parser.get(
                "performance", "restart_cooldown_sec", fallback=RESTART_COOLDOWN_SEC
            ),
            RESTART_COOLDOWN_SEC,
            min_value=1.0,
        )
        MAX_RESTARTS_PER_WINDOW = _as_int(
            parser.get(
                "performance",
                "max_restarts_per_window",
                fallback=MAX_RESTARTS_PER_WINDOW,
            ),
            MAX_RESTARTS_PER_WINDOW,
            min_value=1,
        )
        RESTART_WINDOW_SEC = _as_float(
            parser.get(
                "performance", "restart_window_sec", fallback=RESTART_WINDOW_SEC
            ),
            RESTART_WINDOW_SEC,
            min_value=5.0,
        )

    global USE_GSTREAMER

    if parser.has_section("camera"):
        RESCAN_INTERVAL_MS = _as_int(
            parser.get("camera", "rescan_interval_ms", fallback=RESCAN_INTERVAL_MS),
            RESCAN_INTERVAL_MS,
            min_value=500,
        )
        FAILED_CAMERA_COOLDOWN_SEC = _as_float(
            parser.get(
                "camera",
                "failed_camera_cooldown_sec",
                fallback=FAILED_CAMERA_COOLDOWN_SEC,
            ),
            FAILED_CAMERA_COOLDOWN_SEC,
            min_value=1.0,
        )
        CAMERA_SLOT_COUNT = _as_int(
            parser.get("camera", "slot_count", fallback=CAMERA_SLOT_COUNT),
            CAMERA_SLOT_COUNT,
            min_value=1,
            max_value=8,
        )
        KILL_DEVICE_HOLDERS = _as_bool(
            parser.get("camera", "kill_device_holders", fallback=KILL_DEVICE_HOLDERS),
            KILL_DEVICE_HOLDERS,
        )
        USE_GSTREAMER = _as_bool(
            parser.get("camera", "use_gstreamer", fallback=USE_GSTREAMER), USE_GSTREAMER
        )

    if parser.has_section("profile"):
        PROFILE_CAPTURE_WIDTH = _as_int(
            parser.get("profile", "capture_width", fallback=PROFILE_CAPTURE_WIDTH),
            PROFILE_CAPTURE_WIDTH,
            min_value=160,
            max_value=1920,
        )
        PROFILE_CAPTURE_HEIGHT = _as_int(
            parser.get("profile", "capture_height", fallback=PROFILE_CAPTURE_HEIGHT),
            PROFILE_CAPTURE_HEIGHT,
            min_value=120,
            max_value=1080,
        )
        PROFILE_CAPTURE_FPS = _as_int(
            parser.get("profile", "capture_fps", fallback=PROFILE_CAPTURE_FPS),
            PROFILE_CAPTURE_FPS,
            min_value=1,
            max_value=60,
        )
        PROFILE_UI_FPS = _as_int(
            parser.get("profile", "ui_fps", fallback=PROFILE_UI_FPS),
            PROFILE_UI_FPS,
            min_value=1,
            max_value=60,
        )

    if parser.has_section("health"):
        HEALTH_LOG_INTERVAL_SEC = _as_float(
            parser.get("health", "log_interval_sec", fallback=HEALTH_LOG_INTERVAL_SEC),
            HEALTH_LOG_INTERVAL_SEC,
            min_value=5.0,
        )

    if LOG_FILE_ENV:
        LOG_FILE = LOG_FILE_ENV


def configure_logging():
    level_name = (LOG_LEVEL or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []

    if LOG_FILE:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if LOG_TO_STDOUT:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    logging.captureWarnings(True)


def _read_cpu_load_ratio():
    """Read 1-minute load average normalized to CPU count."""
    try:
        load1, _, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        return min(1.0, load1 / cpu_count)
    except Exception:
        return None


def _read_cpu_temp_c():
    """Read CPU temperature in Celsius if the system exposes it."""
    paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, "r") as f:
                    raw = f.read().strip()
                if raw:
                    val = float(raw)
                    if val > 1000:
                        val = val / 1000.0
                    return val
        except Exception:
            continue
    return None


def _is_system_stressed():
    """
    Check CPU load or temperature thresholds.
    Returns: (stressed: bool, load_ratio: float|None, temp_c: float|None)
    """
    load_ratio = _read_cpu_load_ratio()
    temp_c = _read_cpu_temp_c()

    stressed = False
    if load_ratio is not None and load_ratio >= CPU_LOAD_THRESHOLD:
        stressed = True
    if temp_c is not None and temp_c >= CPU_TEMP_THRESHOLD_C:
        stressed = True

    return stressed, load_ratio, temp_c


def _write_watchdog_heartbeat():
    if os.getenv("WATCHDOG_USEC") is None:
        return
    _systemd_notify("WATCHDOG=1")


def _systemd_notify(message):
    try:
        sock_path = os.environ.get("NOTIFY_SOCKET")
        if not sock_path:
            return
        if sock_path[0] == "@":
            sock_path = "\0" + sock_path[1:]
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.connect(sock_path)
        sock.sendall(message.encode("utf-8"))
        sock.close()
    except Exception:
        logging.debug("systemd notify failed")


def _log_health_summary(
    camera_widgets, placeholder_slots, active_indexes, failed_indexes
):
    online = 0
    for w in camera_widgets:
        if getattr(w, "_latest_frame", None) is not None:
            online += 1
    logging.info(
        "Health cameras online=%d/%d placeholders=%d active=%d failed=%d",
        online,
        len(camera_widgets),
        len(placeholder_slots),
        len(active_indexes),
        len(failed_indexes),
    )
    _write_watchdog_heartbeat()


# ============================================================
# CAMERA CAPTURE WORKER
# ------------------------------------------------------------
# Runs on its own QThread to avoid blocking the UI thread.
# ============================================================


class CaptureWorker(QThread):
    # Signal emitted when a new frame is ready for the UI thread.
    frame_ready = pyqtSignal(object)
    # Signal emitted when camera connection status changes.
    status_changed = pyqtSignal(bool)

    def __init__(
        self,
        stream_link,
        parent=None,
        maxlen=1,
        target_fps=None,
        capture_width=None,
        capture_height=None,
    ):
        """Initialize camera capture settings and state."""
        super().__init__(parent)
        self.stream_link = stream_link
        self._running = True
        self._reconnect_backoff = 1.0
        self._cap = None
        self._last_emit = 0.0
        self._target_fps = target_fps
        self._emit_interval = 1.0 / 30.0
        self.capture_width = capture_width
        self.capture_height = capture_height
        self._online = False
        self._open_fail_count = 0
        # Buffer holds most recent frames, used to decouple capture from UI.
        self.buffer = deque(maxlen=maxlen)
        # Lock protects changes to FPS/emit interval from other threads.
        self._fps_lock = threading.Lock()

    def run(self):
        """Capture loop: open camera, grab frames, emit, reconnect on failure."""
        logging.info("Camera %s thread started", self.stream_link)
        while self._running:
            try:
                # Ensure capture is open; reconnect if it fails.
                if self._cap is None or not self._cap.isOpened():
                    self._open_capture()
                    if not (self._cap and self._cap.isOpened()):
                        self._open_fail_count += 1
                        if self._open_fail_count % 10 == 0:
                            logging.warning(
                                "Camera %s open failed (%d attempts)",
                                self.stream_link,
                                self._open_fail_count,
                            )
                        if self._online:
                            self._online = False
                            self.status_changed.emit(False)
                        time.sleep(self._reconnect_backoff)
                        self._reconnect_backoff = min(
                            self._reconnect_backoff * 1.5, 10.0
                        )
                        continue
                    self._reconnect_backoff = 1.0
                    self._open_fail_count = 0
                    if not self._online:
                        self._online = True
                        self.status_changed.emit(True)

                # Grab & retrieve keeps latency low vs read().
                grabbed = self._cap.grab()
                if not grabbed:
                    self._close_capture()
                    if self._online:
                        self._online = False
                        self.status_changed.emit(False)
                    continue

                ret, frame = self._cap.retrieve()
                if not ret or frame is None:
                    self._close_capture()
                    if self._online:
                        self._online = False
                        self.status_changed.emit(False)
                    continue

                now = time.time()
                with self._fps_lock:
                    emit_interval = self._emit_interval
                # Throttle emits to target FPS to avoid UI overload.
                if now - self._last_emit >= emit_interval:
                    self.buffer.append(frame)
                    self.frame_ready.emit(frame)
                    self._last_emit = now

                self.msleep(1)
            except Exception:
                logging.exception("Exception in CaptureWorker %s", self.stream_link)
                time.sleep(0.2)

        self._close_capture()
        logging.info("Camera %s thread stopped", self.stream_link)

    def _open_capture(self):
        """Open the camera and apply preferred capture settings."""
        try:
            cap = None
            backend_name = "V4L2"

            # Try GStreamer first if enabled (more efficient MJPEG pipeline)
            if (
                USE_GSTREAMER
                and platform.system() == "Linux"
                and isinstance(self.stream_link, int)
            ):
                try:
                    w = int(self.capture_width) if self.capture_width else 640
                    h = int(self.capture_height) if self.capture_height else 480
                    # fps = int(self._target_fps) if self._target_fps else 30
                    # GStreamer pipeline: v4l2src -> MJPEG decode -> BGR output
                    # Using simpler pipeline without strict format to improve compatibility
                    pipeline = (
                        f"v4l2src device=/dev/video{self.stream_link} ! "
                        f"image/jpeg,width={w},height={h} ! "
                        f"jpegdec ! videoconvert ! appsink drop=1"
                    )
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    if cap and cap.isOpened():
                        # Test if we can actually grab a frame
                        test_ret = cap.grab()
                        if test_ret:
                            backend_name = "GStreamer"
                            logging.info(
                                "GStreamer pipeline opened for camera %s",
                                self.stream_link,
                            )
                        else:
                            cap.release()
                            cap = None
                    else:
                        if cap:
                            cap.release()
                        cap = None
                except Exception as e:
                    logging.debug(
                        "GStreamer failed for camera %s: %s", self.stream_link, e
                    )
                    cap = None

            # Fallback to V4L2 if GStreamer failed or not enabled
            if cap is None:
                backend = cv2.CAP_ANY
                if platform.system() == "Linux":
                    backend = cv2.CAP_V4L2
                cap = cv2.VideoCapture(self.stream_link, backend)
                backend_name = "V4L2"

            if not cap or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                return

            # Only apply these settings for V4L2 backend (not needed for GStreamer)
            if backend_name == "V4L2":
                # Request MJPEG if available to reduce decode overhead.
                try:
                    cap.set(
                        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
                    )
                except Exception:
                    pass

                # Apply capture resolution when requested.
                if self.capture_width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
                if self.capture_height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))

                # Reduce internal buffering to keep frames current.
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                # Try to prevent blocking reads on flaky cameras.
                try:
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
                except Exception:
                    pass

                # Request FPS; 0 may let camera choose.
                try:
                    if self._target_fps and self._target_fps > 0:
                        cap.set(cv2.CAP_PROP_FPS, float(self._target_fps))
                    else:
                        cap.set(cv2.CAP_PROP_FPS, 0)
                except Exception:
                    pass

            if cap.isOpened():
                self._cap = cap
                self._configure_fps_from_camera()
                try:
                    raw = int(self._cap.get(cv2.CAP_PROP_FOURCC))
                    fourcc = "".join([chr((raw >> (8 * i)) & 0xFF) for i in range(4)])
                    if fourcc.strip() and fourcc != "MJPG":
                        logging.info(
                            "Camera %s using FOURCC=%s", self.stream_link, fourcc
                        )
                except Exception:
                    pass
                try:
                    actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
                    logging.info(
                        "Camera %s format %dx%d @ %.1f FPS (%s)",
                        self.stream_link,
                        actual_w,
                        actual_h,
                        actual_fps,
                        backend_name,
                    )
                except Exception:
                    pass
                logging.info(
                    "Opened capture %s (requested %sx%s) -> emit fps=%.1f",
                    self.stream_link,
                    self.capture_width,
                    self.capture_height,
                    1.0 / self._emit_interval if self._emit_interval > 0 else 0.0,
                )
                return
            else:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception:
            logging.exception("Failed to open capture %s", self.stream_link)

    def _configure_fps_from_camera(self):
        """Pick a usable FPS value and update emit interval."""
        if self._target_fps and self._target_fps > 0:
            fps = float(self._target_fps)
        else:
            fps = float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

        if fps <= 1.0 or fps > 240.0:
            fps = 30.0

        with self._fps_lock:
            self._emit_interval = 1.0 / max(1.0, fps)

    def set_target_fps(self, fps):
        """Update target FPS and camera setting at runtime."""
        if fps is None:
            return
        try:
            fps = float(fps)
            if fps <= 0:
                return
            with self._fps_lock:
                self._target_fps = fps
                self._emit_interval = 1.0 / max(1.0, fps)
            try:
                if self._cap:
                    self._cap.set(cv2.CAP_PROP_FPS, fps)
            except Exception:
                pass
        except Exception:
            logging.exception("set_target_fps")

    def _close_capture(self):
        """Release camera handle if open."""
        try:
            if self._cap:
                self._cap.release()
                self._cap = None
        except Exception:
            pass

    def stop(self):
        """Stop capture loop and wait briefly for thread exit."""
        self._running = False
        self.wait(2000)
        self._close_capture()


# ============================================================
# FULLSCREEN OVERLAY
# ------------------------------------------------------------
# Transparent top-level widget for fullscreen display.
# ============================================================


class FullscreenOverlay(QtWidgets.QWidget):
    def __init__(self, on_click_exit):
        """Create a full-window view with a centered QLabel."""
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.on_click_exit = on_click_exit
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setStyleSheet("background:black;")
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

    def mousePressEvent(self, a0: QtGui.QMouseEvent | None) -> None:
        """Exit fullscreen on left click/tap."""
        if a0 is not None and a0.button() == QtCore.Qt.MouseButton.LeftButton:
            self.on_click_exit()
        super().mousePressEvent(a0)

    def event(self, a0: QtCore.QEvent | None) -> bool:
        if a0 is not None and a0.type() in (
            QtCore.QEvent.Type.TouchBegin,
            QtCore.QEvent.Type.TouchEnd,
        ):
            self.on_click_exit()
            return True
        return super().event(a0)


# ============================================================
# CAMERA WIDGET
# ------------------------------------------------------------
# One tile in the grid. Manages UI input and rendering.
# ============================================================


class CameraWidget(QtWidgets.QWidget):
    # How long a press needs to be to enter "swap mode".
    hold_threshold_ms = 400

    def __init__(
        self,
        width,
        height,
        stream_link: int | None = 0,
        aspect_ratio=False,
        parent=None,
        buffer_size=1,
        target_fps=None,
        request_capture_size=(640, 480),
        ui_fps=15,
        enable_capture=True,
        placeholder_text=None,
        settings_mode=False,
        on_restart=None,
        on_night_mode_toggle=None,
    ):
        """Initialize tile UI, worker thread, and timers."""
        super().__init__(parent)
        logging.debug("Creating camera %s", stream_link)

        # Widget configuration: touch enabled, expands in grid, dark theme.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setMouseTracking(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.screen_width = max(1, width)
        self.screen_height = max(1, height)
        self.maintain_aspect_ratio = aspect_ratio
        self.camera_stream_link = stream_link
        self.widget_id = f"cam{stream_link}_{id(self)}"

        # State used for fullscreen toggle + drag/hold swap mode.
        self.is_fullscreen = False
        self.grid_position = None
        self._press_widget_id = None
        self._press_time = 0
        self._grid_parent = None
        self._touch_active = False
        self.swap_active = False

        self._fs_overlay = None

        self.capture_enabled = bool(enable_capture)
        self.placeholder_text = placeholder_text
        self.settings_mode = settings_mode
        self.night_mode_enabled = False

        # Visual styles for normal and swap-ready state
        self.normal_style = "border: 2px solid #555; background: black;"
        self.swap_ready_style = "border: 4px solid #FFFF00; background: black;"
        self.setStyleSheet(self.normal_style)
        self.setObjectName(self.widget_id)

        # Video display label or settings title
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.video_label.setMinimumSize(1, 1)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setMouseTracking(True)
        self.video_label.setObjectName(f"{self.widget_id}_label")
        self.video_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Settings tile uses buttons instead of a video stream.
        if self.settings_mode:
            self.video_label.setText(self.placeholder_text or "SETTINGS")
            self.video_label.setStyleSheet("color: #ffffff; font-size: 20px;")

            button_style = (
                "QPushButton { padding: 10px 16px; font-size: 18px; min-width: 100px; }"
            )

            restart_button = QtWidgets.QPushButton("Restart")
            restart_button.setStyleSheet(button_style)
            if on_restart:
                restart_button.clicked.connect(on_restart)

            night_mode_button = QtWidgets.QPushButton("Nightmode: Off")
            night_mode_button.setStyleSheet(button_style)
            if on_night_mode_toggle:
                night_mode_button.clicked.connect(on_night_mode_toggle)
            self.night_mode_button = night_mode_button

            exit_button = QtWidgets.QPushButton("Exit")
            exit_button.setStyleSheet(button_style)
            exit_button.clicked.connect(self._exit_app)

            layout.addStretch(1)
            layout.addWidget(self.video_label)
            layout.addSpacing(6)
            layout.addWidget(restart_button, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addSpacing(4)
            layout.addWidget(night_mode_button, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addSpacing(4)
            layout.addWidget(exit_button, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addStretch(1)
        else:
            layout.addWidget(self.video_label)

        # Render state, staleness tracking, and caches.
        self.frame_count = 0
        self.prev_time = time.time()
        self._latest_frame = None
        self._last_placeholder_text = None
        self._frame_id = 0
        self._last_rendered_id = -1
        self._last_rendered_size = None
        self._last_frame_ts = 0.0
        self._stale_frame_timeout_sec = STALE_FRAME_TIMEOUT_SEC
        self._restart_cooldown_sec = RESTART_COOLDOWN_SEC
        self._restart_window_sec = RESTART_WINDOW_SEC
        self._max_restarts_per_window = MAX_RESTARTS_PER_WINDOW
        self._restart_events = deque(maxlen=MAX_RESTARTS_PER_WINDOW * 2)
        self._last_restart_ts = 0.0
        self._last_status_log_ts = 0.0
        self._last_status_log_interval_sec = 10.0
        self._pixmap_cache = QtGui.QPixmap()
        self._scaled_pixmap_cache = None
        self._scaled_pixmap_cache_size = None
        self._night_gray = None
        self._night_bgr = None

        # Base FPS is the desired target; current FPS is adjusted dynamically.
        self.base_target_fps = target_fps
        self.current_target_fps = target_fps

        # Start capture worker in background thread (if enabled)
        self.worker = None
        if self.capture_enabled and stream_link is not None:
            cap_w, cap_h = (
                request_capture_size if request_capture_size else (None, None)
            )
            self.worker = CaptureWorker(
                stream_link,
                parent=self,
                maxlen=buffer_size,
                target_fps=target_fps,
                capture_width=cap_w,
                capture_height=cap_h,
            )
            self.worker.frame_ready.connect(self.on_frame)
            self.worker.status_changed.connect(self.on_status_changed)
            self.worker.start()
        elif not self.settings_mode:
            # No capture: set placeholder immediately
            self._latest_frame = None
            self._render_placeholder(self.placeholder_text or "DISCONNECTED")

        # Timer to render latest frame at a stable UI FPS.
        # Compensate for render overhead to hit actual target FPS.
        if not self.settings_mode:
            self.ui_render_fps = max(1, int(ui_fps))
            interval = max(1, int(1000 / self.ui_render_fps) - RENDER_OVERHEAD_MS)
            self.render_timer = QTimer(self)
            self.render_timer.setInterval(interval)
            self.render_timer.timeout.connect(self._render_latest_frame)
            self.render_timer.start()
        else:
            self.ui_render_fps = 0
            self.render_timer = None

        # Optional UI FPS diagnostics (only for real cameras)
        if self.capture_enabled and not self.settings_mode and UI_FPS_LOGGING:
            self.ui_timer = QTimer(self)
            self.ui_timer.setInterval(1000)
            self.ui_timer.timeout.connect(self._print_fps)
            self.ui_timer.start()
        else:
            self.ui_timer = None

        # Periodic status logging for observability.
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(5000)
        self._status_timer.timeout.connect(self._log_status)
        self._status_timer.start()

        self.installEventFilter(self)
        self.video_label.installEventFilter(self)

        logging.debug("Widget %s ready", self.widget_id)

    def _exit_app(self):
        """Exit the application gracefully."""
        app = QtWidgets.QApplication.instance()
        if app:
            app.quit()

    def _ensure_fullscreen_overlay(self):
        """Create fullscreen overlay only when needed."""
        if self._fs_overlay is None:
            self._fs_overlay = FullscreenOverlay(self.exit_fullscreen)

    def _apply_ui_fps(self, ui_fps):
        """Update UI render timer to match camera UI FPS.

        Compensates for render overhead to achieve actual target FPS.
        """
        self.ui_render_fps = max(1, int(ui_fps))
        if self.render_timer:
            interval = max(1, int(1000 / self.ui_render_fps) - RENDER_OVERHEAD_MS)
            self.render_timer.setInterval(interval)

    def attach_camera(self, stream_link, target_fps, request_capture_size, ui_fps=None):
        """Attach a camera to an existing placeholder slot."""
        if self.capture_enabled and self.worker:
            return

        self.capture_enabled = True
        self.camera_stream_link = stream_link
        self.base_target_fps = target_fps
        self.current_target_fps = target_fps

        if ui_fps is not None:
            self._apply_ui_fps(ui_fps)

        cap_w, cap_h = request_capture_size if request_capture_size else (None, None)
        self.worker = CaptureWorker(
            stream_link,
            parent=self,
            maxlen=1,
            target_fps=target_fps,
            capture_width=cap_w,
            capture_height=cap_h,
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.start()

        if self.ui_timer is None and UI_FPS_LOGGING:
            self.ui_timer = QTimer(self)
            self.ui_timer.setInterval(1000)
            self.ui_timer.timeout.connect(self._print_fps)
            self.ui_timer.start()

        self._latest_frame = None
        self._render_placeholder("CONNECTING...")
        logging.info("Attached camera %s to widget %s", stream_link, self.widget_id)

    def eventFilter(self, a0: QtCore.QObject | None, a1: QtCore.QEvent | None) -> bool:
        """Handle touch and mouse events from widget or label."""
        if a0 not in (self, self.video_label) or a1 is None:
            return super().eventFilter(a0, a1)

        if a1.type() == QtCore.QEvent.Type.TouchBegin:
            return self._on_touch_begin(a1)
        if a1.type() == QtCore.QEvent.Type.TouchEnd:
            return self._on_touch_end(a1)

        if a1.type() == QtCore.QEvent.Type.MouseButtonPress:
            return self._on_mouse_press(a1)
        if a1.type() == QtCore.QEvent.Type.MouseButtonRelease:
            return self._on_mouse_release(a1)
        return super().eventFilter(a0, a1)

    def _on_touch_begin(self, event):
        """Record touch-down timestamp and source widget."""
        try:
            if not event.points():
                return True
            if len(event.points()) == 1:
                self._touch_active = True
                self._press_time = time.time() * 1000.0
                self._press_widget_id = self.widget_id
                self._grid_parent = self.parent()
                logging.debug("Touch begin %s", self.widget_id)
        except Exception:
            logging.exception("touch begin")
        return True

    def _on_touch_end(self, event):
        """Handle touch-up as a click/hold action."""
        try:
            if not self._touch_active:
                return True
            self._touch_active = False
            self._handle_release_as_left_click()
        except Exception:
            logging.exception("touch end")
        return True

    def _handle_release_as_left_click(self):
        """
        Unified release handler:
        - short tap: fullscreen toggle
        - long press: swap select
        - swap if another camera is selected
        """
        try:
            if not self._press_widget_id or self._press_widget_id != self.widget_id:
                return True

            hold_time = (time.time() * 1000.0) - self._press_time
            logging.debug("Touch release %s hold=%dms", self.widget_id, int(hold_time))

            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, "selected_camera"):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            selected = getattr(swap_parent, "selected_camera", None)
            if selected == self:
                logging.debug("Clear swap %s", self.widget_id)
                setattr(swap_parent, "selected_camera", None)
                self.swap_active = False
                self.reset_style()
                self._reset_mouse_state()
                return True

            if selected and selected != self and not self.is_fullscreen:
                other = selected
                logging.debug("SWAP %s <-> %s", other.widget_id, self.widget_id)
                self.do_swap(other, self, swap_parent)
                other.swap_active = False
                other.reset_style()
                setattr(swap_parent, "selected_camera", None)
                self._reset_mouse_state()
                return True

            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                logging.debug("ENTER swap %s", self.widget_id)
                setattr(swap_parent, "selected_camera", self)
                self.swap_active = True
                self.setStyleSheet(self.swap_ready_style)
                self._reset_mouse_state()
                return True

            logging.debug("Short tap fullscreen %s", self.widget_id)
            self.toggle_fullscreen()

        except Exception:
            logging.exception("touch release")
        finally:
            self._reset_mouse_state()
        return True

    def _on_mouse_press(self, event):
        """Record mouse down position and time."""
        try:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._press_time = time.time() * 1000.0
                self._press_widget_id = self.widget_id
                self._grid_parent = self.parent()
                logging.debug("Press %s", self.widget_id)
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.toggle_fullscreen()
        except Exception:
            logging.exception("mouse press")
        return True

    def _on_mouse_release(self, event):
        """Handle mouse release as click/hold action."""
        try:
            if (
                event.button() != QtCore.Qt.MouseButton.LeftButton
                or not self._press_widget_id
                or self._press_widget_id != self.widget_id
            ):
                return True

            hold_time = (time.time() * 1000.0) - self._press_time
            logging.debug("Release %s hold=%dms", self.widget_id, int(hold_time))

            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, "selected_camera"):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            selected = getattr(swap_parent, "selected_camera", None)
            if selected == self:
                logging.debug("Clear swap %s", self.widget_id)
                setattr(swap_parent, "selected_camera", None)
                self.swap_active = False
                self.reset_style()
                self._reset_mouse_state()
                return True

            if selected and selected != self and not self.is_fullscreen:
                other = selected
                logging.debug("SWAP %s <-> %s", other.widget_id, self.widget_id)
                self.do_swap(other, self, swap_parent)
                other.swap_active = False
                other.reset_style()
                setattr(swap_parent, "selected_camera", None)
                self._reset_mouse_state()
                return True

            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                logging.debug("ENTER swap %s", self.widget_id)
                setattr(swap_parent, "selected_camera", self)
                self.swap_active = True
                self.setStyleSheet(self.swap_ready_style)
                self._reset_mouse_state()
                return True

            logging.debug("Short click fullscreen %s", self.widget_id)
            self.toggle_fullscreen()

        except Exception:
            logging.exception("mouse release")
        finally:
            self._reset_mouse_state()
        return True

    def _reset_mouse_state(self):
        """Clear press state to avoid accidental reuse."""
        self._press_time = 0
        self._press_widget_id = None
        self._grid_parent = None

    def do_swap(self, source, target, layout_parent):
        """Swap two widgets inside the grid layout."""
        try:
            source_pos = getattr(source, "grid_position", None)
            target_pos = getattr(target, "grid_position", None)
            if source_pos is None or target_pos is None:
                logging.debug("Swap failed - missing positions")
                return

            layout = layout_parent.layout()
            layout.removeWidget(source)
            layout.removeWidget(target)
            layout.addWidget(target, *source_pos)
            layout.addWidget(source, *target_pos)
            source.grid_position, target.grid_position = target_pos, source_pos
            logging.debug("Swap complete %s <-> %s", source.widget_id, target.widget_id)
        except Exception:
            logging.exception("do_swap")

    def toggle_fullscreen(self):
        """Toggle between fullscreen and grid view."""
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.go_fullscreen()

    def go_fullscreen(self):
        """Enter fullscreen mode for this camera."""
        if self.is_fullscreen:
            return
        self._ensure_fullscreen_overlay()

        if self._fs_overlay is None:
            return

        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            self._fs_overlay.setGeometry(screen.geometry())

        self._fs_overlay.showFullScreen()
        self._fs_overlay.raise_()
        self._fs_overlay.activateWindow()
        self.is_fullscreen = True

        if self._latest_frame is None and not self.settings_mode:
            self._render_placeholder(self.placeholder_text or "DISCONNECTED")

    def exit_fullscreen(self):
        """Exit fullscreen and return to grid view."""
        if not self.is_fullscreen:
            return
        if self._fs_overlay:
            self._fs_overlay.hide()
        self.is_fullscreen = False

    @pyqtSlot(object)
    def on_frame(self, frame_bgr):
        """Receive latest camera frame from worker."""
        try:
            if frame_bgr is None:
                return
            # Only store; UI thread renders on its timer.
            self._latest_frame = frame_bgr
            self._frame_id += 1
            self._last_frame_ts = time.time()
        except Exception:
            logging.exception("on_frame")

    def _render_placeholder(self, text):
        """Render placeholder text when no frame is available."""
        if self.settings_mode:
            return
        if text == self._last_placeholder_text and not self.swap_active:
            return
        target_label = (
            self._fs_overlay.label
            if (self.is_fullscreen and self._fs_overlay)
            else self.video_label
        )
        target_label.setPixmap(QtGui.QPixmap())
        target_label.setText(text)
        target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        target_label.setStyleSheet("color: #bbbbbb; font-size: 24px;")
        self._last_placeholder_text = text
        if self.swap_active:
            self.setStyleSheet(self.swap_ready_style)

    def _render_latest_frame(self):
        """Convert latest frame to QPixmap and display it."""
        if self.settings_mode:
            return
        try:
            frame_bgr = self._latest_frame
            if frame_bgr is None:
                self._render_placeholder(self.placeholder_text or "DISCONNECTED")
                return

            if (
                self._last_frame_ts
                and (time.time() - self._last_frame_ts) > self._stale_frame_timeout_sec
            ):
                self._latest_frame = None
                self._render_placeholder("DISCONNECTED")
                self._restart_capture_if_stale()
                return

            if self.is_fullscreen and self._fs_overlay:
                target_size = self._fs_overlay.size()
            else:
                target_size = self.video_label.size()

            if (
                self._frame_id == self._last_rendered_id
                and self._last_rendered_size == target_size
            ):
                return

            if self.night_mode_enabled:
                try:
                    if frame_bgr.ndim == 2:
                        h, w = frame_bgr.shape
                    else:
                        h, w = frame_bgr.shape[:2]

                    if self._night_gray is None or self._night_gray.shape != (h, w):
                        self._night_gray = np.empty((h, w), dtype=frame_bgr.dtype)
                    if self._night_bgr is None or self._night_bgr.shape[:2] != (h, w):
                        self._night_bgr = np.empty((h, w, 3), dtype=frame_bgr.dtype)

                    if frame_bgr.ndim == 2:
                        cv2.convertScaleAbs(
                            frame_bgr, alpha=1.6, beta=0, dst=self._night_gray
                        )
                    else:
                        cv2.cvtColor(
                            frame_bgr, cv2.COLOR_BGR2GRAY, dst=self._night_gray
                        )
                        cv2.convertScaleAbs(
                            self._night_gray, alpha=1.6, beta=0, dst=self._night_gray
                        )

                    self._night_bgr[:, :, 0].fill(0)
                    self._night_bgr[:, :, 1].fill(0)
                    self._night_bgr[:, :, 2] = self._night_gray
                    frame_bgr = self._night_bgr
                except Exception:
                    pass

            # Convert numpy frame to Qt image, handling grayscale or BGR.
            if frame_bgr.ndim == 2:
                h, w = frame_bgr.shape[:2]
                bytes_per_line = w
                img = QtGui.QImage(
                    bytes(frame_bgr.data),
                    w,
                    h,
                    bytes_per_line,
                    QtGui.QImage.Format.Format_Grayscale8,
                )
            else:
                h, w = frame_bgr.shape[:2]
                ch = frame_bgr.shape[2] if frame_bgr.ndim > 2 else 1
                bytes_per_line = ch * w
                img = QtGui.QImage(
                    bytes(frame_bgr.data),
                    w,
                    h,
                    bytes_per_line,
                    QtGui.QImage.Format.Format_BGR888,
                )

            self._pixmap_cache.convertFromImage(img)

            # Fullscreen scales to screen size; grid uses label size.
            if self.is_fullscreen and self._fs_overlay:
                if target_size.width() > 0 and target_size.height() > 0:
                    if (
                        self._scaled_pixmap_cache is None
                        or self._scaled_pixmap_cache_size != target_size
                    ):
                        self._scaled_pixmap_cache = QtGui.QPixmap(target_size)
                        self._scaled_pixmap_cache_size = target_size
                    self._scaled_pixmap_cache.fill(Qt.GlobalColor.black)
                    target_rect = QtCore.QRect(
                        0, 0, target_size.width(), target_size.height()
                    )
                    painter = QtGui.QPainter(self._scaled_pixmap_cache)
                    painter.drawPixmap(target_rect, self._pixmap_cache)
                    painter.end()
                    self._fs_overlay.label.setPixmap(self._scaled_pixmap_cache)
                else:
                    self._fs_overlay.label.setPixmap(self._pixmap_cache)
                self._fs_overlay.label.setText("")
            else:
                if (
                    target_size.width() > 0
                    and target_size.height() > 0
                    and self._pixmap_cache.size() != target_size
                ):
                    if (
                        self._scaled_pixmap_cache is None
                        or self._scaled_pixmap_cache_size != target_size
                    ):
                        self._scaled_pixmap_cache = QtGui.QPixmap(target_size)
                        self._scaled_pixmap_cache_size = target_size
                    self._scaled_pixmap_cache.fill(Qt.GlobalColor.black)
                    target_rect = QtCore.QRect(
                        0, 0, target_size.width(), target_size.height()
                    )
                    painter = QtGui.QPainter(self._scaled_pixmap_cache)
                    painter.drawPixmap(target_rect, self._pixmap_cache)
                    painter.end()
                    self.video_label.setPixmap(self._scaled_pixmap_cache)
                else:
                    self.video_label.setPixmap(self._pixmap_cache)
                self.video_label.setText("")

            self._last_rendered_id = self._frame_id
            self._last_rendered_size = target_size
            self._last_placeholder_text = None
            if UI_FPS_LOGGING:
                self.frame_count += 1
        except Exception:
            logging.exception("render frame")

    @pyqtSlot(bool)
    def on_status_changed(self, online):
        """Update UI when camera goes online or offline."""
        if online:
            self.setStyleSheet(self.normal_style)
            self.video_label.setText("")
            self._last_frame_ts = time.time()
        else:
            self._latest_frame = None
            self._last_rendered_id = -1
            self._render_placeholder("DISCONNECTED")

    def reset_style(self):
        """Restore default border styling."""
        self.video_label.setStyleSheet("")
        self.setStyleSheet(
            self.swap_ready_style if self.swap_active else self.normal_style
        )

    def _print_fps(self):
        """Log rendering FPS for this widget."""
        if not UI_FPS_LOGGING:
            return
        try:
            now = time.time()
            elapsed = now - self.prev_time
            if elapsed >= 1.0:
                fps = self.frame_count / elapsed if elapsed > 0 else 0.0
                logging.info("%s FPS: %.1f", self.widget_id, fps)
                self.frame_count = 0
                self.prev_time = now
        except Exception:
            pass

    def set_dynamic_fps(self, fps):
        """Apply dynamic FPS change from stress monitor."""
        if fps is None or not self.capture_enabled:
            return
        try:
            fps = float(fps)
            if fps < MIN_DYNAMIC_FPS:
                fps = MIN_DYNAMIC_FPS
            self.current_target_fps = fps
            if self.worker:
                self.worker.set_target_fps(fps)
        except Exception:
            logging.exception("set_dynamic_fps")

    def set_dynamic_ui_fps(self, ui_fps):
        """Apply dynamic UI FPS change from stress monitor."""
        if self.settings_mode:
            return
        try:
            ui_fps = int(ui_fps)
            if ui_fps < MIN_DYNAMIC_UI_FPS:
                ui_fps = MIN_DYNAMIC_UI_FPS
            self._apply_ui_fps(ui_fps)
        except Exception:
            logging.exception("set_dynamic_ui_fps")

    def _restart_capture_if_stale(self):
        """Restart the capture worker after a stale frame timeout."""
        if not self.capture_enabled or not self.worker:
            return
        now = time.time()
        if (now - self._last_restart_ts) < self._restart_cooldown_sec:
            return
        recent = [
            t for t in self._restart_events if (now - t) <= self._restart_window_sec
        ]
        if len(recent) >= self._max_restarts_per_window:
            logging.warning("Restart limit reached for %s", self.camera_stream_link)
            return
        self._restart_events.append(now)
        self._last_restart_ts = now
        try:
            logging.info(
                "Restarting capture for %s after stale frames", self.camera_stream_link
            )
            self.worker.stop()
        except Exception:
            pass

        cap_w = getattr(self.worker, "capture_width", None)
        cap_h = getattr(self.worker, "capture_height", None)
        target_fps = self.current_target_fps or self.base_target_fps
        self.worker = CaptureWorker(
            self.camera_stream_link,
            parent=self,
            maxlen=1,
            target_fps=target_fps,
            capture_width=cap_w,
            capture_height=cap_h,
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.start()
        self._render_placeholder("CONNECTING...")

    def _log_status(self):
        """Periodic status log for observability."""
        if self.settings_mode:
            return
        now = time.time()
        if (now - self._last_status_log_ts) < self._last_status_log_interval_sec:
            return
        self._last_status_log_ts = now
        format_fourcc = "unknown"
        cap = getattr(self.worker, "_cap", None) if self.worker else None
        if cap is not None:
            try:
                raw = int(cap.get(cv2.CAP_PROP_FOURCC))
                format_fourcc = "".join(
                    [chr((raw >> (8 * i)) & 0xFF) for i in range(4)]
                )
            except Exception:
                format_fourcc = "unknown"
        logging.info(
            "Camera %s status online=%s fps=%.1f ui_fps=%d fourcc=%s",
            self.camera_stream_link,
            "yes" if self._latest_frame is not None else "no",
            float(self.current_target_fps or 0),
            int(self.ui_render_fps or 0),
            format_fourcc,
        )

    def set_night_mode(self, enabled):
        """Enable or disable night mode rendering."""
        self.night_mode_enabled = bool(enabled)

    def set_night_mode_button_label(self, enabled):
        """Update settings tile button label for night mode."""
        if self.settings_mode and hasattr(self, "night_mode_button"):
            label = "Nightmode: On" if enabled else "Nightmode: Off"
            self.night_mode_button.setText(label)

    def cleanup(self):
        """Stop the capture worker thread cleanly."""
        try:
            if hasattr(self, "worker") and self.worker:
                self.worker.stop()
        except Exception:
            pass


# ============================================================
# GRID LAYOUT HELPERS
# ------------------------------------------------------------
# Calculate row/column layout based on camera count.
# ============================================================


def get_smart_grid(num_cameras):
    """Return a sensible grid (rows, cols) for N cameras."""
    if num_cameras <= 1:
        return 1, 1
    elif num_cameras == 2:
        return 1, 2
    elif num_cameras == 3:
        return 1, 3
    elif num_cameras == 4:
        return 2, 2
    elif num_cameras <= 6:
        return 2, 3
    elif num_cameras <= 9:
        return 3, 3
    else:
        cols = min(4, int(num_cameras**0.5 * 1.5))
        rows = (num_cameras + cols - 1) // cols
        return rows, cols


# ============================================================
# SYSTEM / PROCESS HELPERS
# ------------------------------------------------------------
# Used to detect and kill processes holding /dev/video*
# ============================================================


def _run_cmd(cmd):
    """Run a shell command and return stdout, stderr, returncode."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception:
        return "", "", 1


def _get_pids_from_lsof(device_path):
    """Get PIDs holding device using lsof."""
    out, _, code = _run_cmd(f"lsof -t {device_path}")
    if code != 0 or not out:
        return set()
    pids = set()
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def _get_pids_from_fuser(device_path):
    """Get PIDs holding device using fuser."""
    out, _, code = _run_cmd(f"fuser -v {device_path}")
    if code != 0 or not out:
        return set()
    pids = set()
    for match in re.findall(r"\b(\d+)\b", out):
        pids.add(int(match))
    return pids


def _is_pid_alive(pid):
    """Check if a PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def kill_device_holders(device_path, grace=0.4):
    """
    Attempt to terminate any process holding a camera device.
    Useful for kiosk-style setups.
    """
    pids = _get_pids_from_lsof(device_path)
    if not pids:
        pids = _get_pids_from_fuser(device_path)

    pids.discard(os.getpid())
    if not pids:
        return False

    logging.info("Killing holders of %s: %s", device_path, sorted(pids))

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except PermissionError:
            _run_cmd(f"sudo fuser -k {device_path}")
            break
        except Exception:
            pass

    time.sleep(grace)

    for pid in list(pids):
        if _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except PermissionError:
                _run_cmd(f"sudo fuser -k {device_path}")
            except Exception:
                pass

    return True


# ============================================================
# CAMERA DISCOVERY
# ------------------------------------------------------------
# Scan /dev/video* and test which devices are usable.
# ============================================================


def test_single_camera(
    cam_index,
    retries=3,
    retry_delay=0.2,
    allow_kill=True,
    post_kill_retries=2,
    post_kill_delay=0.25,
):
    """Try to open and grab a frame from one camera index."""
    device_path = f"/dev/video{cam_index}"

    def try_open():
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                return False
            if not cap.grab():
                return False
            return True
        finally:
            try:
                cap.release()
            except Exception:
                pass

    for _ in range(retries):
        if try_open():
            return cam_index
        time.sleep(retry_delay)

    if allow_kill and KILL_DEVICE_HOLDERS:
        killed = kill_device_holders(device_path)
        if killed:
            for _ in range(post_kill_retries):
                if try_open():
                    return cam_index
                time.sleep(post_kill_delay)

    return None


def get_video_indexes():
    """List integer indices for /dev/video* devices."""
    video_devices = glob.glob("/dev/video*")
    indexes = []
    for device in sorted(video_devices):
        try:
            index = int(device.split("video")[-1])
            indexes.append(index)
        except Exception:
            pass
    return indexes


def find_working_cameras():
    """Return a list of camera indices that can capture frames."""
    indexes = get_video_indexes()
    if not indexes:
        logging.info("No /dev/video* devices found!")
        return []

    max_workers = min(4, len(indexes))
    logging.info(
        "Testing %d cameras concurrently (workers=%d)...", len(indexes), max_workers
    )
    working = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(test_single_camera, idx) for idx in indexes]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                with lock:
                    working.append(result)
                    logging.info("Camera %d OK", result)

    # Second pass to confirm cameras without killing holders
    if working:
        logging.info("Round 2 - Double-check (no pre-kill)...")
        final_working = []
        with ThreadPoolExecutor(max_workers=min(4, len(working))) as executor:
            futures = [
                executor.submit(
                    test_single_camera,
                    idx,
                    retries=2,
                    retry_delay=0.15,
                    allow_kill=False,
                )
                for idx in working
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    final_working.append(result)
                    logging.info("Confirmed camera %d", result)
        working = final_working

    cv2.destroyAllWindows()
    logging.info("FINAL Working cameras: %s", working)
    return working


# ============================================================
# CLEANUP + PROFILE SELECTION
# ------------------------------------------------------------


def safe_cleanup(widgets):
    """Gracefully stop all camera worker threads."""
    logging.info("Cleaning all cameras")
    for w in list(widgets):
        try:
            w.cleanup()
        except Exception:
            pass


def choose_profile(camera_count):
    """Pick capture resolution and FPS based on camera count."""
    return (
        PROFILE_CAPTURE_WIDTH,
        PROFILE_CAPTURE_HEIGHT,
        PROFILE_CAPTURE_FPS,
        PROFILE_UI_FPS,
    )


# ============================================================
# MAIN ENTRYPOINT
# ------------------------------------------------------------
# Sets up app window, grid layout, cameras, and FPS monitor.
# ============================================================


def main():
    """Create the UI, discover cameras, and start event loop."""
    parser = _load_config(CONFIG_PATH)
    apply_config(parser)
    configure_logging()
    logging.info("Starting camera grid app")
    logging.info("Config loaded from %s", CONFIG_PATH)
    app = QtWidgets.QApplication(sys.argv)
    _systemd_notify("READY=1")
    camera_widgets = []
    all_widgets = []
    placeholder_slots = []

    # Clean shutdown on Ctrl+C
    def on_sigint(sig, frame):
        safe_cleanup(camera_widgets)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)
    atexit.register(lambda: safe_cleanup(camera_widgets))

    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    app.setStyleSheet("QWidget { background: #2b2b2b; color: #ffffff; }")

    mw = QtWidgets.QMainWindow()
    mw.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
    central_widget = QtWidgets.QWidget()
    setattr(central_widget, "selected_camera", None)
    mw.setCentralWidget(central_widget)

    # Show first, then fullscreen (avoids race conditions)
    mw.show()

    def force_fullscreen():
        mw.showFullScreen()
        mw.raise_()
        mw.activateWindow()

    QtCore.QTimer.singleShot(50, force_fullscreen)
    QtCore.QTimer.singleShot(300, force_fullscreen)

    primary_screen = app.primaryScreen()
    screen = (
        primary_screen.availableGeometry()
        if primary_screen
        else QtCore.QRect(0, 0, 1920, 1080)
    )
    working_cameras = find_working_cameras()
    logging.info("Found %d cameras", len(working_cameras))

    known_indexes = set(get_video_indexes())
    active_indexes = set(working_cameras)
    failed_indexes = {idx: time.time() for idx in (known_indexes - active_indexes)}

    layout = QtWidgets.QGridLayout(central_widget)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)

    def restart_app():
        """Restart the entire process (used by settings tile)."""
        logging.info("Restart requested from settings.")
        safe_cleanup(camera_widgets)
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    night_mode_state = {"enabled": False}

    def toggle_night_mode():
        """Toggle night mode for all camera widgets."""
        night_mode_state["enabled"] = not night_mode_state["enabled"]
        enabled = night_mode_state["enabled"]
        for w in all_widgets:
            if hasattr(w, "set_night_mode"):
                w.set_night_mode(enabled)
        settings_tile.set_night_mode_button_label(enabled)

    # Settings tile (always present, top-left)
    settings_tile = CameraWidget(
        width=1,
        height=1,
        stream_link=None,
        parent=central_widget,
        buffer_size=1,
        target_fps=None,
        request_capture_size=None,
        ui_fps=5,
        enable_capture=False,
        placeholder_text="SETTINGS",
        settings_mode=True,
        on_restart=restart_app,
        on_night_mode_toggle=toggle_night_mode,
    )
    all_widgets.append(settings_tile)

    active_camera_count = max(1, min(len(working_cameras), CAMERA_SLOT_COUNT))
    cap_w, cap_h, cap_fps, ui_fps = choose_profile(active_camera_count)
    logging.info("Profile: %dx%d @ %d FPS (UI %d FPS)", cap_w, cap_h, cap_fps, ui_fps)

    # Exactly 3 camera slots at all times
    for slot_idx in range(CAMERA_SLOT_COUNT):
        if slot_idx < len(working_cameras):
            cam_index = working_cameras[slot_idx]
            cw = CameraWidget(
                1,
                1,
                cam_index,
                parent=central_widget,
                buffer_size=1,
                target_fps=cap_fps,
                request_capture_size=(cap_w, cap_h),
                ui_fps=ui_fps,
                enable_capture=True,
            )
            cw.set_night_mode(night_mode_state["enabled"])
            camera_widgets.append(cw)
        else:
            cw = CameraWidget(
                1,
                1,
                stream_link=None,
                parent=central_widget,
                buffer_size=1,
                target_fps=None,
                request_capture_size=None,
                ui_fps=5,
                enable_capture=False,
                placeholder_text="DISCONNECTED",
            )
            cw.set_night_mode(night_mode_state["enabled"])
            placeholder_slots.append(cw)
        all_widgets.append(cw)

    rows, cols = get_smart_grid(len(all_widgets))
    widget_width = max(1, screen.width() // cols)
    widget_height = max(1, screen.height() // rows)

    for cw in all_widgets:
        cw.screen_width = widget_width
        cw.screen_height = widget_height

    for i, cw in enumerate(all_widgets):
        row = i // cols
        col = i % cols
        cw.grid_position = (row, col)
        layout.addWidget(cw, row, col)

    for r in range(rows):
        layout.setRowStretch(r, 1)
    for c in range(cols):
        layout.setColumnStretch(c, 1)

    # Dynamic FPS adjustment based on system stress
    if DYNAMIC_FPS_ENABLED and camera_widgets:
        stress_counter = {"stress": 0, "recover": 0}

        def adjust_fps():
            """Lower or restore FPS based on load/temperature."""
            stressed, load_ratio, temp_c = _is_system_stressed()

            if stressed:
                stress_counter["stress"] += 1
                stress_counter["recover"] = 0
            else:
                stress_counter["recover"] += 1
                stress_counter["stress"] = 0

            if stress_counter["stress"] >= STRESS_HOLD_COUNT:
                for w in camera_widgets:
                    base = w.base_target_fps or 30
                    cur = w.current_target_fps or base
                    new_fps = max(MIN_DYNAMIC_FPS, cur - 2)
                    if new_fps < cur:
                        w.set_dynamic_fps(new_fps)
                    ui_base = w.ui_render_fps or ui_fps
                    new_ui = max(MIN_DYNAMIC_UI_FPS, ui_base - UI_FPS_STEP)
                    if new_ui < ui_base:
                        w.set_dynamic_ui_fps(new_ui)
                stress_counter["stress"] = 0
                logging.info(
                    "Stress detected (load=%s, temp=%s). Lowering FPS.",
                    f"{load_ratio:.2f}" if load_ratio is not None else "n/a",
                    f"{temp_c:.1f}C" if temp_c is not None else "n/a",
                )

            if stress_counter["recover"] >= RECOVER_HOLD_COUNT:
                for w in camera_widgets:
                    base = w.base_target_fps or 30
                    cur = w.current_target_fps or base
                    new_fps = min(base, cur + 2)
                    if new_fps > cur:
                        w.set_dynamic_fps(new_fps)
                    ui_base = w.ui_render_fps or ui_fps
                    new_ui = min(ui_fps, ui_base + UI_FPS_STEP)
                    if new_ui > ui_base:
                        w.set_dynamic_ui_fps(new_ui)
                stress_counter["recover"] = 0
                logging.info("System stable. Restoring FPS.")

        perf_timer = QTimer(mw)
        perf_timer.setInterval(PERF_CHECK_INTERVAL_MS)
        perf_timer.timeout.connect(adjust_fps)
        perf_timer.start()

    # Background rescan to attach new cameras to empty slots
    if placeholder_slots:

        def rescan_and_attach():
            """Scan for new cameras and attach them to placeholders."""
            if not placeholder_slots:
                return

            now = time.time()
            indexes = get_video_indexes()

            candidates = []
            for idx in indexes:
                if idx in active_indexes:
                    continue
                last_failed = failed_indexes.get(idx)
                if last_failed and (now - last_failed) < FAILED_CAMERA_COOLDOWN_SEC:
                    continue
                candidates.append(idx)

            if not candidates:
                return

            for idx in candidates:
                if not placeholder_slots:
                    break

                ok = test_single_camera(
                    idx,
                    retries=2,
                    retry_delay=0.15,
                    allow_kill=False,
                )
                if ok is not None:
                    slot = placeholder_slots.pop(0)
                    slot.attach_camera(ok, cap_fps, (cap_w, cap_h), ui_fps=ui_fps)
                    slot.set_night_mode(night_mode_state["enabled"])
                    camera_widgets.append(slot)
                    active_indexes.add(ok)
                    failed_indexes.pop(ok, None)
                    logging.info("Attached camera %d to empty slot", ok)
                else:
                    failed_indexes[idx] = now

        rescan_timer = QTimer(mw)
        rescan_timer.setInterval(RESCAN_INTERVAL_MS)
        rescan_timer.timeout.connect(rescan_and_attach)
        rescan_timer.start()

    if HEALTH_LOG_INTERVAL_SEC > 0:
        health_timer = QTimer(mw)
        health_timer.setInterval(int(HEALTH_LOG_INTERVAL_SEC * 1000))
        health_timer.timeout.connect(
            lambda: _log_health_summary(
                camera_widgets,
                placeholder_slots,
                active_indexes,
                failed_indexes,
            )
        )
        health_timer.start()

    app.aboutToQuit.connect(lambda: safe_cleanup(camera_widgets))
    QtGui.QShortcut(
        QtGui.QKeySequence("q"), mw, lambda: (safe_cleanup(camera_widgets), app.quit())
    )

    logging.info("Short click=fullscreen toggle. Hold 400ms=swap mode. Ctrl+Q=quit.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
