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
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import glob
import subprocess
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
import sys
import cv2
import time
import traceback
from collections import deque
import atexit
import signal
import platform
import os
import re

# ============================================================
# DEBUG PRINTS (disabled by default)
# ------------------------------------------------------------
# Simple toggle for extra print logs without changing code.
# ============================================================
# DEBUG_PRINTS = True
DEBUG_PRINTS = False


def dprint(*args, **kwargs):
    """Lightweight debug print wrapper."""
    if DEBUG_PRINTS:
        print(*args, **kwargs)


# ============================================================
# LOGGING
# ------------------------------------------------------------
# Standard logging for normal runtime info and errors.
# ============================================================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# DYNAMIC PERFORMANCE TUNING
# ------------------------------------------------------------
# Controls how FPS adapts when CPU or temperature is high.
# ============================================================
DYNAMIC_FPS_ENABLED = True
PERF_CHECK_INTERVAL_MS = 2000
MIN_DYNAMIC_FPS = 5
CPU_LOAD_THRESHOLD = 0.85   # 85% avg load
CPU_TEMP_THRESHOLD_C = 70.0  # Celsius
STRESS_HOLD_COUNT = 2       # consecutive checks before reducing fps
RECOVER_HOLD_COUNT = 3      # consecutive checks before increasing fps

# ============================================================
# CAMERA RESCAN (HOT-PLUG SUPPORT)
# ------------------------------------------------------------
RESCAN_INTERVAL_MS = 5000
FAILED_CAMERA_COOLDOWN_SEC = 30.0


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
                        time.sleep(self._reconnect_backoff)
                        self._reconnect_backoff = min(
                            self._reconnect_backoff * 1.5, 10.0)
                        continue
                    self._reconnect_backoff = 1.0
                    self.status_changed.emit(True)

                # Grab & retrieve keeps latency low vs read().
                grabbed = self._cap.grab()
                if not grabbed:
                    self._close_capture()
                    self.status_changed.emit(False)
                    continue

                ret, frame = self._cap.retrieve()
                if not ret or frame is None:
                    self._close_capture()
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
                logging.exception(
                    "Exception in CaptureWorker %s", self.stream_link)
                time.sleep(0.2)

        self._close_capture()
        logging.info("Camera %s thread stopped", self.stream_link)

    def _open_capture(self):
        """Open the camera and apply preferred capture settings."""
        try:
            backend = cv2.CAP_ANY
            if platform.system() == "Linux":
                backend = cv2.CAP_V4L2
            cap = cv2.VideoCapture(self.stream_link, backend)
            if not cap or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                return

            # Request MJPEG if available to reduce decode overhead.
            try:
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
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
        self.setStyleSheet("background:black;")
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                 QtWidgets.QSizePolicy.Policy.Ignored)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

    def mousePressEvent(self, event):
        """Exit fullscreen on left click/tap."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.on_click_exit()
        super().mousePressEvent(event)

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
        stream_link=0,
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
            QtWidgets.QSizePolicy.Policy.Expanding
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

        # Visual styles for normal and swap-ready state
        self.normal_style = "border: 2px solid #555; background: black;"
        self.swap_ready_style = "border: 4px solid #FFFF00; background: black;"
        self.setStyleSheet(self.normal_style)
        self.setObjectName(self.widget_id)

        # Video display label or settings title
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                       QtWidgets.QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(1, 1)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setMouseTracking(True)
        self.video_label.setObjectName(f"{self.widget_id}_label")
        self.video_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Settings tile uses buttons instead of a video stream.
        if self.settings_mode:
            self.video_label.setText(self.placeholder_text or "SETTINGS")
            self.video_label.setStyleSheet("color: #ffffff; font-size: 20px;")

            button_style = "QPushButton { padding: 10px 16px; font-size: 18px; min-width: 100px; }"

            restart_button = QtWidgets.QPushButton("Restart")
            restart_button.setStyleSheet(button_style)
            if on_restart:
                restart_button.clicked.connect(on_restart)

            exit_button = QtWidgets.QPushButton("Exit")
            exit_button.setStyleSheet(button_style)
            exit_button.clicked.connect(self._exit_app)

            layout.addStretch(1)
            layout.addWidget(self.video_label)
            layout.addSpacing(12)
            layout.addWidget(
                restart_button, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addSpacing(8)
            layout.addWidget(
                exit_button, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addStretch(1)
        else:
            layout.addWidget(self.video_label)

        # FPS counters for logging UI render performance.
        self.frame_count = 0
        self.prev_time = time.time()
        self._latest_frame = None

        # Base FPS is the desired target; current FPS is adjusted dynamically.
        self.base_target_fps = target_fps
        self.current_target_fps = target_fps

        # Start capture worker in background thread (if enabled)
        self.worker = None
        if self.capture_enabled and stream_link is not None:
            cap_w, cap_h = request_capture_size if request_capture_size else (
                None, None)
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
        # This is intentionally decoupled from capture FPS.
        if not self.settings_mode:
            self.ui_render_fps = max(1, int(ui_fps))
            self.render_timer = QTimer(self)
            self.render_timer.setInterval(int(1000 / self.ui_render_fps))
            self.render_timer.timeout.connect(self._render_latest_frame)
            self.render_timer.start()
        else:
            self.ui_render_fps = 0
            self.render_timer = None

        # Timer to print UI FPS diagnostics (only for real cameras)
        if self.capture_enabled and not self.settings_mode:
            self.ui_timer = QTimer(self)
            self.ui_timer.setInterval(1000)
            self.ui_timer.timeout.connect(self._print_fps)
            self.ui_timer.start()
        else:
            self.ui_timer = None

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
        """Update UI render timer to match camera UI FPS."""
        self.ui_render_fps = max(1, int(ui_fps))
        if self.render_timer:
            self.render_timer.setInterval(int(1000 / self.ui_render_fps))

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

        cap_w, cap_h = request_capture_size if request_capture_size else (
            None, None)
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

        if self.ui_timer is None:
            self.ui_timer = QTimer(self)
            self.ui_timer.setInterval(1000)
            self.ui_timer.timeout.connect(self._print_fps)
            self.ui_timer.start()

        self._latest_frame = None
        self._render_placeholder("CONNECTING...")
        logging.info("Attached camera %s to widget %s",
                     stream_link, self.widget_id)

    def eventFilter(self, obj, event):
        """Handle touch and mouse events from widget or label."""
        if obj not in (self, self.video_label):
            return super().eventFilter(obj, event)

        if event.type() == QtCore.QEvent.Type.TouchBegin:
            return self._on_touch_begin(event)
        if event.type() == QtCore.QEvent.Type.TouchEnd:
            return self._on_touch_end(event)

        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            return self._on_mouse_press(event)
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            return self._on_mouse_release(event)
        return super().eventFilter(obj, event)

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
            logging.debug("Touch release %s hold=%dms",
                          self.widget_id, int(hold_time))

            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, 'selected_camera'):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            if swap_parent.selected_camera == self:
                logging.debug("Clear swap %s", self.widget_id)
                swap_parent.selected_camera = None
                self.swap_active = False
                self.reset_style()
                self._reset_mouse_state()
                return True

            if (swap_parent.selected_camera and
                    swap_parent.selected_camera != self and
                    not self.is_fullscreen):
                other = swap_parent.selected_camera
                logging.debug("SWAP %s <-> %s",
                              other.widget_id, self.widget_id)
                self.do_swap(other, self, swap_parent)
                other.swap_active = False
                other.reset_style()
                swap_parent.selected_camera = None
                self._reset_mouse_state()
                return True

            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                logging.debug("ENTER swap %s", self.widget_id)
                swap_parent.selected_camera = self
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
            if (event.button() != QtCore.Qt.MouseButton.LeftButton or
                    not self._press_widget_id or self._press_widget_id != self.widget_id):
                return True

            hold_time = (time.time() * 1000.0) - self._press_time
            logging.debug("Release %s hold=%dms",
                          self.widget_id, int(hold_time))

            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, 'selected_camera'):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            if swap_parent.selected_camera == self:
                logging.debug("Clear swap %s", self.widget_id)
                swap_parent.selected_camera = None
                self.swap_active = False
                self.reset_style()
                self._reset_mouse_state()
                return True

            if (swap_parent.selected_camera and
                    swap_parent.selected_camera != self and
                    not self.is_fullscreen):
                other = swap_parent.selected_camera
                logging.debug("SWAP %s <-> %s",
                              other.widget_id, self.widget_id)
                self.do_swap(other, self, swap_parent)
                other.swap_active = False
                other.reset_style()
                swap_parent.selected_camera = None
                self._reset_mouse_state()
                return True

            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                logging.debug("ENTER swap %s", self.widget_id)
                swap_parent.selected_camera = self
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
            source_pos = getattr(source, 'grid_position', None)
            target_pos = getattr(target, 'grid_position', None)
            if source_pos is None or target_pos is None:
                logging.debug("Swap failed - missing positions")
                return

            layout = layout_parent.layout()
            layout.removeWidget(source)
            layout.removeWidget(target)
            layout.addWidget(target, *source_pos)
            layout.addWidget(source, *target_pos)
            source.grid_position, target.grid_position = target_pos, source_pos
            logging.debug("Swap complete %s <-> %s",
                          source.widget_id, target.widget_id)
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
        except Exception:
            logging.exception("on_frame")

    def _render_placeholder(self, text):
        """Render placeholder text when no frame is available."""
        if self.settings_mode:
            return
        target_label = self._fs_overlay.label if (
            self.is_fullscreen and self._fs_overlay) else self.video_label
        target_label.setPixmap(QtGui.QPixmap())
        target_label.setText(text)
        target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        target_label.setStyleSheet("color: #bbbbbb; font-size: 24px;")
        if self.swap_active:
            self.setStyleSheet(self.swap_ready_style)

    def _render_latest_frame(self):
        """Convert latest frame to QPixmap and display it."""
        if self.settings_mode:
            return
        try:
            frame_bgr = self._latest_frame
            if frame_bgr is None:
                self._render_placeholder(
                    self.placeholder_text or "DISCONNECTED")
                return

            # Convert numpy frame to Qt image, handling grayscale or BGR.
            if frame_bgr.ndim == 2:
                h, w = frame_bgr.shape
                bytes_per_line = w
                img = QtGui.QImage(
                    frame_bgr.data, w, h, bytes_per_line,
                    QtGui.QImage.Format.Format_Grayscale8
                )
            else:
                h, w, ch = frame_bgr.shape
                bytes_per_line = ch * w
                img = QtGui.QImage(
                    frame_bgr.data, w, h, bytes_per_line,
                    QtGui.QImage.Format.Format_BGR888
                )

            pix = QtGui.QPixmap.fromImage(img)

            # Fullscreen scales to screen size; grid uses label size.
            if self.is_fullscreen and self._fs_overlay:
                target_size = self._fs_overlay.size()
                if target_size.width() > 0 and target_size.height() > 0:
                    pix = pix.scaled(
                        target_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation
                    )
                self._fs_overlay.label.setPixmap(pix)
                self._fs_overlay.label.setText("")
            else:
                self.video_label.setPixmap(pix)
                self.video_label.setText("")

            self.frame_count += 1
        except Exception:
            logging.exception("render frame")

    @pyqtSlot(bool)
    def on_status_changed(self, online):
        """Update UI when camera goes online or offline."""
        if online:
            self.setStyleSheet(self.normal_style)
            self.video_label.setText("")
        else:
            self._latest_frame = None
            self._render_placeholder("DISCONNECTED")

    def reset_style(self):
        """Restore default border styling."""
        self.video_label.setStyleSheet("")
        self.setStyleSheet(
            self.swap_ready_style if self.swap_active else self.normal_style)

    def _print_fps(self):
        """Log rendering FPS for this widget."""
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

    def cleanup(self):
        """Stop the capture worker thread cleanly."""
        try:
            if hasattr(self, 'worker') and self.worker:
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
            cmd, shell=True, capture_output=True, text=True, timeout=2)
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

    if allow_kill:
        killed = kill_device_holders(device_path)
        if killed:
            for _ in range(post_kill_retries):
                if try_open():
                    return cam_index
                time.sleep(post_kill_delay)

    return None


def get_video_indexes():
    """List integer indices for /dev/video* devices."""
    video_devices = glob.glob('/dev/video*')
    indexes = []
    for device in sorted(video_devices):
        try:
            index = int(device.split('video')[-1])
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
    logging.info("Testing %d cameras concurrently (workers=%d)...",
                 len(indexes), max_workers)
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
                executor.submit(test_single_camera, idx, retries=2,
                                retry_delay=0.15, allow_kill=False)
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
    return 640, 480, 20, 20

# ============================================================
# MAIN ENTRYPOINT
# ------------------------------------------------------------
# Sets up app window, grid layout, cameras, and FPS monitor.
# ============================================================


def main():
    """Create the UI, discover cameras, and start event loop."""
    logging.info("Starting camera grid app")
    app = QtWidgets.QApplication(sys.argv)
    camera_widgets = []
    all_widgets = []
    placeholder_slots = []

    CAMERA_SLOT_COUNT = 3

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
    central_widget.selected_camera = None
    mw.setCentralWidget(central_widget)

    # Show first, then fullscreen (avoids race conditions)
    mw.show()

    def force_fullscreen():
        mw.showFullScreen()
        mw.raise_()
        mw.activateWindow()

    QtCore.QTimer.singleShot(50, force_fullscreen)
    QtCore.QTimer.singleShot(300, force_fullscreen)

    screen = app.primaryScreen().availableGeometry()
    working_cameras = find_working_cameras()
    logging.info("Found %d cameras", len(working_cameras))

    known_indexes = set(get_video_indexes())
    active_indexes = set(working_cameras)
    failed_indexes = {idx: time.time()
                      for idx in (known_indexes - active_indexes)}

    layout = QtWidgets.QGridLayout(central_widget)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)

    def restart_app():
        """Restart the entire process (used by settings tile)."""
        logging.info("Restart requested from settings.")
        safe_cleanup(camera_widgets)
        python = sys.executable
        os.execv(python, [python] + sys.argv)

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
    )
    all_widgets.append(settings_tile)

    active_camera_count = max(1, min(len(working_cameras), CAMERA_SLOT_COUNT))
    cap_w, cap_h, cap_fps, ui_fps = choose_profile(active_camera_count)
    logging.info("Profile: %dx%d @ %d FPS (UI %d FPS)",
                 cap_w, cap_h, cap_fps, ui_fps)

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
                stress_counter["stress"] = 0
                logging.info("Stress detected (load=%s, temp=%s). Lowering FPS.",
                             f"{load_ratio:.2f}" if load_ratio is not None else "n/a",
                             f"{temp_c:.1f}C" if temp_c is not None else "n/a")

            if stress_counter["recover"] >= RECOVER_HOLD_COUNT:
                for w in camera_widgets:
                    base = w.base_target_fps or 30
                    cur = w.current_target_fps or base
                    new_fps = min(base, cur + 2)
                    if new_fps > cur:
                        w.set_dynamic_fps(new_fps)
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
                    slot.attach_camera(
                        ok, cap_fps, (cap_w, cap_h), ui_fps=ui_fps)
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

    app.aboutToQuit.connect(lambda: safe_cleanup(camera_widgets))
    QtGui.QShortcut(QtGui.QKeySequence('q'), mw,
                    lambda: (safe_cleanup(camera_widgets), app.quit()))

    logging.info(
        "Short click=fullscreen toggle. Hold 400ms=swap mode. Ctrl+Q=quit.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
