# ============================================================
# TABLE OF CONTENTS
# ------------------------------------------------------------
# 1. DEBUG PRINTS
# 2. LOGGING
# 3. DYNAMIC PERFORMANCE TUNING
# 4. CAMERA RESCAN (HOT-PLUG SUPPORT)
# 5. CAMERA CAPTURE WORKER (GPU-ACCELERATED)
# 6. OPENGL VIDEO WIDGET
# 7. FULLSCREEN OVERLAY
# 8. CAMERA WIDGET
# 9. GRID LAYOUT HELPERS
# 10. SYSTEM / PROCESS HELPERS
# 11. CAMERA DISCOVERY
# 12. CLEANUP + PROFILE SELECTION
# 13. MAIN ENTRYPOINT
# ============================================================
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import glob
import subprocess
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import (
    glClear, glClearColor, glEnable, glDisable,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glBegin, glEnd, glVertex2f, glTexCoord2f,
    GL_COLOR_BUFFER_BIT, GL_TEXTURE_2D, GL_QUADS,
    GL_RGB, GL_UNSIGNED_BYTE, GL_LINEAR,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    glViewport, glMatrixMode, glLoadIdentity, glOrtho,
    GL_PROJECTION, GL_MODELVIEW, GL_BLEND, GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA, glBlendFunc, glDeleteTextures
)
import sys
import cv2
import time
from collections import deque
import atexit
import signal
import platform
import os
import re
import numpy as np

# ============================================================
# OPTIONAL: picamera2 for Pi Camera modules (GPU decode)
# ============================================================
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# DEBUG PRINTS (disabled by default)
# ============================================================
DEBUG_PRINTS = False

def dprint(*args, **kwargs):
    """Lightweight debug print wrapper."""
    if DEBUG_PRINTS:
        print(*args, **kwargs)

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# DYNAMIC PERFORMANCE TUNING (Pi-optimized thresholds)
# ============================================================
DYNAMIC_FPS_ENABLED = True
PERF_CHECK_INTERVAL_MS = 2000
MIN_DYNAMIC_FPS = 5  # Lower floor for Pi
CPU_LOAD_THRESHOLD = 0.75  # More aggressive for Pi
CPU_TEMP_THRESHOLD_C = 70.0  # Pi throttles at 80-85°C
STRESS_HOLD_COUNT = 2
RECOVER_HOLD_COUNT = 3

# ============================================================
# CAMERA RESCAN (HOT-PLUG SUPPORT)
# ============================================================
RESCAN_INTERVAL_MS = 10000  # Less frequent on Pi
FAILED_CAMERA_COOLDOWN_SEC = 30.0

# ============================================================
# DETECT RASPBERRY PI
# ============================================================
def is_raspberry_pi():
    """Detect if running on a Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        return 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo
    except:
        return False

IS_RASPBERRY_PI = is_raspberry_pi()

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
    """Check CPU load or temperature thresholds."""
    load_ratio = _read_cpu_load_ratio()
    temp_c = _read_cpu_temp_c()

    stressed = False
    if load_ratio is not None and load_ratio >= CPU_LOAD_THRESHOLD:
        stressed = True
    if temp_c is not None and temp_c >= CPU_TEMP_THRESHOLD_C:
        stressed = True

    return stressed, load_ratio, temp_c

# ============================================================
# GSTREAMER PIPELINE BUILDER (Hardware-accelerated)
# ============================================================
def build_gstreamer_pipeline(device_index, width=640, height=480, fps=15):
    """
    Build a GStreamer pipeline for hardware-accelerated capture.
    Uses v4l2 hardware decode when available.
    """
    if IS_RASPBERRY_PI:
        # Pi-optimized pipeline with hardware decode
        pipeline = (
            f"v4l2src device=/dev/video{device_index} ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1"
        )
    else:
        # Generic pipeline
        pipeline = (
            f"v4l2src device=/dev/video{device_index} ! "
            f"video/x-raw,width={width},height={height} ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1"
        )
    return pipeline

# ============================================================
# CAMERA CAPTURE WORKER (GPU-ACCELERATED)
# ============================================================
class CaptureWorker(QThread):
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(bool)

    def __init__(
        self,
        stream_link,
        parent=None,
        maxlen=1,
        target_fps=None,
        capture_width=None,
        capture_height=None,
        use_picamera2=False,
        use_gstreamer=True,
    ):
        """Initialize camera capture with GPU acceleration options."""
        super().__init__(parent)
        self.stream_link = stream_link
        self._running = True
        self._reconnect_backoff = 1.0
        self._cap = None
        self._picam = None
        self._last_emit = 0.0
        self._target_fps = target_fps or 15
        self._emit_interval = 1.0 / self._target_fps
        self.capture_width = capture_width or 640
        self.capture_height = capture_height or 480
        self.buffer = deque(maxlen=maxlen)
        self._fps_lock = threading.Lock()
        
        # GPU acceleration options
        self.use_picamera2 = use_picamera2 and PICAMERA2_AVAILABLE
        self.use_gstreamer = use_gstreamer and not self.use_picamera2
        
        # Lower thread priority on Pi
        if IS_RASPBERRY_PI:
            try:
                os.nice(5)
            except:
                pass

    def run(self):
        """Capture loop with GPU-accelerated backends."""
        logging.info("Camera %s thread started (picamera2=%s, gstreamer=%s)", 
                     self.stream_link, self.use_picamera2, self.use_gstreamer)
        
        while self._running:
            try:
                # Try picamera2 first for Pi Camera modules
                if self.use_picamera2 and self._picam is None:
                    if self._try_open_picamera2():
                        self.status_changed.emit(True)
                        self._capture_loop_picamera2()
                        continue
                
                # Fall back to OpenCV (with optional GStreamer)
                if self._cap is None or not self._cap.isOpened():
                    self._open_capture()
                    if not (self._cap and self._cap.isOpened()):
                        time.sleep(self._reconnect_backoff)
                        self._reconnect_backoff = min(self._reconnect_backoff * 1.5, 10.0)
                        continue
                    self._reconnect_backoff = 1.0
                    self.status_changed.emit(True)

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
                
                if now - self._last_emit >= emit_interval:
                    self.buffer.append(frame)
                    self.frame_ready.emit(frame)
                    self._last_emit = now

                self.msleep(1)
            except Exception:
                logging.exception("Exception in CaptureWorker %s", self.stream_link)
                time.sleep(0.2)

        self._close_capture()
        self._close_picamera2()
        logging.info("Camera %s thread stopped", self.stream_link)

    def _try_open_picamera2(self):
        """Try to open a Pi Camera using picamera2 (GPU-accelerated)."""
        if not PICAMERA2_AVAILABLE:
            return False
        try:
            self._picam = Picamera2(self.stream_link if isinstance(self.stream_link, int) else 0)
            config = self._picam.create_preview_configuration(
                main={"size": (self.capture_width, self.capture_height), "format": "RGB888"},
                buffer_count=2
            )
            self._picam.configure(config)
            self._picam.start()
            logging.info("Opened picamera2 for camera %s (GPU decode)", self.stream_link)
            return True
        except Exception as e:
            logging.debug("picamera2 failed for %s: %s", self.stream_link, e)
            self._close_picamera2()
            return False

    def _capture_loop_picamera2(self):
        """Capture frames using picamera2 (GPU-accelerated)."""
        while self._running and self._picam:
            try:
                # capture_array uses GPU for decode
                frame = self._picam.capture_array()
                if frame is None:
                    continue
                
                # Convert RGB to BGR for consistency
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                now = time.time()
                with self._fps_lock:
                    emit_interval = self._emit_interval
                
                if now - self._last_emit >= emit_interval:
                    self.buffer.append(frame_bgr)
                    self.frame_ready.emit(frame_bgr)
                    self._last_emit = now

                self.msleep(1)
            except Exception:
                logging.exception("picamera2 capture error")
                break
        
        self._close_picamera2()
        self.status_changed.emit(False)

    def _open_capture(self):
        """Open camera with GStreamer hardware acceleration or fallback."""
        try:
            cap = None
            
            # Try GStreamer pipeline first (hardware-accelerated on Pi)
            if self.use_gstreamer and isinstance(self.stream_link, int):
                pipeline = build_gstreamer_pipeline(
                    self.stream_link,
                    self.capture_width,
                    self.capture_height,
                    int(self._target_fps)
                )
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap and cap.isOpened():
                    self._cap = cap
                    logging.info("Opened GStreamer pipeline for camera %s (GPU path)", self.stream_link)
                    return
                else:
                    logging.debug("GStreamer failed, falling back to V4L2")
                    try:
                        cap.release()
                    except:
                        pass

            # Fallback to V4L2 (software decode)
            backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.stream_link, backend)
            
            if not cap or not cap.isOpened():
                try:
                    cap.release()
                except:
                    pass
                return

            # Request MJPEG to reduce CPU decode overhead
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except:
                pass

            if self.capture_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
            if self.capture_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass

            try:
                if self._target_fps and self._target_fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, float(self._target_fps))
            except:
                pass

            if cap.isOpened():
                self._cap = cap
                logging.info("Opened V4L2 capture for camera %s (software decode)", self.stream_link)
            else:
                try:
                    cap.release()
                except:
                    pass
        except Exception:
            logging.exception("Failed to open capture %s", self.stream_link)

    def _close_capture(self):
        """Release OpenCV capture."""
        try:
            if self._cap:
                self._cap.release()
                self._cap = None
        except:
            pass

    def _close_picamera2(self):
        """Release picamera2 resources."""
        try:
            if self._picam:
                self._picam.stop()
                self._picam.close()
                self._picam = None
        except:
            pass

    def set_target_fps(self, fps):
        """Update target FPS at runtime."""
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
            except:
                pass
        except Exception:
            logging.exception("set_target_fps")

    def stop(self):
        """Stop capture loop."""
        self._running = False
        self.wait(2000)
        self._close_capture()
        self._close_picamera2()

# ============================================================
# OPENGL VIDEO WIDGET (GPU-accelerated rendering)
# ============================================================
class GLVideoWidget(QOpenGLWidget):
    """OpenGL-based video display widget for GPU-accelerated rendering."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._texture_id = None
        self._frame_data = None
        self._frame_size = (0, 0)
        self._lock = threading.Lock()
        self._initialized = False
        self._placeholder_text = None
        
        # Set size policy
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.setMinimumSize(1, 1)

    def initializeGL(self):
        """Initialize OpenGL context."""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Generate texture
        self._texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        self._initialized = True

    def resizeGL(self, w, h):
        """Handle widget resize."""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 1, 0, -1, 1)  # Flip Y for image coordinates
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        """Render the current frame using OpenGL."""
        glClear(GL_COLOR_BUFFER_BIT)
        
        with self._lock:
            frame_data = self._frame_data
            frame_size = self._frame_size
        
        if frame_data is None or frame_size[0] == 0:
            # No frame - could render placeholder text here
            return
        
        # Update texture with new frame data
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            frame_size[0], frame_size[1], 0,
            GL_RGB, GL_UNSIGNED_BYTE, frame_data
        )
        
        # Draw textured quad filling the entire widget (no aspect ratio preservation)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(1, 0)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(0, 1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def update_frame(self, frame_bgr):
        """Update the frame to be rendered (called from capture thread)."""
        if frame_bgr is None:
            return
        
        try:
            # Convert BGR to RGB for OpenGL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Ensure contiguous memory layout
            frame_rgb = np.ascontiguousarray(frame_rgb)
            
            h, w = frame_rgb.shape[:2]
            
            with self._lock:
                self._frame_data = frame_rgb.tobytes()
                self._frame_size = (w, h)
            
            # Schedule repaint on UI thread
            self.update()
        except Exception:
            logging.exception("update_frame error")

    def clear_frame(self):
        """Clear the current frame."""
        with self._lock:
            self._frame_data = None
            self._frame_size = (0, 0)
        self.update()

    def cleanup(self):
        """Release OpenGL resources."""
        if self._initialized and self._texture_id:
            try:
                glDeleteTextures([self._texture_id])
            except:
                pass

# ============================================================
# FULLSCREEN OVERLAY (OpenGL-based)
# ============================================================
class FullscreenOverlay(QtWidgets.QWidget):
    def __init__(self, on_click_exit):
        """Create a full-window view with OpenGL rendering."""
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.on_click_exit = on_click_exit
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background:black;")
        
        # Use OpenGL widget for GPU rendering
        self.gl_widget = GLVideoWidget(self)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.gl_widget)

    def update_frame(self, frame_bgr):
        """Update the displayed frame."""
        self.gl_widget.update_frame(frame_bgr)

    def mousePressEvent(self, event):
        """Exit fullscreen on left click/tap."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.on_click_exit()
        super().mousePressEvent(event)

    def cleanup(self):
        """Release resources."""
        self.gl_widget.cleanup()

# ============================================================
# CAMERA WIDGET (GPU-accelerated)
# ============================================================
class CameraWidget(QtWidgets.QWidget):
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
        use_picamera2=False,
        use_gstreamer=True,
    ):
        """Initialize tile UI with GPU-accelerated rendering."""
        super().__init__(parent)
        logging.debug("Creating camera %s", stream_link)

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
        self.use_picamera2 = use_picamera2
        self.use_gstreamer = use_gstreamer

        self.normal_style = "border: 2px solid #555; background: black;"
        self.swap_ready_style = "border: 4px solid #FFFF00; background: black;"
        self.setStyleSheet(self.normal_style)
        self.setObjectName(self.widget_id)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if self.settings_mode:
            # Settings tile uses standard QLabel
            self.gl_widget = None
            self.video_label = QtWidgets.QLabel(self)
            self.video_label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding
            )
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_label.setText(self.placeholder_text or "SETTINGS")
            self.video_label.setStyleSheet("color: #ffffff; font-size: 20px;")

            restart_button = QtWidgets.QPushButton("Restart")
            restart_button.setStyleSheet(
                "QPushButton { padding: 10px 16px; font-size: 18px; }"
            )
            if on_restart:
                restart_button.clicked.connect(on_restart)

            layout.addStretch(1)
            layout.addWidget(self.video_label)
            layout.addSpacing(12)
            layout.addWidget(restart_button, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addStretch(1)
        else:
            # Use OpenGL widget for GPU rendering
            self.gl_widget = GLVideoWidget(self)
            self.video_label = None
            layout.addWidget(self.gl_widget)
            
            # Placeholder label overlay (shown when no frame)
            self.placeholder_label = QtWidgets.QLabel(self)
            self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder_label.setStyleSheet("color: #bbbbbb; font-size: 24px; background: transparent;")
            self.placeholder_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.placeholder_label.hide()

        self.frame_count = 0
        self.prev_time = time.time()
        self._latest_frame = None

        self.base_target_fps = target_fps
        self.current_target_fps = target_fps

        # Start capture worker
        self.worker = None
        if self.capture_enabled and stream_link is not None:
            cap_w, cap_h = request_capture_size if request_capture_size else (640, 480)
            self.worker = CaptureWorker(
                stream_link,
                parent=self,
                maxlen=buffer_size,
                target_fps=target_fps,
                capture_width=cap_w,
                capture_height=cap_h,
                use_picamera2=self.use_picamera2,
                use_gstreamer=self.use_gstreamer,
            )
            self.worker.frame_ready.connect(self.on_frame)
            self.worker.status_changed.connect(self.on_status_changed)
            self.worker.start()
        elif not self.settings_mode:
            self._latest_frame = None
            self._show_placeholder(self.placeholder_text or "DISCONNECTED")

        # UI render timer (GPU rendering is much faster)
        if not self.settings_mode:
            self.ui_render_fps = max(10, min(30, int(ui_fps)))
            self.render_timer = QTimer(self)
            self.render_timer.setInterval(round(1000 / self.ui_render_fps))
            self.render_timer.timeout.connect(self._render_latest_frame)
            self.render_timer.start()
        else:
            self.ui_render_fps = 0
            self.render_timer = None

        # FPS logging timer
        if self.capture_enabled and not self.settings_mode:
            self.ui_timer = QTimer(self)
            self.ui_timer.setInterval(1000)
            self.ui_timer.timeout.connect(self._print_fps)
            self.ui_timer.start()
        else:
            self.ui_timer = None

        self.installEventFilter(self)
        if self.gl_widget:
            self.gl_widget.installEventFilter(self)
        if self.video_label:
            self.video_label.installEventFilter(self)

        logging.debug("Widget %s ready", self.widget_id)

    def _ensure_fullscreen_overlay(self):
        """Create fullscreen overlay only when needed."""
        if self._fs_overlay is None:
            self._fs_overlay = FullscreenOverlay(self.exit_fullscreen)

    def _show_placeholder(self, text):
        """Show placeholder text."""
        if self.settings_mode:
            return
        if hasattr(self, 'placeholder_label'):
            self.placeholder_label.setText(text)
            self.placeholder_label.setGeometry(self.rect())
            self.placeholder_label.show()
            self.placeholder_label.raise_()

    def _hide_placeholder(self):
        """Hide placeholder text."""
        if hasattr(self, 'placeholder_label'):
            self.placeholder_label.hide()

    def resizeEvent(self, event):
        """Handle resize to reposition placeholder."""
        super().resizeEvent(event)
        if hasattr(self, 'placeholder_label'):
            self.placeholder_label.setGeometry(self.rect())

    def attach_camera(self, stream_link, target_fps, request_capture_size, ui_fps=None):
        """Attach a camera to an existing placeholder slot."""
        if self.capture_enabled and self.worker:
            return

        self.capture_enabled = True
        self.camera_stream_link = stream_link
        self.base_target_fps = target_fps
        self.current_target_fps = target_fps

        if ui_fps is not None:
            self.ui_render_fps = min(30, max(10, int(ui_fps)))
            if self.render_timer:
                self.render_timer.setInterval(int(1000 / self.ui_render_fps))

        cap_w, cap_h = request_capture_size if request_capture_size else (640, 480)
        self.worker = CaptureWorker(
            stream_link,
            parent=self,
            maxlen=1,
            target_fps=target_fps,
            capture_width=cap_w,
            capture_height=cap_h,
            use_picamera2=self.use_picamera2,
            use_gstreamer=self.use_gstreamer,
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
        self._show_placeholder("CONNECTING...")
        logging.info("Attached camera %s to widget %s", stream_link, self.widget_id)

    def eventFilter(self, obj, event):
        """Handle touch and mouse events."""
        valid_objects = [self]
        if self.gl_widget:
            valid_objects.append(self.gl_widget)
        if self.video_label:
            valid_objects.append(self.video_label)
            
        if obj not in valid_objects:
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
        """Record touch-down timestamp."""
        try:
            if not event.points():
                return True
            if len(event.points()) == 1:
                self._touch_active = True
                self._press_time = time.time() * 1000.0
                self._press_widget_id = self.widget_id
                self._grid_parent = self.parent()
        except Exception:
            logging.exception("touch begin")
        return True

    def _on_touch_end(self, event):
        """Handle touch-up."""
        try:
            if not self._touch_active:
                return True
            self._touch_active = False
            self._handle_release_as_left_click()
        except Exception:
            logging.exception("touch end")
        return True

    def _handle_release_as_left_click(self):
        """Unified release handler."""
        try:
            if not self._press_widget_id or self._press_widget_id != self.widget_id:
                return True

            hold_time = (time.time() * 1000.0) - self._press_time

            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, 'selected_camera'):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            if swap_parent.selected_camera == self:
                swap_parent.selected_camera = None
                self.swap_active = False
                self.reset_style()
                self._reset_mouse_state()
                return True

            if (swap_parent.selected_camera and
                    swap_parent.selected_camera != self and
                    not self.is_fullscreen):
                other = swap_parent.selected_camera
                self.do_swap(other, self, swap_parent)
                other.swap_active = False
                other.reset_style()
                swap_parent.selected_camera = None
                self._reset_mouse_state()
                return True

            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                swap_parent.selected_camera = self
                self.swap_active = True
                self.setStyleSheet(self.swap_ready_style)
                self._reset_mouse_state()
                return True

            self.toggle_fullscreen()

        except Exception:
            logging.exception("touch release")
        finally:
            self._reset_mouse_state()
        return True

    def _on_mouse_press(self, event):
        """Record mouse down."""
        try:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._press_time = time.time() * 1000.0
                self._press_widget_id = self.widget_id
                self._grid_parent = self.parent()
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.toggle_fullscreen()
        except Exception:
            logging.exception("mouse press")
        return True

    def _on_mouse_release(self, event):
        """Handle mouse release."""
        try:
            if (event.button() != QtCore.Qt.MouseButton.LeftButton or
                    not self._press_widget_id or self._press_widget_id != self.widget_id):
                return True
            self._handle_release_as_left_click()
        except Exception:
            logging.exception("mouse release")
        return True

    def _reset_mouse_state(self):
        """Clear press state."""
        self._press_time = 0
        self._press_widget_id = None
        self._grid_parent = None

    def do_swap(self, source, target, layout_parent):
        """Swap two widgets in the grid."""
        try:
            source_pos = getattr(source, 'grid_position', None)
            target_pos = getattr(target, 'grid_position', None)
            if source_pos is None or target_pos is None:
                return

            layout = layout_parent.layout()
            layout.removeWidget(source)
            layout.removeWidget(target)
            layout.addWidget(target, *source_pos)
            layout.addWidget(source, *target_pos)
            source.grid_position, target.grid_position = target_pos, source_pos
        except Exception:
            logging.exception("do_swap")

    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.go_fullscreen()

    def go_fullscreen(self):
        """Enter fullscreen."""
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

    def exit_fullscreen(self):
        """Exit fullscreen."""
        if not self.is_fullscreen:
            return
        if self._fs_overlay:
            self._fs_overlay.hide()
        self.is_fullscreen = False

    @pyqtSlot(object)
    def on_frame(self, frame_bgr):
        """Receive frame from worker."""
        try:
            if frame_bgr is None:
                return
            self._latest_frame = frame_bgr
        except Exception:
            logging.exception("on_frame")

    def _render_latest_frame(self):
        """Render the latest frame using GPU."""
        if self.settings_mode:
            return
        try:
            frame_bgr = self._latest_frame
            if frame_bgr is None:
                self._show_placeholder(self.placeholder_text or "DISCONNECTED")
                return

            self._hide_placeholder()

            # GPU-accelerated rendering via OpenGL
            if self.gl_widget:
                self.gl_widget.update_frame(frame_bgr)

            # Also update fullscreen overlay if active
            if self.is_fullscreen and self._fs_overlay:
                self._fs_overlay.update_frame(frame_bgr)

            self.frame_count += 1
        except Exception:
            logging.exception("render frame")

    @pyqtSlot(bool)
    def on_status_changed(self, online):
        """Update UI on status change."""
        if online:
            self.setStyleSheet(self.normal_style)
            self._hide_placeholder()
        else:
            self._latest_frame = None
            if self.gl_widget:
                self.gl_widget.clear_frame()
            self._show_placeholder("DISCONNECTED")

    def reset_style(self):
        """Restore default styling."""
        self.setStyleSheet(self.swap_ready_style if self.swap_active else self.normal_style)

    def _print_fps(self):
        """Log FPS."""
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
        """Apply dynamic FPS change."""
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
        """Stop worker and release resources."""
        try:
            if hasattr(self, 'worker') and self.worker:
                self.worker.stop()
            if hasattr(self, 'gl_widget') and self.gl_widget:
                self.gl_widget.cleanup()
            if hasattr(self, '_fs_overlay') and self._fs_overlay:
                self._fs_overlay.cleanup()
        except Exception:
            pass

# ============================================================
# GRID LAYOUT HELPERS
# ============================================================
def get_smart_grid(num_cameras):
    """Return a sensible grid (rows, cols) for N cameras."""
    if num_cameras <= 1: return 1, 1
    elif num_cameras == 2: return 1, 2
    elif num_cameras == 3: return 1, 3
    elif num_cameras == 4: return 2, 2
    elif num_cameras <= 6: return 2, 3
    elif num_cameras <= 9: return 3, 3
    else:
        cols = min(4, int(num_cameras**0.5 * 1.5))
        rows = (num_cameras + cols - 1) // cols
        return rows, cols

# ============================================================
# SYSTEM / PROCESS HELPERS
# ============================================================
def _run_cmd(cmd):
    """Run a shell command."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
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
    """Terminate processes holding a camera device."""
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
    logging.info("Testing %d cameras concurrently (workers=%d)...", len(indexes), max_workers)
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

    if working:
        logging.info("Round 2 - Double-check (no pre-kill)...")
        final_working = []
        with ThreadPoolExecutor(max_workers=min(4, len(working))) as executor:
            futures = [
                executor.submit(test_single_camera, idx, retries=2, retry_delay=0.15, allow_kill=False)
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
# CLEANUP + PROFILE SELECTION (Pi-optimized)
# ============================================================
def safe_cleanup(widgets):
    """Gracefully stop all camera worker threads."""
    logging.info("Cleaning all cameras")
    for w in list(widgets):
        try:
            w.cleanup()
        except Exception:
            pass

def choose_profile(camera_count):
    """Pick capture resolution and FPS based on camera count (Pi-optimized)."""
    return 640, 480, 20, 20

# ============================================================
# MAIN ENTRYPOINT
# ============================================================
def main():
    """Create the UI, discover cameras, and start event loop."""
    logging.info("Starting camera grid app (Raspberry Pi: %s)", IS_RASPBERRY_PI)
    logging.info("picamera2 available: %s", PICAMERA2_AVAILABLE)
    
    app = QtWidgets.QApplication(sys.argv)
    camera_widgets = []
    all_widgets = []
    placeholder_slots = []

    CAMERA_SLOT_COUNT = 3

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
    failed_indexes = {idx: time.time() for idx in (known_indexes - active_indexes)}

    layout = QtWidgets.QGridLayout(central_widget)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)

    def restart_app():
        """Restart the entire process."""
        logging.info("Restart requested from settings.")
        safe_cleanup(camera_widgets)
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    # Settings tile
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
    logging.info("Profile: %dx%d @ %d FPS (UI %d FPS)", cap_w, cap_h, cap_fps, ui_fps)

    # Camera slots
    for slot_idx in range(CAMERA_SLOT_COUNT):
        if slot_idx < len(working_cameras):
            cam_index = working_cameras[slot_idx]
            cw = CameraWidget(
                1, 1,
                cam_index,
                parent=central_widget,
                buffer_size=1,
                target_fps=cap_fps,
                request_capture_size=(cap_w, cap_h),
                ui_fps=ui_fps,
                enable_capture=True,
                use_picamera2=PICAMERA2_AVAILABLE and IS_RASPBERRY_PI,
                use_gstreamer=IS_RASPBERRY_PI,
            )
            camera_widgets.append(cw)
        else:
            cw = CameraWidget(
                1, 1,
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

    # Dynamic FPS adjustment
    if DYNAMIC_FPS_ENABLED and camera_widgets:
        stress_counter = {"stress": 0, "recover": 0}

        def adjust_fps():
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

    # Background rescan
    if placeholder_slots:
        def rescan_and_attach():
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

    logging.info("Short click=fullscreen toggle. Hold 400ms=swap mode. Q=quit.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()