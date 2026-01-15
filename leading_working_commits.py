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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# CAMERA THREAD = Reads video (without freezing hopefully)
class CaptureWorker(QThread):
    """
    Reads frames from a camera device in a background QThread.
    Throttles frame emission to `target_fps` to reduce main-thread load.
    Requests desired capture resolution (capture_width/capture_height) where supported.
    Emits `frame_ready` with raw BGR numpy arrays (no big copies).
    """
    frame_ready = pyqtSignal(object)   # NumPy BGR frame
    status_changed = pyqtSignal(bool)  # Online/offline

    def __init__(
        self,
        stream_link,
        parent=None,
        maxlen=1,
        target_fps=None,  # None = auto-detect from camera (max reported)
        capture_width=None,
        capture_height=None,
    ):
        super().__init__(parent)
        self.stream_link = stream_link
        self._running = True
        self._reconnect_backoff = 1.0
        self._cap = None
        self._last_emit = 0.0
        self._target_fps = target_fps  # None means auto
        self._emit_interval = 1.0 / 30.0
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.buffer = deque(maxlen=maxlen)

    def run(self):
        logging.info("Camera %s thread started", self.stream_link)
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._open_capture()
                    if not (self._cap and self._cap.isOpened()):
                        # failed -> exponential backoff
                        time.sleep(self._reconnect_backoff)
                        self._reconnect_backoff = min(self._reconnect_backoff * 1.5, 10.0)
                        continue
                    # Reset backoff and notify UI
                    self._reconnect_backoff = 1.0
                    self.status_changed.emit(True)

                # Grab + read for minimal latency: prefer grab() then retrieve()
                grabbed = self._cap.grab()
                if not grabbed:
                    logging.debug("grab failed for %s", self.stream_link)
                    self._close_capture()
                    self.status_changed.emit(False)
                    continue

                ret, frame = self._cap.retrieve()
                if not ret or frame is None:
                    logging.debug("retrieve failed for %s", self.stream_link)
                    self._close_capture()
                    self.status_changed.emit(False)
                    continue

                now = time.time()
                # Throttle frame emission to detected/target fps
                if now - self._last_emit >= self._emit_interval:
                    # Use latest frame only (drop intermediate frames)
                    self.buffer.append(frame)
                    self.frame_ready.emit(frame)
                    self._last_emit = now

                # Yield CPU a bit; keep small sleep to avoid busy loop
                self.msleep(1)  # 1 ms
            except Exception:
                logging.exception("Exception in CaptureWorker %s", self.stream_link)
                time.sleep(0.2)

        self._close_capture()
        logging.info("Camera %s thread stopped", self.stream_link)

    def _open_capture(self):
        """Try to open capture and apply sane settings."""
        try:
            # platform-specific flags can help on Windows vs Linux
            backend = cv2.CAP_ANY
            # On Linux prefer V4L2 for USB cameras
            if platform.system() == "Linux":
                backend = cv2.CAP_V4L2
            cap = cv2.VideoCapture(self.stream_link, backend)
            if not cap or not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                return

            # Prefer MJPG where possible to reduce CPU
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            except Exception:
                pass

            # Request lower resolution for performance if provided
            if self.capture_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
            if self.capture_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))

            # Decrease internal buffer to keep latency low
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Try to set FPS hint (drivers may ignore)
            try:
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
        # If target_fps is set, use it directly
        if self._target_fps and self._target_fps > 0:
            fps = float(self._target_fps)
        else:
            # Auto-detect from driver (maximum reported)
            fps = float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

        # Some drivers return 0 or nonsense; fallback to 30
        if fps <= 1.0 or fps > 240.0:
            fps = 30.0

        self._emit_interval = 1.0 / max(1.0, fps)

    def _close_capture(self):
        try:
            if self._cap:
                self._cap.release()
                self._cap = None
        except Exception:
            pass

    def stop(self):
        self._running = False
        # Wait for thread to exit (give it a little time)
        self.wait(2000)
        self._close_capture()


# CAMERA WIDGET = Each camera lives here
class CameraWidget(QtWidgets.QWidget):
    """
    One widget per camera. Optimized frame handling:
    - The worker throttles frames (auto-detected fps)
    - Avoids expensive cv2.resize and avoids unnecessary copies
    """
    hold_threshold_ms = 400

    def __init__(self, width, height, stream_link=0, aspect_ratio=False, parent=None, buffer_size=1, target_fps=None, request_capture_size=(640,480)):
        super().__init__(parent)
        logging.debug("Creating camera %s", stream_link)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setMouseTracking(True)

        self.screen_width = max(1, width)
        self.screen_height = max(1, height)
        self.maintain_aspect_ratio = aspect_ratio
        self.camera_stream_link = stream_link
        self.widget_id = f"cam{stream_link}_{id(self)}"

        self.is_fullscreen = False
        self.grid_position = None
        self._saved_parent = None
        self._saved_position = None
        self._press_widget_id = None
        self._press_time = 0
        self._grid_parent = None
        self._touch_active = False

        self._fullscreen_window_flags = Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        self._normal_window_flags = Qt.WindowType.Widget

        self.normal_style = "border: 2px solid #555; background: black;"
        self.swap_ready_style = "border: 4px solid #FFFF00; background: black;"
        self.setStyleSheet(self.normal_style)
        self.setObjectName(self.widget_id)

        # Video display label
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                       QtWidgets.QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(1, 1)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Let QLabel handle visual scaling; avoid resizing frames repeatedly
        self.video_label.setScaledContents(True)
        self.video_label.setMouseTracking(True)
        self.video_label.setObjectName(f"{self.widget_id}_label")
        self.video_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        # FPS counter
        self.frame_count = 0
        self.prev_time = time.time()

        # Worker: ask for a reasonable capture size to lower CPU
        cap_w, cap_h = request_capture_size if request_capture_size else (None, None)
        self.worker = CaptureWorker(
            stream_link,
            parent=self,
            maxlen=buffer_size,
            target_fps=target_fps,  # None = auto-detect
            capture_width=cap_w,
            capture_height=cap_h,
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.start()

        # UI timer prints FPS once per second
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(1000)
        self.ui_timer.timeout.connect(self._print_fps)
        self.ui_timer.start()

        # mouse event handling
        self.installEventFilter(self)
        self.video_label.installEventFilter(self)

        logging.debug("Widget %s ready", self.widget_id)

    def eventFilter(self, obj, event):
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
        try:
            if not event.points():
                return True
            # Treat single-finger touch like left mouse press
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
        try:
            if not self._touch_active:
                return True
            self._touch_active = False
            # Mirror left mouse release behavior
            self._handle_release_as_left_click()
        except Exception:
            logging.exception("touch end")
        return True

    def _handle_release_as_left_click(self):
        try:
            if not self._press_widget_id or self._press_widget_id != self.widget_id:
                return True

            hold_time = (time.time() * 1000.0) - self._press_time
            logging.debug("Touch release %s hold=%dms", self.widget_id, int(hold_time))

            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, 'selected_camera'):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            if swap_parent.selected_camera == self:
                logging.debug("Clear swap %s", self.widget_id)
                swap_parent.selected_camera = None
                self.reset_style()
                self._reset_mouse_state()
                return True

            if (swap_parent.selected_camera and
                    swap_parent.selected_camera != self and
                    not self.is_fullscreen):
                other = swap_parent.selected_camera
                logging.debug("SWAP %s <-> %s", other.widget_id, self.widget_id)
                self.do_swap(other, self, swap_parent)
                other.reset_style()
                swap_parent.selected_camera = None
                self._reset_mouse_state()
                return True

            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                logging.debug("ENTER swap %s", self.widget_id)
                swap_parent.selected_camera = self
                self.video_label.setStyleSheet(self.swap_ready_style)
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
        try:
            if (event.button() != QtCore.Qt.MouseButton.LeftButton or
                    not self._press_widget_id or self._press_widget_id != self.widget_id):
                return True

            hold_time = (time.time() * 1000.0) - self._press_time
            logging.debug("Release %s hold=%dms", self.widget_id, int(hold_time))

            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, 'selected_camera'):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            if swap_parent.selected_camera == self:
                logging.debug("Clear swap %s", self.widget_id)
                swap_parent.selected_camera = None
                self.reset_style()
                self._reset_mouse_state()
                return True

            if (swap_parent.selected_camera and
                    swap_parent.selected_camera != self and
                    not self.is_fullscreen):
                other = swap_parent.selected_camera
                logging.debug("SWAP %s <-> %s", other.widget_id, self.widget_id)
                self.do_swap(other, self, swap_parent)
                other.reset_style()
                swap_parent.selected_camera = None
                self._reset_mouse_state()
                return True

            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                logging.debug("ENTER swap %s", self.widget_id)
                swap_parent.selected_camera = self
                self.video_label.setStyleSheet(self.swap_ready_style)
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
        self._press_time = 0
        self._press_widget_id = None
        self._grid_parent = None

    def do_swap(self, source, target, layout_parent):
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
            logging.debug("Swap complete %s <-> %s", source.widget_id, target.widget_id)
        except Exception:
            logging.exception("do_swap")

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.go_fullscreen()

    def go_fullscreen(self):
        if self.is_fullscreen:
            return
        try:
            logging.debug("%s -> fullscreen", self.widget_id)
            self.setUpdatesEnabled(False)
            self._saved_parent = self.parent()
            self._saved_position = getattr(self, 'grid_position', None)
            if self._saved_parent and self._saved_parent.layout():
                try:
                    self._saved_parent.layout().removeWidget(self)
                except Exception:
                    pass
            self.setParent(None)
            self.setWindowFlags(self._fullscreen_window_flags)
            self.showFullScreen()
            self.raise_()
            self.activateWindow()
            self.is_fullscreen = True
        except Exception:
            logging.exception("go_fullscreen")
        finally:
            self.setUpdatesEnabled(True)

    def exit_fullscreen(self):
        if not self.is_fullscreen:
            return
        try:
            logging.debug("%s <- grid[%s]", self.widget_id, self._saved_position)
            self.setUpdatesEnabled(False)
            self.setWindowFlags(self._normal_window_flags)
            if self._saved_parent and self._saved_position is not None:
                self.setParent(self._saved_parent)
                layout = self._saved_parent.layout()
                if layout:
                    layout.addWidget(self, *self._saved_position)
            self.show()
            self.is_fullscreen = False
            if self._saved_parent and self._saved_parent.window():
                self._saved_parent.window().showFullScreen()
        except Exception:
            logging.exception("exit_fullscreen")
        finally:
            self.setUpdatesEnabled(True)

    @pyqtSlot(object)
    def on_frame(self, frame_bgr):
        """
        Called on the main thread via signal. Expects a BGR numpy frame.
        Optimization notes:
        - Avoid cv2.resize: QLabel.scaledContents=True will scale the pixmap.
        - Use QImage.Format_BGR888 to avoid BGR->RGB conversion.
        """
        try:
            if frame_bgr is None:
                return

            h, w, ch = frame_bgr.shape
            bytes_per_line = ch * w

            img = QtGui.QImage(frame_bgr.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_label.setPixmap(pix)

            self.frame_count += 1
        except Exception:
            logging.exception("on_frame")

    @pyqtSlot(bool)
    def on_status_changed(self, online):
        if online:
            self.setStyleSheet(self.normal_style)
        else:
            self.video_label.clear()

    def reset_style(self):
        self.video_label.setStyleSheet("")
        self.setStyleSheet(self.normal_style)

    def _print_fps(self):
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

    def cleanup(self):
        try:
            if hasattr(self, 'worker') and self.worker:
                self.worker.stop()
        except Exception:
            pass


# === HELPER FUNCTIONS ===
def get_smart_grid(num_cameras):
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


def _run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception:
        return "", "", 1


def _get_pids_from_lsof(device_path):
    # lsof -t gives only PIDs, one per line
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
    # fuser outputs PIDs; use -v for verbose, parse digits
    out, _, code = _run_cmd(f"fuser -v {device_path}")
    if code != 0 or not out:
        return set()
    # Extract all PIDs from output
    pids = set()
    for match in re.findall(r"\b(\d+)\b", out):
        pids.add(int(match))
    return pids


def _is_pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def kill_device_holders(device_path, grace=0.4):
    """
    Targeted kill: identify PIDs holding this device and kill only them.
    Uses lsof first, falls back to fuser.
    """
    pids = _get_pids_from_lsof(device_path)
    if not pids:
        pids = _get_pids_from_fuser(device_path)

    # Never kill ourselves
    pids.discard(os.getpid())

    if not pids:
        return False

    logging.info("Killing holders of %s: %s", device_path, sorted(pids))

    # Try graceful kill first
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except PermissionError:
            # Fallback to fuser with sudo for that device
            _run_cmd(f"sudo fuser -k {device_path}")
            break
        except Exception:
            pass

    time.sleep(grace)

    # Force kill if still alive
    for pid in list(pids):
        if _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except PermissionError:
                _run_cmd(f"sudo fuser -k {device_path}")
            except Exception:
                pass

    return True


def test_single_camera(
    cam_index,
    retries=3,
    retry_delay=0.2,
    allow_kill=True,
    post_kill_retries=2,
    post_kill_delay=0.25,
):
    """
    Efficient, reliable probe:
    - Retry a few times (short delay) before killing anything.
    - Only do targeted kill once if still failing.
    - After kill, retry a couple times.
    """
    device_path = f"/dev/video{cam_index}"

    def try_open():
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                return False
            # Quick grab to ensure device is responsive
            if not cap.grab():
                return False
            return True
        finally:
            try:
                cap.release()
            except Exception:
                pass

    # First phase: quick retries, no kill
    for _ in range(retries):
        if try_open():
            return cam_index
        time.sleep(retry_delay)

    # Optional targeted cleanup if still failing
    if allow_kill:
        killed = kill_device_holders(device_path)
        if killed:
            for _ in range(post_kill_retries):
                if try_open():
                    return cam_index
                time.sleep(post_kill_delay)

    return None


def get_video_indexes():
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
    indexes = get_video_indexes()
    if not indexes:
        logging.info("No /dev/video* devices found!")
        return []

    max_workers = min(4, len(indexes))  # Lower CPU pressure for Raspberry Pi
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


def safe_cleanup(widgets):
    logging.info("Cleaning all cameras")
    for w in list(widgets):
        try:
            w.cleanup()
        except Exception:
            pass


# === MAIN APPLICATION ===
def main():
    logging.info("Starting camera grid app")
    app = QtWidgets.QApplication(sys.argv)
    camera_widgets = []

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
    mw.showFullScreen()

    screen = app.primaryScreen().availableGeometry()
    working_cameras = find_working_cameras()
    logging.info("Found %d cameras", len(working_cameras))

    layout = QtWidgets.QGridLayout(central_widget)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)

    if working_cameras:
        rows, cols = get_smart_grid(len(working_cameras))
        widget_width = max(1, screen.width() // cols)
        widget_height = max(1, screen.height() // rows)

        # Create widgets (limit to 9)
        for cam_index in working_cameras[:9]:
            # Request a sensible capture resolution that matches widget size but limited
            req_w = min(widget_width, 1280)
            req_h = min(widget_height, 720)
            cw = CameraWidget(widget_width, widget_height, cam_index, parent=central_widget,
                              buffer_size=1, target_fps=None, request_capture_size=(req_w, req_h))
            camera_widgets.append(cw)

        for i, cw in enumerate(camera_widgets):
            row = i // cols
            col = i % cols
            cw.grid_position = (row, col)
            layout.addWidget(cw, row, col)
    else:
        label = QtWidgets.QLabel("NO CAMERAS FOUND")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: #888;")
        layout.addWidget(label, 0, 0)

    app.aboutToQuit.connect(lambda: safe_cleanup(camera_widgets))
    QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Q'), mw,
                    lambda: (safe_cleanup(camera_widgets), app.quit()))

    logging.info("Short click=fullscreen toggle. Hold 400ms=swap mode. Ctrl+Q=quit.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
