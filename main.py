"""
Multi-camera grid viewer
"""

import glob
import subprocess
from PyQt6 import QtCore, QtGui, QtWidgets  # GUI framework - makes windows, buttons
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer  # Core Qt features
import sys  # System utilities - exit cleanly
import cv2  # OpenCV - reads camera video frames
import time 
import traceback  # Error reporting - show crashes clearly
from collections import deque  # Ring buffer - keeps last 4 frames
import atexit  # Runs cleanup code when program exits
import signal  # Handles Ctrl+C gracefully

# CAMERA THREAD = Reads video (without freezing hopefully)
class CaptureWorker(QThread):
    """
    Runs in background thread. Reads camera frames every 10ms.
    Emits new frames to main thread via signals (thread-safe).
    Auto-reconnects if camera disconnects.
    """
    frame_ready = pyqtSignal(object)  # Sends frame to GUI
    status_changed = pyqtSignal(bool)  # Online/offline status

    def __init__(self, stream_link, parent=None, maxlen=4):
        super().__init__(parent)
        self.stream_link = stream_link  # Camera ID (0, 1, 2...)
        self._running = True  # Stop flag
        self._reconnect_backoff = 1.0  # Wait longer each reconnect attempt
        self._cap = None  # OpenCV camera object
        self.buffer = deque(maxlen=maxlen)  # Keeps last 4 frames

    def run(self):
        """Infinite loop - grab frames until stopped."""
        print(f"DEBUG: Camera {self.stream_link} thread started")
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._open_capture()
                    if not (self._cap and self._cap.isOpened()):
                        # Failed - wait longer next time (exponential backoff)
                        time.sleep(self._reconnect_backoff)
                        self._reconnect_backoff = min(self._reconnect_backoff * 1.5, 10.0)
                        continue
                    self.status_changed.emit(True)  # Tell GUI: "I'm back online"

                # Read one frame
                status, frame = self._cap.read()
                if not status or frame is None:
                    self._close_capture()
                    self.status_changed.emit(False)
                    continue

                # Send frame to GUI thread
                self.buffer.append(frame)
                self.frame_ready.emit(frame)
                time.sleep(0.01)  # Don't overload CPU because my main targets were Raspberry Pis
                
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)
        
        self._close_capture()

    def _open_capture(self):
        """Try V4L2 first (Linux USB cameras), then any backend."""
        try:
                cap = cv2.VideoCapture(self.stream_link, cv2.CAP_ANY)
                if cap.isOpened():
                    # MJPG = lower CPU usage
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    cap.set(cv2.CAP_PROP_FPS, 30)           # Set 30 FPS
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Clear old frames
                    self._cap = cap
                    return
                cap.release()
        except:
            pass

    def _close_capture(self):
        """Safely close camera."""
        try:
            if self._cap:
                self._cap.release()
                self._cap = None
        except:
            pass

    def stop(self):
        """Stop thread cleanly."""
        self._running = False
        self.wait(timeout=2000)  # Wait up to 2 seconds
        self._close_capture()

# CAMERA WIDGET = Each camera lives here
class CameraWidget(QtWidgets.QWidget):
    """
    One widget per camera. Shows video, handles clicks, fullscreen, swapping.
    Thread-safe - updates only happen on main GUI thread.
    """
    hold_threshold_ms = 400  # Hold this long to enter swap mode

    def __init__(self, width, height, stream_link=0, aspect_ratio=False, parent=None, buffer_size=4):
        super().__init__(parent)
        print(f"DEBUG: Creating camera {stream_link}")
        
        # Widget settings
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMouseTracking(True)  # Track mouse position
        
        # Sizing
        self.screen_width = max(1, width)
        self.screen_height = max(1, height)
        self.maintain_aspect_ratio = aspect_ratio
        self.camera_stream_link = stream_link  # Which /dev/video?
        
        # Unique ID for each camera widget
        self.widget_id = f"cam{stream_link}_{id(self)}"

        # State flags
        self.is_fullscreen = False
        self.grid_position = None  # (row, col) in grid
        self._saved_parent = None  # Parent at fullscreen time
        self._saved_position = None
        self._press_widget_id = None  # Locks exact widget during click
        self._press_time = 0  # Click timing
        self._grid_parent = None  # Layout parent at click/press time

        # Colors
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
        self.video_label.setScaledContents(True)  # Auto-fit video
        self.video_label.setMouseTracking(True)
        self.video_label.setObjectName(f"{self.widget_id}_label")

        # Layout = video fills entire widget
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        # FPS counter
        self.frame_count = 0
        self.prev_time = time.time()

        # Start background camera thread
        self.worker = CaptureWorker(stream_link, parent=self, maxlen=buffer_size)
        self.worker.frame_ready.connect(self.on_frame)  # Thread → GUI
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.start()

        # FPS timer = runs on main thread
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(1000)
        self.ui_timer.timeout.connect(self._print_fps)
        self.ui_timer.start()

        # Mouse event handling = hoping for correct behavior
        self.installEventFilter(self)
        self.video_label.installEventFilter(self)
        print(f"DEBUG: Widget {self.widget_id} ready")

    def eventFilter(self, obj, event):
        """
        Catches mouse events BEFORE they reach normal handlers.
        obj = self or video_label → handle our events only.
        """
        if obj not in (self, self.video_label):
            return super().eventFilter(obj, event)
            
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            return self._on_mouse_press(event)
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            return self._on_mouse_release(event)
        return super().eventFilter(obj, event)

    def _on_mouse_press(self, event):
        """Mouse DOWN - record which widget and when."""
        try:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # LOCK IDENTITY - prevents wrong camera bug
                self._press_time = time.time() * 1000.0
                self._press_widget_id = self.widget_id  # Exact widget
                self._grid_parent = self.parent()  # Layout at press time
                print(f"DEBUG: Press {self.widget_id}")
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.toggle_fullscreen()
        except Exception:
            traceback.print_exc()
        return True  # STOP EVENT - no bubbling to parents

    def _on_mouse_release(self, event):
        """
        Mouse UP - 4 possible outcomes based on priority:
        1. Click selected camera → clear swap mode
        2. Other camera selected → swap positions  
        3. Hold 400ms → enter swap mode
        4. Short click → fullscreen toggle
        """
        try:
            # Wrong widget? Ignore the event
            if (event.button() != QtCore.Qt.MouseButton.LeftButton or 
                not self._press_widget_id or self._press_widget_id != self.widget_id):
                return True

            hold_time = (time.time() * 1000.0) - self._press_time
            print(f"DEBUG: Release {self.widget_id}, hold={hold_time:.0f}ms")

            # Need layout parent for swap operations
            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, 'selected_camera'):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            # PRIORITY 1: Click SELECTED camera → clear swap
            if swap_parent.selected_camera == self:
                print(f"DEBUG: Clear swap {self.widget_id}")
                swap_parent.selected_camera = None
                self.reset_style()
                self._reset_mouse_state()
                return True

            # PRIORITY 2: Swap with OTHER selected camera
            if (swap_parent.selected_camera and 
                swap_parent.selected_camera != self and 
                not self.is_fullscreen):
                other = swap_parent.selected_camera
                print(f"DEBUG: SWAP {other.widget_id} ↔ {self.widget_id}")
                self.do_swap(other, self, swap_parent)
                other.reset_style()
                swap_parent.selected_camera = None
                self._reset_mouse_state()
                return True

            # PRIORITY 3: Long hold → enter swap mode
            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                print(f"DEBUG: ENTER swap {self.widget_id}")
                swap_parent.selected_camera = self
                self.video_label.setStyleSheet(self.swap_ready_style)
                self._reset_mouse_state()
                return True

            # PRIORITY 4: Short click → fullscreen
            print(f"DEBUG: Short click fullscreen {self.widget_id}")
            self.toggle_fullscreen()
            
        except Exception:
            traceback.print_exc()
        finally:
            self._reset_mouse_state()
        return True

    def _reset_mouse_state(self):
        """Clear temporary mouse tracking."""
        self._press_time = 0
        self._press_widget_id = None
        self._grid_parent = None

    def do_swap(self, source, target, layout_parent):
        """
        Atomic swap of two widgets in grid layout.
        Updates grid_position so fullscreen works correctly.
        """
        try:
            source_pos = getattr(source, 'grid_position', None)
            target_pos = getattr(target, 'grid_position', None)
            if source_pos is None or target_pos is None:
                print(f"DEBUG: Swap failed - missing positions")
                return

            layout = layout_parent.layout()
            # Remove both, add swapped = ATOMIC operation
            layout.removeWidget(source)
            layout.removeWidget(target)
            layout.addWidget(target, *source_pos)
            layout.addWidget(source, *target_pos)
            # Update memory
            source.grid_position, target.grid_position = target_pos, source_pos
            print(f"DEBUG: Swap complete {source.widget_id} ↔ {target.widget_id}")
        except Exception:
            traceback.print_exc()

    def toggle_fullscreen(self):
        """Enter OR exit fullscreen."""
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.go_fullscreen()

    def go_fullscreen(self):
        """Detach widget, make fullscreen window."""
        if self.is_fullscreen:
            return
        try:
            print(f"DEBUG: {self.widget_id} → fullscreen")
            # Remember original camera widget configuration
            self._saved_parent = self.parent()
            self._saved_position = getattr(self, 'grid_position', None)
            
            if self._saved_parent and self._saved_parent.layout():
                self._saved_parent.layout().removeWidget(self)

            # Become standalone fullscreen window
            self.setParent(None)
            self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
            self.showFullScreen()
            self.is_fullscreen = True
        except Exception:
            traceback.print_exc()

    def exit_fullscreen(self):
        """Return to exact grid position."""
        if not self.is_fullscreen:
            return
        try:
            print(f"DEBUG: {self.widget_id} ← grid[{self._saved_position}]")
            
            # Must reset flags BEFORE reparenting
            self.setWindowFlags(Qt.WindowType.Widget)
            self.show()  # Show as normal widget first
            
            # Return to exact spot
            if self._saved_parent and self._saved_position:
                self.setParent(self._saved_parent)
                layout = self._saved_parent.layout()
                if layout:
                    layout.addWidget(self, *self._saved_position)
            
            self.is_fullscreen = False
            
            # Restore main window fullscreen
            if self._saved_parent and self._saved_parent.window():
                self._saved_parent.window().showFullScreen()
        except Exception:
            traceback.print_exc()

    @pyqtSlot(object)
    def on_frame(self, frame):
        """
        Worker thread tells us "camera alive=True/False"  
        @pyqtSlot makes sure we update GUI on main thread
        
        Mental model:
        Camera dies → Worker: "online=False" ──signal───> on_status_changed()
        Camera back → Worker: "online=True" ──signal───> on_status_changed()
                                            │
                                    Updates border/clear on MAIN thread
        """
        # Called by worker thread signal. Runs on main GUI thread.
        # Converts OpenCV frame → Qt image → display.
        try:
            if frame is None:
                return
                
            # Resize to fit
            if self.is_fullscreen:
                w, h = self.width(), self.height()
                if w > 0 and h > 0:
                    frame_resized = cv2.resize(frame, (w, h))
                else:
                    return
            else:
                frame_resized = cv2.resize(frame, (self.screen_width, self.screen_height))

            # BGR → RGB, then Qt image format
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            img = QtGui.QImage(frame_rgb.data.tobytes(), w, h, bytes_per_line, 
                             QtGui.QImage.Format.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_label.setPixmap(pix)
            self.frame_count += 1  # For FPS
        except Exception:
            pass

    @pyqtSlot(bool)
    def on_status_changed(self, online):
        """
        Worker thread tells us "camera alive=True/False"  
        @pyqtSlot makes sure we update GUI on main thread
        
        Mental model:
        Camera dies → Worker: "online=False" ──signal───> on_status_changed()
        Camera back → Worker: "online=True" ──signal───> on_status_changed()
                                            │
                                    Updates border/clear on MAIN thread
        """
        # Camera connected/disconnected
        if online:
            self.setStyleSheet(self.normal_style)
        else:
            self.video_label.clear()  # Remove stale frame

    def reset_style(self):
        # Remove yellow swap border
        self.video_label.setStyleSheet("")
        self.setStyleSheet(self.normal_style)

    def _print_fps(self):
        try:
            now = time.time()
            elapsed = now - self.prev_time
            if elapsed >= 1.0:
                fps = self.frame_count / elapsed if elapsed > 0 else 0.0
                print(f"DEBUG: {self.widget_id} FPS: {fps:.1f}")
                self.frame_count = 0
                self.prev_time = now
        except:
            pass

    def cleanup(self):
        """Stop camera thread safely."""
        try:
            if hasattr(self, 'worker') and self.worker:
                self.worker.stop()
        except:
            pass

# === HELPER FUNCTIONS ===
def get_smart_grid(num_cameras):
    """Calculate best rows/cols for N cameras. Max 9."""
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

def get_video_indexes():
    """Get all /dev/videoX indexes."""
    video_devices = glob.glob('/dev/video*')
    indexes = []
    
    for device in sorted(video_devices):
        index = int(device.split('video')[-1])
        indexes.append(index)
    
    return indexes

def find_working_cameras():
    """Test /dev/video0-4 + V4L2 devices."""
    # Fix for glob pattern - explicitly loop over devices
    video_devices = glob.glob('/dev/video*')
    if not video_devices:
        print("No /dev/video* devices found!")
        return []
    
    for device in video_devices:
        device_path = device  # /dev/video0, /dev/video1, etc.
        subprocess.run(f"sudo fuser -vk {device_path}", shell=True)
    
    indexes = get_video_indexes()
    working = []
    for cam_index in indexes: 
        print(cam_index)
        cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            print("Cannot open camera")
            continue
        else:
            working.append(cam_index)
        cap.release()
        cv2.destroyAllWindows()
    return working




def safe_cleanup(widgets):
    """Stop all camera threads."""
    print("DEBUG: Cleaning all cameras")
    for w in widgets[:]:
        try:
            w.cleanup()
        except:
            pass

# === MAIN APPLICATION ===
def main():
    """Create fullscreen grid of working cameras."""
    print("DEBUG: Starting camera grid app")
    app = QtWidgets.QApplication(sys.argv)
    camera_widgets = []

    # Clean shutdown handlers
    def on_sigint(sig, frame):
        safe_cleanup(camera_widgets)
        sys.exit(0)
    signal.signal(signal.SIGINT, on_sigint)
    atexit.register(lambda: safe_cleanup(camera_widgets))

    # Dark theme (fallback to Fusion)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    app.setStyleSheet("QWidget { background: #2b2b2b; color: #ffffff; }")

    # Frameless fullscreen main window
    mw = QtWidgets.QMainWindow()
    mw.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
    central_widget = QtWidgets.QWidget()
    central_widget.selected_camera = None  # Swap mode manager
    mw.setCentralWidget(central_widget)
    mw.showFullScreen()

    # Screen size and camera detection
    screen = app.primaryScreen().availableGeometry()
    working_cameras = find_working_cameras()
    print(f"DEBUG: Found {len(working_cameras)} cameras")

    # Grid layout
    layout = QtWidgets.QGridLayout(central_widget)
    layout.setContentsMargins(10,10,10,10)
    layout.setSpacing(10)

    if working_cameras:
        rows, cols = get_smart_grid(len(working_cameras))
        widget_width = screen.width() // cols
        widget_height = screen.height() // rows
        
        # Create widgets (max 9)
        for cam_index in working_cameras[:9]:
            cw = CameraWidget(widget_width, widget_height, cam_index, parent=central_widget)
            camera_widgets.append(cw)

        # Position in grid
        for i, cw in enumerate(camera_widgets):
            row = i // cols
            col = i % cols
            cw.grid_position = (row, col)
            layout.addWidget(cw, row, col)
    else:
        # No cameras message
        label = QtWidgets.QLabel("NO CAMERAS FOUND")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: #888;")
        layout.addWidget(label, 0, 0)

    # Quit handlers
    app.aboutToQuit.connect(lambda: safe_cleanup(camera_widgets))
    QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Q'), mw, 
                   lambda: (safe_cleanup(camera_widgets), app.quit()))

    print("DEBUG: Short click=fullscreen toggle. Hold 400ms=swap mode. Ctrl+Q=quit.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
