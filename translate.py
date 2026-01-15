"""
Multi-camera grid viewer - COMPLETE PyQt5 Raspberry Pi version
FIXES: QShortcut error + RPi camera detection + APT safe
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
import sys
import cv2
import time 
import traceback
from collections import deque
import atexit
import signal

# SAFE OPTIONAL IMPORTS
try:
    from cv2_enumerate_cameras import enumerate_cameras
    ENUM_AVAILABLE = True
except ImportError:
    ENUM_AVAILABLE = False
    print("INFO: cv2_enumerate_cameras not found - using basic detection")

try:
    import qdarkstyle
    DARKSTYLE_AVAILABLE = True
except ImportError:
    DARKSTYLE_AVAILABLE = False
    print("INFO: qdarkstyle not found - using Fusion")

try:
    import imutils
    IMUTILS_AVAILABLE = True
except ImportError:
    IMUTILS_AVAILABLE = False
    print("INFO: imutils not found - using cv2")

# CAMERA THREAD
class CaptureWorker(QThread):
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(bool)

    def __init__(self, stream_link, parent=None, maxlen=4):
        super().__init__(parent)
        self.stream_link = stream_link
        self._running = True
        self._reconnect_backoff = 1.0
        self._cap = None
        self.buffer = deque(maxlen=maxlen)

    def run(self):
        print(f"DEBUG: Camera {self.stream_link} thread started")
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._open_capture()
                    if not (self._cap and self._cap.isOpened()):
                        time.sleep(self._reconnect_backoff)
                        self._reconnect_backoff = min(self._reconnect_backoff * 1.5, 10.0)
                        continue
                    self.status_changed.emit(True)

                status, frame = self._cap.read()
                if not status or frame is None:
                    self._close_capture()
                    self.status_changed.emit(False)
                    continue

                self.buffer.append(frame)
                self.frame_ready.emit(frame)
                time.sleep(0.01)  # RPi CPU friendly
                
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)
        
        self._close_capture()

    def _open_capture(self):
        """RPi-optimized backends."""
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_GSTREAMER, cv2.CAP_FFMPEG]
        for api in backends:
            try:
                cap = cv2.VideoCapture(self.stream_link, api)
                if cap.isOpened():
                    print(f"SUCCESS: Camera {self.stream_link} with backend {api}")
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    self._cap = cap
                    return
                cap.release()
            except:
                pass

    def _close_capture(self):
        try:
            if self._cap:
                self._cap.release()
                self._cap = None
        except:
            pass

    def stop(self):
        self._running = False
        self.wait(timeout=2000)
        self._close_capture()

# CAMERA WIDGET
class CameraWidget(QtWidgets.QWidget):
    hold_threshold_ms = 400

    def __init__(self, width, height, stream_link=0, aspect_ratio=False, parent=None, buffer_size=4):
        super().__init__(parent)
        print(f"DEBUG: Creating camera {stream_link}")
        
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        
        self.screen_width = max(1, width)
        self.screen_height = max(1, height)
        self.camera_stream_link = stream_link
        
        self.widget_id = f"cam{stream_link}_{id(self)}"
        self.is_fullscreen = False
        self.grid_position = None
        self._saved_parent = None
        self._saved_position = None
        self._press_widget_id = None
        self._press_time = 0
        self._grid_parent = None

        self.normal_style = "border: 2px solid #555; background: black;"
        self.swap_ready_style = "border: 4px solid #FFFF00; background: black;"
        self.setStyleSheet(self.normal_style)
        self.setObjectName(self.widget_id)

        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video_label.setMinimumSize(1, 1)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setMouseTracking(True)
        self.video_label.setObjectName(f"{self.widget_id}_label")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        self.frame_count = 0
        self.prev_time = time.time()

        self.worker = CaptureWorker(stream_link, parent=self, maxlen=buffer_size)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.start()

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(1000)
        self.ui_timer.timeout.connect(self._print_fps)
        self.ui_timer.start()

        self.installEventFilter(self)
        self.video_label.installEventFilter(self)
        print(f"DEBUG: Widget {self.widget_id} ready")

    def eventFilter(self, obj, event):
        if obj not in (self, self.video_label):
            return super().eventFilter(obj, event)
            
        if event.type() == QtCore.QEvent.MouseButtonPress:
            return self._on_mouse_press(event)
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            return self._on_mouse_release(event)
        return super().eventFilter(obj, event)

    def _on_mouse_press(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self._press_time = time.time() * 1000.0
                self._press_widget_id = self.widget_id
                self._grid_parent = self.parent()
                print(f"DEBUG: Press {self.widget_id}")
            elif event.button() == Qt.RightButton:
                self.toggle_fullscreen()
        except Exception:
            traceback.print_exc()
        return True

    def _on_mouse_release(self, event):
        try:
            if (event.button() != Qt.LeftButton or 
                not self._press_widget_id or self._press_widget_id != self.widget_id):
                return True

            hold_time = (time.time() * 1000.0) - self._press_time
            print(f"DEBUG: Release {self.widget_id}, hold={hold_time:.0f}ms")

            swap_parent = self._grid_parent
            if not swap_parent or not hasattr(swap_parent, 'selected_camera'):
                self._reset_mouse_state()
                self.toggle_fullscreen()
                return True

            if swap_parent.selected_camera == self:
                print(f"DEBUG: Clear swap {self.widget_id}")
                swap_parent.selected_camera = None
                self.reset_style()
                self._reset_mouse_state()
                return True

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

            if hold_time >= self.hold_threshold_ms and not self.is_fullscreen:
                print(f"DEBUG: ENTER swap {self.widget_id}")
                swap_parent.selected_camera = self
                self.video_label.setStyleSheet(self.swap_ready_style)
                self._reset_mouse_state()
                return True

            print(f"DEBUG: Short click fullscreen {self.widget_id}")
            self.toggle_fullscreen()
            
        except Exception:
            traceback.print_exc()
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
                print(f"DEBUG: Swap failed - missing positions")
                return

            layout = layout_parent.layout()
            layout.removeWidget(source)
            layout.removeWidget(target)
            layout.addWidget(target, *source_pos)
            layout.addWidget(source, *target_pos)
            source.grid_position, target.grid_position = target_pos, source_pos
            print(f"DEBUG: Swap complete {source.widget_id} ↔ {target.widget_id}")
        except Exception:
            traceback.print_exc()

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.go_fullscreen()

    def go_fullscreen(self):
        if self.is_fullscreen:
            return
        try:
            print(f"DEBUG: {self.widget_id} → fullscreen")
            self._saved_parent = self.parent()
            self._saved_position = getattr(self, 'grid_position', None)
            
            if self._saved_parent and self._saved_parent.layout():
                self._saved_parent.layout().removeWidget(self)

            self.setParent(None)
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            self.showFullScreen()
            self.is_fullscreen = True
        except Exception:
            traceback.print_exc()

    def exit_fullscreen(self):
        if not self.is_fullscreen:
            return
        try:
            print(f"DEBUG: {self.widget_id} ← grid[{self._saved_position}]")
            
            self.setWindowFlags(Qt.Widget)
            self.show()
            
            if self._saved_parent and self._saved_position:
                self.setParent(self._saved_parent)
                layout = self._saved_parent.layout()
                if layout:
                    layout.addWidget(self, *self._saved_position)
            
            self.is_fullscreen = False
            
            if self._saved_parent and self._saved_parent.window():
                self._saved_parent.window().showFullScreen()
        except Exception:
            traceback.print_exc()

    @pyqtSlot(object)
    def on_frame(self, frame):
        try:
            if frame is None:
                return
                
            if self.is_fullscreen:
                w, h = self.width(), self.height()
                if w > 0 and h > 0:
                    frame_resized = cv2.resize(frame, (w, h))
                else:
                    return
            else:
                frame_resized = cv2.resize(frame, (self.screen_width, self.screen_height))

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            img = QtGui.QImage(frame_rgb.data.tobytes(), w, h, bytes_per_line, 
                             QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_label.setPixmap(pix)
            self.frame_count += 1
        except Exception:
            pass

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
                print(f"DEBUG: {self.widget_id} FPS: {fps:.1f}")
                self.frame_count = 0
                self.prev_time = now
        except:
            pass

    def cleanup(self):
        try:
            if hasattr(self, 'worker') and self.worker:
                self.worker.stop()
        except:
            pass

# HELPER FUNCTIONS
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

def find_working_cameras():
    """RPi-optimized camera detection."""
    working = []
    test_indices = [0,1,2,3,4]
    
    print("=== SCANNING CAMERAS ===")
    for i in test_indices:
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    print(f"✅ CAMERA {i} OK ({frame.shape})")
                    working.append(i)
                else:
                    print(f"❌ CAMERA {i} opened but no frames")
            else:
                print(f"❌ NO /dev/video{i}")
        except Exception as e:
            print(f"❌ CAMERA {i} error: {e}")
    
    print(f"FOUND {len(working)} working cameras: {working}")
    return working

def safe_cleanup(widgets):
    print("DEBUG: Cleaning all cameras")
    for w in widgets[:]:
        try:
            w.cleanup()
        except:
            pass

# MAIN APPLICATION
def main():
    print("DEBUG: Starting RPi camera grid app")
    app = QtWidgets.QApplication(sys.argv)
    camera_widgets = []

    def on_sigint(sig, frame):
        safe_cleanup(camera_widgets)
        sys.exit(0)
    signal.signal(signal.SIGINT, on_sigint)
    atexit.register(lambda: safe_cleanup(camera_widgets))

    # Theme
    if DARKSTYLE_AVAILABLE:
        try:
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
            print("DEBUG: Dark theme loaded")
        except:
            app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    else:
        app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    # Main window
    mw = QtWidgets.QMainWindow()
    mw.setWindowFlags(Qt.FramelessWindowHint)
    central_widget = QtWidgets.QWidget()
    central_widget.selected_camera = None
    mw.setCentralWidget(central_widget)
    mw.showFullScreen()

    # Camera detection
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
        
        for cam_index in working_cameras[:9]:
            cw = CameraWidget(widget_width, widget_height, cam_index, parent=central_widget)
            camera_widgets.append(cw)

        for i, cw in enumerate(camera_widgets):
            row = i // cols
            col = i % cols
            cw.grid_position = (row, col)
            layout.addWidget(cw, row, col)
    else:
        label = QtWidgets.QLabel("NO CAMERAS FOUND\n\nRun: ls /dev/video*\nPlug USB webcams")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: #888;")
        layout.addWidget(label, 0, 0)

    # FIXED QShortcut
    QShortcut(QKeySequence('Ctrl+Q'), mw, 
              lambda: (safe_cleanup(camera_widgets), app.quit()))

    print("DEBUG: Short click=fullscreen. Hold 400ms=swap. Ctrl+Q=quit.")
    print("DEBUG: Check: ls /dev/video*")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
