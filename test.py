from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from cv2_enumerate_cameras import enumerate_cameras
import qdarkstyle
from threading import Thread
from collections import deque
import time
import sys
import cv2
import imutils
import signal
import atexit

class CameraWidget(QtWidgets.QWidget):
    # Camera display widget with fullscreen and swap
    
    def __init__(self, width, height, stream_link=0, aspect_ratio=False, parent=None, deque_size=1):
        super().__init__(parent)
        
        # Video buffer and sizes
        self.deque = deque(maxlen=deque_size)
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio
        self.camera_stream_link = stream_link
        self.online = False
        self.capture = None
        self._threads_running = True
        
        # UI state
        self.is_fullscreen = False
        self.original_parent = parent
        self.grid_position = None
        self.selected_for_swap = False
        self.hold_start_time = 0
        self.is_holding = False
        self.hold_threshold = 400  # ms for swap mode
        self.click_parent = None   # Fix swap mode race condition
        
        # Styles
        self.normal_style = "border: 2px solid #555; background: black;"
        self.swap_ready_style = "border: 4px solid #FFFF00; background: black;"
        
        # FPS counter
        self.frame_count = 0
        self.prev_time = time.time()
        
        # Create video label
        self.video_frame = None
        self._setup_video_frame()
        
        # Start camera threads
        self.test_and_init_camera()
        self.frame_thread = Thread(target=self._safe_get_frame, daemon=True)
        self.frame_thread.start()
        self.display_thread = Thread(target=self._safe_update_display, daemon=True)
        self.display_thread.start()

    def _setup_video_frame(self):
        # Setup QLabel for video display
        self.video_frame = QtWidgets.QLabel(self)
        self.video_frame.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.video_frame.setMinimumSize(1, 1)
        self.reset_style()
        self.video_frame.setMouseTracking(True)
        self.video_frame.mousePressEvent = self.on_mouse_press
        self.video_frame.mouseReleaseEvent = self.on_mouse_release
        self.video_frame.setScaledContents(True)

    def cleanup(self):
        # Stop threads and release camera
        self._threads_running = False
        if self.capture:
            self.capture.release()
            self.capture = None
        self.online = False
        self.deque.clear()

    def reset_style(self):
        # Reset border to normal
        if self.video_frame:
            self.video_frame.setStyleSheet(self.normal_style)
        self.selected_for_swap = False

    def on_mouse_press(self, event):
        # Store parent at press time (fixes swap race condition)
        try:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.click_parent = self.original_parent or self.parent()
                self.hold_start_time = time.time() * 1000
                self.is_holding = True
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.toggle_fullscreen()
        except:
            pass

    def on_mouse_release(self, event):
        # Handle quick click (fullscreen) vs long hold (swap)
        try:
            if not (event.button() == QtCore.Qt.MouseButton.LeftButton and self.is_holding):
                return
                
            hold_time = (time.time() * 1000) - self.hold_start_time
            parent = self.click_parent
            if not parent:
                return
                
            if hasattr(parent, 'selected_camera') and parent.selected_camera:
                # Swap with selected camera
                if parent.selected_camera != self:
                    self.do_swap(parent.selected_camera, self)
                parent.selected_camera.reset_style()
                parent.selected_camera = None
                print(f"SWAP OK")
            elif hold_time >= self.hold_threshold:
                # Enter swap mode
                parent.selected_camera = self
                if self.video_frame:
                    self.video_frame.setStyleSheet(self.swap_ready_style)
                print(f"CAM {self.camera_stream_link} SWAP READY")
            else:
                # Quick click = fullscreen
                self.toggle_fullscreen()
                
            self.is_holding = False
            self.click_parent = None
        except:
            pass

    def do_swap(self, source, target):
        # Swap grid positions between two cameras
        try:
            source_pos = getattr(source, 'grid_position', None)
            target_pos = getattr(target, 'grid_position', None)
            if not source_pos or not target_pos:
                return
                
            layout_parent = getattr(source, 'original_parent', None) or getattr(source, 'click_parent', None)
            if not layout_parent or not hasattr(layout_parent, 'layout'):
                return
                
            layout = layout_parent.layout()
            layout.removeWidget(source.video_frame)
            layout.removeWidget(target.video_frame)
            layout.addWidget(target.video_frame, *source_pos)
            layout.addWidget(source.video_frame, *target_pos)
            
            source.grid_position = target_pos
            target.grid_position = source_pos
        except:
            pass

    def toggle_fullscreen(self):
        # Switch fullscreen on/off
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.go_fullscreen()

    def go_fullscreen(self):
        # Remove from grid, make fullscreen window
        if self.is_fullscreen or not self.video_frame:
            return
        try:
            self.original_parent = self.video_frame.parent()
            if not self.original_parent or not hasattr(self.original_parent, 'layout'):
                return
                
            layout = self.original_parent.layout()
            # Find current grid position
            for row in range(layout.rowCount()):
                for col in range(layout.columnCount()):
                    item = layout.itemAtPosition(row, col)
                    if item and item.widget() == self.video_frame:
                        self.grid_position = (row, col)
                        break
            
            layout.removeWidget(self.video_frame)
            self.video_frame.setParent(None)
            self.video_frame.setWindowFlags(QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.FramelessWindowHint)
            self.video_frame.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)
            self.video_frame.show()
            self.is_fullscreen = True
        except:
            pass

    def exit_fullscreen(self):
        # Return to grid position
        if not self.is_fullscreen or not self.video_frame:
            return
        try:
            self.video_frame.setWindowFlags(QtCore.Qt.WindowType.Widget)
            self.video_frame.setWindowState(QtCore.Qt.WindowState.WindowNoState)
            
            if not self.original_parent or not hasattr(self.original_parent, 'layout'):
                return
                
            self.video_frame.setParent(self.original_parent)
            layout = self.original_parent.layout()
            if self.grid_position:
                layout.addWidget(self.video_frame, *self.grid_position)
            
            self.video_frame.show()
            self.is_fullscreen = False
            # Restore main window fullscreen
            main_window = self.original_parent.window()
            if main_window:
                main_window.showFullScreen()
        except:
            pass

    def test_and_init_camera(self):
        # Try V4L2 then ANY backend
        try:
            for api in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                self.capture = cv2.VideoCapture(self.camera_stream_link, api)
                if self.capture.isOpened():
                    self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    self.online = True
                    print(f"CAM {self.camera_stream_link} OK")
                    return
            print(f"CAM {self.camera_stream_link} FAILED")
        except:
            pass

    def _safe_spin(self, seconds):
        # Thread-safe processEvents
        try:
            time_end = time.time() + seconds
            while time.time() < time_end and self._threads_running:
                QtWidgets.QApplication.processEvents()
        except:
            pass

    def _safe_get_frame(self):
        # Capture frames from camera
        while self._threads_running:
            try:
                if self.capture and self.capture.isOpened() and self.online:
                    status, frame = self.capture.read()
                    if status and frame is not None:
                        self.deque.append(frame)
                    else:
                        self.capture.release()
                        self.capture = None
                        self.online = False
                else:
                    if self._threads_running:
                        print(f'RECONNECT {self.camera_stream_link}')
                        self.test_and_init_camera()
                        self._safe_spin(2)
                self._safe_spin(0.001)
            except:
                self._safe_spin(0.1)

    def _safe_update_display(self):
        # Update video display 30fps
        while self._threads_running:
            try:
                if self.online and self.deque:
                    self.set_frame()
                    self.update_fps()
                elif self.video_frame:
                    self.video_frame.clear()
                self._safe_spin(0.033)
            except:
                self._safe_spin(0.1)

    def update_fps(self):
        # Print FPS every second
        try:
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.prev_time >= 1.0:
                fps = self.frame_count / (current_time - self.prev_time)
                print(f"CAM {self.camera_stream_link} FPS {fps:.1f}")
                self.frame_count = 0
                self.prev_time = current_time
        except:
            pass

    def set_frame(self):
        # Convert OpenCV frame to Qt pixmap
        try:
            if (not self.online or not self.deque or not hasattr(self, 'video_frame') or 
                not self.video_frame or not self.video_frame.isEnabled()):
                return
                
            frame = self.deque[-1]
            if frame is None:
                return
            
            # Resize for fullscreen or grid
            if self.is_fullscreen:
                w, h = self.video_frame.width(), self.video_frame.height()
                if w > 0 and h > 0:
                    frame = cv2.resize(frame, (w, h))
                else:
                    return
            elif self.maintain_aspect_ratio:
                frame = imutils.resize(frame, width=self.screen_width)
            else:
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            
            self.img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            self.pix = QtGui.QPixmap.fromImage(self.img)
            
            if self.video_frame and self.video_frame.isEnabled():
                self.video_frame.setPixmap(self.pix)
        except:
            pass

    def get_video_frame(self):
        # Return video QLabel for layout
        return getattr(self, 'video_frame', None)

def get_smart_grid(num_cameras):
    # Calculate optimal grid layout
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
    # Test camera indices for working devices
    print("SCAN CAMERAS")
    working_cameras = []
    test_indices = [0, 1, 2, 3, 4] + [cam.index for cam in enumerate_cameras(cv2.CAP_V4L2)]
    
    for i in test_indices:
        if i in working_cameras: continue
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                working_cameras.append(i)
                cap.release()
                print(f"FOUND /dev/video{i}")
            else:
                cap.release()
        except:
            pass
    return working_cameras

def emergency_cleanup(camera_widgets):
    # Close all cameras on exit
    print("CLEANUP")
    for widget in camera_widgets[:]:
        try:
            widget._threads_running = False
            if widget.capture:
                widget.capture.release()
        except:
            pass
    camera_widgets.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    camera_widgets = []
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        emergency_cleanup(camera_widgets)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(lambda: emergency_cleanup(camera_widgets))
    
    try:
        # Dark theme fallback
        try:
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
        except:
            app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        
        # Main frameless window
        mw = QtWidgets.QMainWindow()
        mw.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        
        central_widget = QtWidgets.QWidget()
        central_widget.selected_camera = None  # For swap mode
        mw.setCentralWidget(central_widget)
        mw.showFullScreen()
        
        # Grid layout
        screen = app.primaryScreen().availableGeometry()
        working_cameras = find_working_cameras()
        
        layout = QtWidgets.QGridLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        if working_cameras:
            rows, cols = get_smart_grid(len(working_cameras))
            widget_width = screen.width() // cols
            widget_height = screen.height() // rows
            
            # Create camera widgets
            for cam_index in working_cameras[:9]:
                cam_widget = CameraWidget(widget_width, widget_height, cam_index, parent=central_widget)
                camera_widgets.append(cam_widget)

            print(f"{len(camera_widgets)} cams {rows}x{cols}")
            
            # Add to grid
            for i, cam_widget in enumerate(camera_widgets):
                row = i // cols
                col = i % cols
                cam_widget.grid_position = (row, col)
                layout.addWidget(cam_widget.get_video_frame(), row, col)
        else:
            label = QtWidgets.QLabel("NO CAMERAS")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size: 24px; color: #888;")
            layout.addWidget(label, 0, 0)
        
        # Cleanup and quit
        app.aboutToQuit.connect(lambda: emergency_cleanup(camera_widgets))
        QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Q'), mw, lambda: (emergency_cleanup(camera_widgets), app.quit()))
        
        sys.exit(app.exec())
        
    except:
        emergency_cleanup(camera_widgets)

