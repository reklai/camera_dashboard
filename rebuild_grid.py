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

class CameraWidget(QtWidgets.QWidget):
    def __init__(self, width, height, stream_link=0, aspect_ratio=False, parent=None, deque_size=1):
        super().__init__(parent)
        
        self.deque = deque(maxlen=deque_size)
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio
        self.camera_stream_link = stream_link
        self.online = False
        self.capture = None
        
        # Fullscreen state tracking
        self.is_fullscreen = False
        self.original_parent = parent
        self.original_geometry = None
        self.grid_position = None
        
        self.video_frame = QtWidgets.QLabel(self)
        self.video_frame.setStyleSheet("border: 2px solid #555; background: black;")
        self.video_frame.setMouseTracking(True)
        self.video_frame.mousePressEvent = self.toggle_fullscreen
        self.video_frame.setScaledContents(True)
        
        # FPS tracking
        self.frame_count = 0
        self.prev_time = time.time()
        
        # Test camera first
        self.test_and_init_camera()
        
        # Start frame grabbing thread
        self.frame_thread = Thread(target=self.get_frame, daemon=True)
        self.frame_thread.start()
        
        # Start display update thread (30 FPS)
        self.display_thread = Thread(target=self.update_display_loop, daemon=True)
        self.display_thread.start()

        print(f'Started camera {self.camera_stream_link}: {"✅" if self.online else "❌"}')

    def toggle_fullscreen(self, event):
        """Toggle fullscreen on video frame click"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if not self.is_fullscreen:
                self.go_fullscreen()
            else:
                self.exit_fullscreen()

    def go_fullscreen(self):
        """Enter fullscreen mode"""
        if self.is_fullscreen:
            return
        
        # Store original state
        self.original_geometry = self.video_frame.geometry()
        self.original_parent = self.video_frame.parent()
        
        # Find grid position from layout
        layout = self.original_parent.layout() if self.original_parent else None
        if layout:
            for row in range(layout.rowCount()):
                for col in range(layout.columnCount()):
                    item = layout.itemAtPosition(row, col)
                    if item and item.widget() == self.video_frame:
                        self.grid_position = (row, col)
                        break
        
        # Remove from layout
        if layout:
            layout.removeWidget(self.video_frame)
        
        # Make standalone fullscreen window
        self.video_frame.setParent(None)
        self.video_frame.setWindowFlags(
            QtCore.Qt.WindowType.Window | 
            QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.video_frame.setWindowState(QtCore.Qt.WindowState.WindowFullScreen)
        self.video_frame.show()
        
        # Update sizing for fullscreen
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.is_fullscreen = True

    def exit_fullscreen(self):
        """Exit fullscreen mode"""
        if not self.is_fullscreen:
            return
    
        # Restore window properties
        self.video_frame.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.video_frame.setWindowState(QtCore.Qt.WindowState.WindowNoState)
        
        # Reparent to original parent
        if self.original_parent:
            self.video_frame.setParent(self.original_parent)
            layout = self.original_parent.layout()
            if layout and self.grid_position:
                layout.addWidget(self.video_frame, *self.grid_position)
            else:
                layout.addWidget(self.video_frame)
        
        self.video_frame.show()
        self.is_fullscreen = False
        self.original_parent.window().showFullScreen()

    def test_and_init_camera(self):
        # Try MJPG first, then default
        for api in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            self.capture = cv2.VideoCapture(self.camera_stream_link, api)
            if self.capture.isOpened():
                # Set MJPG codec for better performance
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.online = True
                print(f"Camera {self.camera_stream_link} working with {api} (MJPG)")
                return
        print(f"❌ Camera {self.camera_stream_link} failed")

    def spin(self, seconds):
        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()

    def get_frame(self):
        """Background frame capture"""
        while True:
            try:
                if self.capture and self.capture.isOpened() and self.online:
                    status, frame = self.capture.read()
                    if status:
                        self.deque.append(frame)
                    else:
                        self.capture.release()
                        self.online = False
                else:
                    print(f'Reconnecting camera {self.camera_stream_link}...')
                    self.video_frame.clear()
                    self.test_and_init_camera()
                    self.spin(2)
                self.spin(0.001)
            except Exception as e:
                print(f"Camera {self.camera_stream_link} thread error: {e}")
                self.spin(0.1)

    def update_display_loop(self):
        """30 FPS display update (33ms interval) + FPS counter"""
        while True:
            if self.online and self.deque:
                self.set_frame()
                self.update_fps()
            else:
                self.video_frame.clear()
            self.spin(0.033)  # 33ms = ~30 FPS

    def update_fps(self):
        """Calculate and print FPS every second"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.prev_time >= 1.0:
            fps = self.frame_count / (current_time - self.prev_time)
            print(f"📹 Camera {self.camera_stream_link} FPS: {fps:.1f}")
            self.frame_count = 0
            self.prev_time = current_time

    def set_frame(self):
        """Display frame"""
        if not self.online or not self.deque:
            return

        frame = self.deque[-1]
        
        if self.maintain_aspect_ratio:
            self.frame = imutils.resize(frame, width=self.screen_width)
        else:
            self.frame = cv2.resize(frame, (self.screen_width, self.screen_height))

        # Convert BGR to RGB and create QImage
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        self.img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        self.pix = QtGui.QPixmap.fromImage(self.img)
        self.video_frame.setPixmap(self.pix)

    def get_video_frame(self):
        return self.video_frame

def get_smart_grid(num_cameras):
    # Grid Layout
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

def exit_application():
    QtWidgets.QApplication.quit()

# Main application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Dark theme
    try:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    except:
        app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    
    # Main window
    mw = QtWidgets.QMainWindow()
    mw.setWindowTitle('Dynamic Multi-Camera GUI - Click to Fullscreen')
    mw.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
    
    central_widget = QtWidgets.QWidget()
    mw.setCentralWidget(central_widget)
    mw.showFullScreen()

    # Screen dimensions
    screen = app.primaryScreen().availableGeometry()
    
    # Find working cameras
    print("Scanning for working cameras...")
    working_cameras = []
    
    test_indices = [0, 1, 2, 3, 4] + [cam.index for cam in enumerate_cameras(cv2.CAP_V4L2)]
    
    for i in test_indices:
        if i in working_cameras: continue
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            working_cameras.append(i)
            cap.release()
            print(f"Found working camera: /dev/video{i}")
        else:
            cap.release()

    # Create dynamic layout
    layout = QtWidgets.QGridLayout(central_widget)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)
    
    if working_cameras:
        # Dynamic sizing based on grid
        rows, cols = get_smart_grid(len(working_cameras))
        widget_width = screen.width() // cols
        widget_height = screen.height() // rows
        
        # Create camera widgets
        camera_widgets = []
        for cam_index in working_cameras[:9]:
            cam_widget = CameraWidget(widget_width, widget_height, cam_index, parent=central_widget)
            camera_widgets.append(cam_widget)

        # Place in smart grid
        rows, cols = get_smart_grid(len(camera_widgets))
        print(f"📺 {len(camera_widgets)} cameras in {rows}×{cols} grid")
        
        for i, cam_widget in enumerate(camera_widgets):
            row = i // cols
            col = i % cols
            cam_widget.grid_position = (row, col)  # Store position for restore
            layout.addWidget(cam_widget.get_video_frame(), row, col)
    else:
        label = QtWidgets.QLabel("No working cameras found")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: #888;")
        layout.addWidget(label, 0, 0)

    # Exit shortcut
    QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Q'), mw, exit_application)

    sys.exit(app.exec())

