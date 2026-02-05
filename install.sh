#!/usr/bin/env bash
set -euo pipefail

# Camera Dashboard installer for Raspberry Pi / Linux
# Run this script from the project root directory:
#   chmod +x install.sh
#   ./install.sh

# ---------- helper functions ----------

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo_section() {
  echo
  echo "========================================"
  echo "$1"
  echo "========================================"
}

# ---------- 0) basic checks ----------

if [[ "$EUID" -eq 0 ]]; then
  echo "Do NOT run this script as root or with sudo."
  echo "Use a normal user and only enter your password for sudo when prompted."
  exit 1
fi

if ! command_exists sudo; then
  echo "sudo is required but not found. Please install/configure sudo and try again."
  exit 1
fi

if ! command_exists python3; then
  echo "python3 is required but not found. Please install Python 3 and try again."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 1) update the system ----------

echo_section "1) Updating system packages (sudo apt update && sudo apt upgrade -y)"
sudo apt update
sudo apt upgrade -y

# ---------- 2) install system dependencies ----------

echo_section "2) Installing system dependencies (sudo apt install ...)"

# Core Python and Qt6 dependencies
# Note: OpenGL support is included in python3-pyqt6 (via QtOpenGLWidgets)
sudo apt install -y \
  python3 python3-pip python3-venv \
  python3-pyqt6 python3-opencv python3-pyudev python3-numpy \
  libgl1 libegl1 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
  libqt6gui6 libqt6widgets6 libqt6opengl6 \
  v4l-utils

# GStreamer for hardware-accelerated video capture (jpegdec pipeline)
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad || true

# ---------- 3) create virtual environment with system packages ----------

echo_section "3) Creating Python virtual environment (.venv with system-site-packages)"

# Remove old venv if it exists without system packages
if [[ -d ".venv" ]]; then
  if ! grep -q "include-system-site-packages = true" ".venv/pyvenv.cfg" 2>/dev/null; then
    echo "Removing old venv (missing system-site-packages)..."
    rm -rf .venv
  fi
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv --system-site-packages .venv
else
  echo ".venv already exists with system packages, reusing it."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ---------- 4) verify python packages ----------

echo_section "4) Verifying Python packages"

pip install --upgrade pip

# Install test dependencies
pip install --quiet pytest pytest-qt

# Test imports
if python3 -c "from PyQt6 import QtCore, QtGui, QtWidgets; from PyQt6.QtOpenGLWidgets import QOpenGLWidget; import cv2; import pyudev; import pytest; print('All imports OK')" 2>/dev/null; then
  echo "All required Python packages are available."
else
  echo "ERROR: Required Python packages not available!"
  echo "Please ensure python3-pyqt6, python3-opencv, and python3-pyudev are installed."
  exit 1
fi

# ---------- 5) fix camera permissions ----------

echo_section "5) Adding current user to 'video' group for camera access"

if ! groups "$USER" | grep -q "\bvideo\b"; then
  sudo usermod -aG video "$USER"
  echo "Added $USER to video group."
  echo "You may need to log out and back in (or reboot) for group changes to take effect."
else
  echo "$USER is already in the video group."
fi

echo "Camera devices:"
ls -l /dev/video* 2>/dev/null | head -10 || echo "No video devices found"

# ---------- 6) create logs directory ----------

echo_section "6) Creating logs directory"

mkdir -p "$SCRIPT_DIR/logs"
echo "Logs directory: $SCRIPT_DIR/logs"

# ---------- 7) install systemd service ----------

echo_section "7) Installing systemd service"

SERVICE_NAME="camera-dashboard.service"
SERVICE_SRC="$SCRIPT_DIR/${SERVICE_NAME}"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}"

# Generate service file with correct paths
cat > "$SERVICE_SRC" <<EOF
[Unit]
Description=Camera Dashboard
After=network.target graphical.target
Wants=graphical.target

[Service]
Type=notify
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/.venv/bin/python3 $SCRIPT_DIR/main.py
Restart=always
RestartSec=2
User=$USER
Group=$USER
Environment=DISPLAY=:0
Environment=PYTHONUNBUFFERED=1
Environment=CAMERA_DASHBOARD_CONFIG=$SCRIPT_DIR/config.ini
Environment=CAMERA_DASHBOARD_LOG_FILE=$SCRIPT_DIR/logs/camera_dashboard.log
Environment=QT_QPA_PLATFORM=xcb
WatchdogSec=15
NotifyAccess=main
Nice=-5
NoNewPrivileges=true

[Install]
WantedBy=graphical.target
EOF

echo "Generated service file with paths for user: $USER"
echo "  WorkingDirectory: $SCRIPT_DIR"
echo "  ExecStart: $SCRIPT_DIR/.venv/bin/python3 $SCRIPT_DIR/main.py"

if [[ -f "${SERVICE_SRC}" ]]; then
  echo "Installing ${SERVICE_NAME} to ${SERVICE_DST}"
  sudo cp "${SERVICE_SRC}" "${SERVICE_DST}"
  echo "Reloading systemd"
  sudo systemctl daemon-reload
  echo "Enabling ${SERVICE_NAME}"
  sudo systemctl enable "${SERVICE_NAME}"
  echo ""
  echo "Service installed but NOT started automatically."
  echo "To start the service, run:"
  echo "  sudo systemctl start ${SERVICE_NAME}"
  echo ""
  echo "To check status:"
  echo "  sudo systemctl status ${SERVICE_NAME}"
else
  echo "Service file not found: ${SERVICE_SRC}"
  echo "Skipping systemd install."
fi

# ---------- 8) quick test ----------

echo_section "8) Quick test"

if timeout 5 python3 -c "
from PyQt6 import QtWidgets
import cv2
import sys

# Test OpenCV can access a camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret:
        print('Camera test: OK (captured frame)')
    else:
        print('Camera test: WARNING (opened but no frame)')
else:
    print('Camera test: WARNING (could not open camera 0)')

print('Qt test: OK')
" 2>/dev/null; then
  echo "Quick test passed!"
else
  echo "Quick test had issues (this may be normal if no display is available)"
fi

echo_section "9) Finished"

cat <<EOF
Installation complete!

To run the app manually:

  cd $SCRIPT_DIR
  source .venv/bin/activate
  python3 main.py

To run via systemd service:

  sudo systemctl start camera-dashboard
  sudo systemctl status camera-dashboard

To run tests:

  ./test.sh           # Run all tests
  ./test.sh -v        # Verbose output
  ./test.sh -k "config"  # Run specific tests

To view logs:

  journalctl -u camera-dashboard -f
  # or
  tail -f logs/camera_dashboard.log

Notes:
- The app requires a display (X11/Wayland)
- Run as a normal user, not root
- Exit with Ctrl+Q or Q key, or Ctrl+C in terminal

EOF
