# Camera Dashboard

A multi-camera monitoring system optimized for Raspberry Pi, designed for blind-spot monitoring on cargo vehicles. Features real-time video streaming with GStreamer acceleration, dynamic performance tuning, hot-plug support.

### Note: Comprehensive Logging, Testing, and Performance Optimization was done through OPENCODE

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%20%7C%20Linux-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Features

### Multi-Camera Support

- Automatically detects and displays up to 3 USB cameras simultaneously
- Smart grid layout adapts to available camera count
- Hot-plug support - cameras can be connected/disconnected at runtime

### Interactive Interface

- **Touch/Mouse Controls**: Single tap/click for fullscreen, long press to swap positions
- **Swap Mode**: Reorganize camera layout with intuitive gestures
- **Night Mode**: Toggle enhanced visibility for low-light conditions
- **Settings Tile**: Quick access to restart, night mode, and exit

### Performance Optimization

- **GStreamer Pipeline**: Hardware-accelerated MJPEG decoding (with V4L2 fallback)
- **Dynamic FPS Adjustment**: Automatically reduces frame rate under CPU/thermal stress
- **Threaded Architecture**: Separate capture threads ensure smooth UI performance
- **Efficient Rendering**: 15 FPS UI refresh rate balances smoothness and CPU usage

### System Integration

- **Systemd Service**: Auto-start on boot with watchdog monitoring
- **Zero Configuration**: Works out of the box with standard USB cameras
- **Robust Recovery**: Automatic camera reconnection with exponential backoff

---

## System Requirements

### Supported Platforms

- **Raspberry Pi 5** (recommended)
- **Raspberry Pi 4** (64-bit OS recommended)
- **Linux** (Ubuntu 22.04+, Debian 12+)

### Hardware Tested Against

- USB webcams compatible with V4L2 (MJPEG support recommended)
- Minimum 2GB RAM (4GB+ recommended for 3 cameras)
- Display with X11 or Wayland

### Software Minimum Dependencies

- Python 3.8+
- PyQt6 (Qt6 GUI framework)
- OpenCV (with GStreamer support)
- pyudev (USB device detection)
- GStreamer 1.0 (optional, for optimized capture)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Reece-Reklai/camera_dashboard.git
cd camera_dashboard
chmod +x install.sh
./install.sh

# Run the application
source .venv/bin/activate
python3 main.py
```

## Installation

### Automated Installation (Recommended)

```bash
chmod +x install.sh
./install.sh
```

The installer will:

1. Update system packages
2. Install system dependencies (PyQt6, OpenCV, GStreamer)
3. Create a Python virtual environment with system-site-packages
4. Configure camera permissions (adds user to `video` group)
5. Install and enable the systemd service

### Manual Installation

<details>
<summary>Click to expand manual steps</summary>

#### 1. Update System

```bash
sudo apt update && sudo apt upgrade -y
```

#### 2. Install System Dependencies

```bash
sudo apt install -y \
  python3 python3-pip python3-venv \
  python3-pyqt6 python3-opencv python3-pyudev python3-numpy \
  libgl1 libegl1 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
  libqt6gui6 libqt6widgets6 v4l-utils \
  gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

#### 3. Create Virtual Environment

```bash
# Use --system-site-packages to access system PyQt6/OpenCV
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

#### 4. Fix Camera Permissions

```bash
sudo usermod -aG video $USER
# Log out and back in for changes to take effect
```

#### 5. Create Logs Directory

```bash
mkdir -p logs
```

</details>

---

## Usage

### Running the Application

```bash
# Manual run
source .venv/bin/activate
python3 main.py

# Or via systemd service
sudo systemctl start camera-dashboard
sudo systemctl status camera-dashboard
```

### Controls

| Action                   | Result                              |
| ------------------------ | ----------------------------------- |
| **Short Click/Tap**      | Toggle fullscreen view              |
| **Long Press (400ms+)**  | Enter swap mode (yellow border)     |
| **Click Another Camera** | Swap positions with selected camera |
| **Q or Ctrl+Q**          | Exit application                    |
| **Ctrl+C**               | Exit from terminal                  |

### Camera Status Indicators

| Border Color    | Status                         |
| --------------- | ------------------------------ |
| Gray            | Camera connected and streaming |
| Yellow          | Camera selected for swapping   |
| "DISCONNECTED"  | No camera detected in slot     |
| "CONNECTING..." | Camera being initialized       |

### Settings Tile (Top-Left)

- **Restart**: Restart the application
- **Nightmode**: Toggle night vision mode (red-tinted, enhanced brightness)
- **Exit**: Close the application

---

## Configuration Settings

### Config File: `config.ini`

```ini
[logging]
level = INFO                          # DEBUG, INFO, WARNING, ERROR
file = ./logs/camera_dashboard.log
max_bytes = 5242880                   # 5MB log rotation
backup_count = 3
stdout = true

[performance]
dynamic_fps = true                    # Auto-adjust FPS under stress
perf_check_interval_ms = 2000         # How often to check system load
min_dynamic_fps = 5                   # Minimum capture FPS
min_dynamic_ui_fps = 12               # Minimum UI render FPS
cpu_load_threshold = 0.75             # 75% CPU triggers FPS reduction
cpu_temp_threshold_c = 70.0           # 70°C triggers FPS reduction

[camera]
rescan_interval_ms = 15000            # Hot-plug detection interval (15s)
failed_camera_cooldown_sec = 30.0     # Retry delay for failed cameras
slot_count = 3                        # Number of camera slots
kill_device_holders = true            # Kill processes blocking cameras
use_gstreamer = true                  # Use GStreamer for capture (faster)

[profile]
capture_width = 640
capture_height = 480
capture_fps = 20                      # Camera capture rate
ui_fps = 15                           # UI refresh rate

[health]
log_interval_sec = 30                 # Health log frequency
```

### Environment Variables

```bash
# Override config file path
export CAMERA_DASHBOARD_CONFIG=/path/to/config.ini

# Override log file path
export CAMERA_DASHBOARD_LOG_FILE=/path/to/app.log
```

---

## Systemd Service

The installer automatically creates and enables a systemd service.

### Service Management

```bash
# Start/stop/restart
sudo systemctl start camera-dashboard
sudo systemctl stop camera-dashboard
sudo systemctl restart camera-dashboard

# Check status
sudo systemctl status camera-dashboard

# View logs
journalctl -u camera-dashboard -f

# Disable auto-start
sudo systemctl disable camera-dashboard
```

### Service Features

- **Auto-restart**: Restarts on crash (2-second delay)
- **Watchdog**: 15-second watchdog timeout with health pings
- **Nice Priority**: Runs at elevated priority (-5)
- **Security**: NoNewPrivileges enabled

---

## Performance Metrics

### Raspberry Pi 5 Benchmarks

| Cameras | Resolution | Capture FPS | UI FPS | CPU Usage | Memory |
| ------- | ---------- | ----------- | ------ | --------- | ------ |
| 1       | 640x480    | 20          | 15     | ~15%      | ~150MB |
| 2       | 640x480    | 20          | 15     | ~25%      | ~180MB |
| 3       | 640x480    | 20          | 15     | ~35%      | ~200MB |

---

## Troubleshooting common issues

### Cameras Not Detected

```bash
# Check if cameras are recognized
ls -l /dev/video*
v4l2-ctl --list-devices

# Verify user is in video group
groups $USER

# Test camera directly
ffplay /dev/video0
```

### GStreamer Issues

```bash
# Test GStreamer pipeline
gst-launch-1.0 v4l2src device=/dev/video0 ! jpegdec ! videoconvert ! autovideosink

# Disable GStreamer in config.ini
use_gstreamer = false
```

### Application Crashes / Unknown behavior, errors

```bash
# Check logs
cat logs/camera_dashboard.log | tail -50
journalctl -u camera-dashboard --no-pager | tail -50

# Run with debug output
DEBUG_PRINTS=true python3 main.py
```

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Main Application                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ CameraWidget│  │ CameraWidget│  │ CameraWidget│     │
│  │   + Worker  │  │   + Worker  │  │   + Worker  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐     │
│  │CaptureWorker│  │CaptureWorker│  │CaptureWorker│     │
│  │  (QThread)  │  │  (QThread)  │  │  (QThread)  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│  ┌──────▼──────────────────────────────────▼──────┐    │
│  │           GStreamer / V4L2 Backend             │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Threading Model

- **Main Thread**: Qt event loop, UI rendering
- **Capture Threads**: One QThread per camera for frame capture
- **Timer Callbacks**: Performance monitoring, health logging, device rescanning

### Data Flow

1. `CaptureWorker` grabs frames via GStreamer or V4L2
2. Frames emitted to main thread via Qt signals
3. UI renders at fixed interval (15 FPS) using latest frame
4. Performance monitor adjusts FPS based on system load

---

## File Structure

```
camera_dashboard/
├── main.py                   # Application entry point
├── core/                     # Core functionality
│   ├── __init__.py           # Exports: config, camera, performance
│   ├── config.py             # Configuration loading, logging setup, constants
│   ├── camera.py             # CaptureWorker thread, camera discovery
│   └── performance.py        # CPU load/temp monitoring, stress detection
├── ui/                       # User interface
│   ├── __init__.py           # Exports: CameraWidget, get_smart_grid
│   ├── widgets.py            # CameraWidget, FullscreenOverlay
│   └── layout.py             # Grid layout helpers
├── utils/                    # Utilities
│   ├── __init__.py           # Exports: system helpers
│   └── helpers.py            # Process management, systemd integration
├── config.ini                # Configuration file
├── install.sh                # Automated installer
├── camera-dashboard.service  # Systemd service (auto-generated)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE.MIT               # MIT License
├── logs/                     # Log files (created at runtime)
│   └── camera_dashboard.log
└── .venv/                    # Python virtual environment
```

### Module Overview

| Module | Description |
| ------ | ----------- |
| `core.config` | Configuration loading from INI, environment variables, logging setup |
| `core.camera` | `CaptureWorker` QThread for video capture, camera discovery functions |
| `core.performance` | CPU load and temperature monitoring, stress detection |
| `ui.widgets` | `CameraWidget` for camera tiles, `FullscreenOverlay` for fullscreen view |
| `ui.layout` | Grid layout calculation based on camera count |
| `utils.helpers` | System utilities, process management, systemd notifications |

---

## License

This project is licensed under the MIT License - see [LICENSE.MIT](LICENSE.MIT) for details.

---
