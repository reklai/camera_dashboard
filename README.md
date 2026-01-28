# Camera Dashboard

A multi-camera system meant for cargo vechicle optimized for Raspberry Pi but compatible with any Linux system. Features real-time video streaming, dynamic performance tuning, hot-plugging support, and an intuitive touch-enabled interface.

![Camera Dashboard](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey.svg)

## Quick Start

```bash
# Clone and install
git clone <repository-url>
cd camera_dashboard
chmod +x install.sh
./install.sh

# Run the application
source .venv/bin/activate
python3 main.py
```

---

## Key Features

### **Multi-Camera Support**
- Automatically detects and displays up to 3 USB cameras simultaneously
- Smart grid layout adapts to available camera count
- Hot-plugging support - cameras can be connected/disconnected at runtime

### **Interactive Interface**
- **Touch/Mouse Controls**: Single click to fullscreen, long press to swap camera positions
- **Drag & Drop**: Reorganize camera layout with intuitive gestures
- **Responsive Design**: Scales beautifully across different screen sizes

### **Performance Optimized**
- **Dynamic FPS Adjustment**: Automatically reduces frame rate when CPU load or temperature is high
- **Threaded Architecture**: Separate capture threads ensure smooth UI performance
- **Resource Management**: Efficient memory usage with frame buffering

### **System Integration**
- **Zero-Configuration**: Works out of the box with standard USB cameras
- **Cross-Platform**: Optimized for Raspberry Pi but runs on any Linux system
- **Robust Error Handling**: Automatic camera reconnection with exponential backoff

---

## System Requirements

### Supported Platforms
- **Raspberry Pi 4/5** (64-bit recommended)
- **Linux Operating Systems** (Ubuntu, Debian, etc.)

### Hardware
- USB webcams or IP cameras compatible with V4L2
- Minimum 2GB RAM (4GB+ recommended for multiple cameras)

### Software
- Python 3.8+
- PyQt6
- OpenCV
- pyudev (for device detection)

---

## Step-By-Step Installation

### Option 1: Automated Installation (Recommended)

```bash
# Download and run the installer
chmod +x install.sh
./install.sh
```

### Option 2: Manual Installation

<details>
<summary>Click to expand manual steps</summary>

#### 1. Update System
```bash
sudo apt update
sudo apt upgrade -y
```

#### 2. Install System Dependencies
```bash
sudo apt install -y \
  python3 python3-pip python3-venv \
  libgl1 libegl1 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
  libqt6gui6 libqt6widgets6
```

#### 3. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 4. Install Python Packages
```bash
pip install --upgrade pip
pip install PyQt6 opencv-python pyudev
```

#### 5. Set Camera Permissions
```bash
sudo usermod -aG video $USER
# Log out and back in for changes to take effect
```

</details>

---

## Usage Design

### Basic Controls
| Action | Result |
|--------|--------|
| **Short Click** | Toggle fullscreen view |
| **Long Press (400ms+)** | Enter swap mode (yellow border) |
| **Second Click** | Swap camera positions |
| **Ctrl+Q** | Exit application |
| **Ctrl+C** | Exit from terminal |

### Camera Status
- **Green Border**: Camera connected and streaming
- **Yellow Border**: Camera selected for swapping
- **"Disconnected"**: No camera detected in slot

### Settings Tile
The top-left tile provides:
- Application restart functionality
- System information display
- Quick access to main menu

---

## Architecture Overview

### Core Components


### Threading Model

- **Main Thread**: UI rendering, event handling
- **Capture Threads**: One per camera, handles video capture
- **Timer Threads**: Performance monitoring and device rescanning

### Data Flow

1. **Capture**: `CaptureWorker` grabs frames from OpenCV
2. **Signal**: Frame data emitted via Qt signals
3. **Display**: UI thread renders frames at controlled rate
4. **Adapt**: Performance monitoring adjusts capture FPS dynamically

---

## Configuration

### Environment Variables
```bash
# Enable debug logging
export DEBUG_PRINTS=true

# Disable dynamic FPS adjustment
export DYNAMIC_FPS_ENABLED=false
```

### Performance Tuning
Key constants in `main.py`:

```python
# Dynamic FPS thresholds
CPU_LOAD_THRESHOLD = 0.85      # 85% CPU load
CPU_TEMP_THRESHOLD_C = 70.0     # 70°C temperature
MIN_DYNAMIC_FPS = 5             # Minimum FPS

# Rescan intervals
RESCAN_INTERVAL_MS = 5000       # Camera hot-plug detection
FAILED_CAMERA_COOLDOWN_SEC = 30 # Retry delay for failed cameras
```

---

## Troubleshooting

### Common Issues

<details>
<summary>Cameras not detected</summary>

**Symptoms**: Shows "Disconnected" for all slots
**Solutions**:
1. Check camera connections: `ls -l /dev/video*`
2. Verify user permissions: `groups $USER` (should include "video")
3. Test with cheese/vlc to confirm cameras work
4. Restart application after connecting cameras

</details>

<details>
<summary>Application crashes on startup</summary>

**Symptoms**: PyQt6 related errors
**Solutions**:
1. Install system packages: `sudo apt install python3-pyqt6`
2. Check virtual environment: `source .venv/bin/activate`
3. Update graphics drivers on Raspberry Pi

</details>

<details>
<summary>Poor performance or lag</summary>

**Symptoms**: Choppy video, high CPU usage
**Solutions**:
1. Reduce camera resolution in code
2. Enable dynamic FPS adjustment (default)
3. Use higher quality USB cameras
4. Close other applications

</details>

### Debug Mode
Enable detailed logging:

```python
# In main.py, line 40
DEBUG_PRINTS = True
```

### Log Analysis
```bash
# Monitor performance in real-time
python3 main.py 2>&1 | grep FPS
```

---

## 🔒 Security Considerations

### Camera Permissions
- Application runs as normal user (not root)
- Camera access controlled via Linux group permissions
- No network access required for local USB cameras

### Data Privacy
- All video processing happens locally
- No data transmitted to external services
- Frame buffers limited to prevent memory accumulation

---

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd camera_dashboard

# Set up development environment
python3 -m venv dev_venv
source dev_venv/bin/activate
pip install PyQt6 opencv-python pyudev

# Run in development mode
python3 main.py
```

### Code Style
- Follow PEP 8 Python conventions
- Use type hints where appropriate
- Add logging for new features
- Document complex algorithms

### Testing
```bash
# Test camera detection
python3 -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"

# Test PyQt6 installation
python3 -c "from PyQt6 import QtWidgets; print('PyQt6 OK')"
```

---

## 📊 Performance Metrics

### Raspberry Pi 5 Benchmarks
| Cameras | Resolution | FPS | CPU Usage | Memory |
|---------|------------|-----|-----------|---------|
| 1 | 640x480 | 30 | 15% | 120MB |
| 2 | 640x480 | 20 | 35% | 200MB |
| 3 | 640x480 | 15 | 50% | 280MB |

### Dynamic FPS Behavior
- **Normal Operation**: Target FPS maintained
- **High CPU Load**: FPS gradually reduced to minimum
- **Temperature >70°C**: Immediate FPS reduction
- **Stable System**: FPS gradually restored

---

## Technical Documentation

For deep technical insights, see [project.md](project.md) which covers:
- Detailed threading architecture
- Frame processing pipeline
- Signal/slot communication patterns
- Performance optimization techniques

---

## Acknowledgments

- **OpenCV** team for excellent computer vision library
- **PyQt6** for robust cross-platform GUI framework
- **Raspberry Pi Foundation** for making computing accessible

---
