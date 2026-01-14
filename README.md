Camera Grid Viewer

Fullscreen multi-camera display with click-to-fullscreen and hold-to-swap.
Quick Start

bash
# 1. Install
chmod +x install.sh
./install.sh

# 2. Run
./main.py

Controls

    Click any camera = toggle fullscreen

    Hold 400ms = yellow border (swap mode)

    Click other camera = swap positions

    Click yellow camera = clear swap mode

    Ctrl+Q = quit

## Install (manual)

```
pip install PyQt6 opencv-python qdarkstyle imutils cv2-enumerate-cameras
chmod +x main.py
```

Linux only: sudo usermod -a -G video $USER (then logout/login)
Files

### Project Structure: 

├── camera_grid.py     # Main app
├── install.sh         # Setup script
├── requirements.txt   # pip install -r
└── README.md          # This file

##### Terminal prints debug activities:

-   Found 2 cameras: [0, 1]
-   Press cam0_140123456
-   ENTER swap cam0_140123456  
-   SWAP cam0_140123456 ↔ cam1_140123789

###### Require Packages Installments:

-   **PyQt6**
-   **opencv-python**
-   **qdarkstyle**
-   **imutils**
-   **cv2-enumerate-cameras**

Tested

    Arch linux via AMD CPU and Nvidia GPU

    Raspberry Pi 4 and 5 via Intel5 CPU

