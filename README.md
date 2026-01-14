Camera Grid Viewer

Fullscreen multi n-camera setup display with click-to-fullscreen and hold-to-swap.

# Quick Start

# 1. Install
Ubuntu/Linux - Recommended (isolated environment)

## Create isolated Python environment -> Activate it -> Install package safely isolated from systems
```
python3 -m venv camera_env 
source camera_env/bin/activate
pip install --upgrade pip
pip install PyQt6 opencv-python qdarkstyle imutils cv2-enumerate-cameras
```

## Run (source if you havent before running)
source camera_env/bin/activate
python main.py

## To deactivate env later:
deactivate

## Application Features

    Click any camera = toggle fullscreen

    Hold 400ms = yellow border (swap mode)

    Click other camera = swap positions

    Click yellow camera = clear swap mode

    Ctrl+Q = quit

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

