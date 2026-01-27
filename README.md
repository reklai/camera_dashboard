# Camera Dashboard

## Supported OS
- Raspberry Pi 4/5 (64‑bit recommended)
- Linux Operating Systems

---

## 1) Update the system

```bash
sudo apt update
sudo apt upgrade -y
```

---

## 2) Install system dependencies (required for PyQt6)

```bash
sudo apt install -y \
  python3 python3-pip python3-venv \
  libgl1 libegl1 libxkbcommon0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
  libqt6gui6 libqt6widgets6
```

> These libraries prevent PyQt6 from crashing at startup on Raspberry Pi.

---

## 3) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4) Install Python packages via pip

```bash
pip install --upgrade pip
pip install PyQt6 opencv-python
pip install pyudev
```

## If `pyqt6 / opencv-python / pyudev` fails to install on your Pi, use the system package instead:
>
> ```bash
> sudo apt install -y python3-opencv
> sudo apt install python3-pyqt6
> sudo apt-get install -y python3-pyudev
> sudo apt install python3-pyqt6 python3-opencv python3-opengl
> sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good
> sudo apt install python3-picamera2
> ```

---

## 5) Fix camera permissions (recommended)

You should not run the entire app as `sudo`.  
Instead, grant camera access to your user:

```bash
sudo usermod -aG video $USER
```

Then log out and back in (or reboot).

Check permissions:

```bash
ls -l /dev/video*
```

If the group is `video`, you can run the app normally.

---

## 6) Run the app

```bash
python3 main.py
```

---

## If you must run as sudo (not recommended)

This can break GUI permissions and your venv, but if you must:

```bash
sudo -E python3 main.py
```

---

## About using `sudo`

Only use `sudo` for apt install.  
Do not run `python3 -m venv` or `pip install` with `sudo`.

Using `sudo` for those steps:
- Installs packages system‑wide (not inside the venv)
- Can make the venv owned by root
- Can break GUI permissions

---

## Notes

- USB cameras should be visible as `/dev/video*`
- If no cameras are found, the app will show "Disconnected"
- Exit with Ctrl+Q or Ctrl+C in terminal or Q inside application window

---
