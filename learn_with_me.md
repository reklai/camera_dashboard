# Camera Grid App — Architecture and Deep Concepts

Concepts I needed to learn for this project to be possible

---

## 1) Full Program Flow Diagram

```text
┌──────────────────────────────────────────────────────────┐
│                          main()                          │
│ - build QApplication + MainWindow                        │
│ - discover cameras                                       │
│ - build grid layout                                      │
│ - start dynamic FPS monitor                              │
└───────────────────────────────┬──────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────┐
│                 find_working_cameras()                   │
│ - scan /dev/video*                                       │
│ - test in parallel                                       │
│ - kill holders if blocked                                │
└───────────────────────────────┬──────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────┐
│               CameraWidget (per camera)                  │
│ - UI tile + label                                        │
│ - input handling (tap/hold)                              │
│ - fullscreen overlay                                     │
│ - swap logic                                             │
│ - UI render timer                                        │
└───────────────────────────────┬──────────────────────────┘
                                │ creates
                                ▼
┌──────────────────────────────────────────────────────────┐
│              CaptureWorker (QThread)                     │
│ - open camera                                            │
│ - grab/retrieve frames                                   │
│ - emit frame_ready                                       │
│ - reconnect on failure                                   │
└───────────────────────────────┬──────────────────────────┘
                                │ emits (signal)
                                ▼
┌──────────────────────────────────────────────────────────┐
│         CameraWidget.on_frame(frame)                     │
│ - store _latest_frame                                    │
└───────────────────────────────┬──────────────────────────┘
                                │ timer tick (UI FPS)
                                ▼
┌──────────────────────────────────────────────────────────┐
│         CameraWidget._render_latest_frame()              │
│ - convert frame -> QImage -> QPixmap                     │
│ - display on grid or fullscreen overlay                  │
└───────────────────────────────┬──────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────┐
│                FullscreenOverlay (opt)                   │
│ - show single camera                                     │
│ - click to exit                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 2) Signals vs Threads Diagram (Visual)

```text
THREAD MODEL (Qt)

UI THREAD (Main Thread)                        WORKER THREAD (CaptureWorker)
──────────────────────────────────             ───────────���─────────────────
QApplication event loop                         CaptureWorker.run()
CameraWidget                                   Open camera
QTimers (render + fps log)                      grab() + retrieve()
on_frame() slot                                frame_ready.emit(frame)

            ┌──────────────────────────────────────────────────────────────┐
            │             Qt Signal/Slot Boundary (Thread-Safe)            │
            └──────────────────────────────────────────────────────────────┘

frame_ready (signal)  ─────────────────────────────────────────────►  on_frame (slot)
                                                                  (executes in UI thread)

IMPORTANT:
- The worker never touches the UI.
- The UI never blocks on camera reads.
- Signals transfer ownership safely across threads.
```

---

## 3) Deep Explanation of Core Concepts

### A) Threads (CaptureWorker)
Camera I/O is slow and blocking, so each camera runs inside a QThread (`CaptureWorker`).  
This prevents the UI from freezing while frames are being captured.

Key design:
- Worker thread only captures.
- UI thread only renders.

---

### B) Signals and Slots (Thread-Safe Bridge)
When the worker emits:

```python
self.frame_ready.emit(frame)
```

Qt delivers that signal to the UI thread:

```python
self.worker.frame_ready.connect(self.on_frame)
```

That means `on_frame()` always runs in the UI thread even though the frame is produced in another thread. This is the correct and safe way to move data across threads in Qt.

---

### C) Frame Flow Pipeline
Actual data flow:

1. Worker grabs and retrieves frame from OpenCV  
2. Worker emits `frame_ready(frame)`  
3. UI slot `on_frame()` stores it as `_latest_frame`  
4. UI timer renders the most recent frame into the QLabel  

This avoids rendering too often and keeps UI responsive.

---

### D) UI Rendering and QTimer
Instead of rendering on every camera frame, the UI uses a timer:

```python
self.render_timer.timeout.connect(self._render_latest_frame)
```

This keeps rendering stable at a chosen UI FPS (e.g., 15), even if the camera captures at 30+ FPS.

---

### E) Fullscreen Overlay
Fullscreen is handled by a separate fullscreen widget (`FullscreenOverlay`).  
The camera tile stays in the grid, but the fullscreen overlay shows the same frames.

This avoids complex reparenting and keeps rendering consistent.

---

### F) Swap Logic
Swapping works by:
- Tracking which tile is selected (long press)
- Reordering widgets inside the grid layout
- Keeping widget contents intact

Only their positions change, not the camera streams.

---

### G) Dynamic FPS Tuning
A system stress monitor reduces camera FPS when CPU load or temperature is high.  
When the system stabilizes, FPS is restored.  
This keeps the app stable on low-power devices.

---

## 4) Quick Summary

- Capture runs in worker threads.
- UI rendering runs only in the main thread.
- Signals move frames safely across threads.
- Render timer controls UI FPS separately from capture FPS.
- Fullscreen uses overlay, not reparenting.
- Swap logic reorders grid positions.
- Dynamic FPS protects system stability.

---

# I hate coding sometimes . . .

### OpenCV color order
OpenCV gives BGR, but Qt expects RGB. If you forget conversion, colors look wrong.

### QImage lifetime
If you build a QImage from a temporary NumPy buffer, the memory can be freed early. You must ensure the buffer stays alive or deep-copy.

### Thread shutdown
QThread must be stopped cleanly. If you quit the app while capture is running, you can get crashes or hanging threads.

### Camera re-open loops
Some cameras fail after disconnects. You need a retry delay or backoff, or you can hammer the device.

### Widget resizing
If you don’t keep aspect ratio when scaling, previews stretch. If you do keep it, you need to handle black bars.

### Fullscreen overlay focus
A fullscreen widget can steal focus and prevent clicks on the underlying grid until it’s closed.

### Swap state edge cases
If swap-select is active and the user clicks the same tile again, you must handle “cancel” properly.

### FPS timer drift
If your timer interval is too low, the UI can drift or lag under load. Use a stable interval and measure actual FPS.
