# Codec Thread Translateion
import threading, queue, cv2
from queue import Empty

class ThreadedCapture:
    def __init__(self, src=0, backend=cv2.CAP_V4L2, width=1920, height=1080):
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=128)  # Smaller = less lag
        self.stopped = False
        
    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
    
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            try:
                if ret: 
                    self.q.put_nowait(frame)  # Non-blocking put
            except queue.Full:
                self.q.get_nowait()  # Drop oldest frame
                self.q.put_nowait(frame)
    
    def read(self):
        try:
            return True, self.q.get_nowait()
        except Empty:
            return False, None
    
    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)  # Wait for clean shutdown
        self.cap.release()

# CORRECT USAGE
cap = ThreadedCapture(4, cv2.CAP_V4L2)  # Use V4L2 on Linux
cap.start()
fps = cap.cap.get(cv2.CAP_PROP_FPS)
codec = cap.cap.get(cv2.CAP_PROP_FOURCC)
print(fps)
print(codec)
try:
    while True:
        ret, frame = cap.read()
        if not ret: continue
        cv2.imshow('Feed', frame)
        if cv2.waitKey(1) == ord('q'): break
finally:
    cap.stop()
    cv2.destroyAllWindows()