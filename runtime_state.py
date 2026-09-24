"""Small hardware-independent bridges for the camera and render loops."""
from collections import deque
from copy import deepcopy
import threading
import time


class LatestPose:
    """A one-slot mailbox: callbacks never access Panda3D or accumulate work."""
    def __init__(self, max_age=0.35, clock=time.monotonic):
        self.max_age = max_age
        self.clock = clock
        self._lock = threading.Lock()
        self._data = {}
        self._received = None

    def publish(self, data):
        snapshot = deepcopy(data)
        with self._lock:
            self._data = snapshot
            self._received = self.clock()

    def read(self):
        with self._lock:
            if self._received is None or self.clock() - self._received > self.max_age:
                return {}
            return deepcopy(self._data)


class FrameMetrics:
    def __init__(self, max_samples=18000):
        self.intervals = deque(maxlen=max_samples)
        self.frames = 0
        self.started = None
        self.previous = None

    def tick(self, now):
        if self.started is None:
            self.started = now
        if self.previous is not None:
            self.intervals.append((now - self.previous) * 1000)
        self.previous = now
        self.frames += 1

    def summary(self):
        samples = sorted(self.intervals)
        duration = (self.previous - self.started) if self.frames > 1 else 0
        def percentile(fraction):
            return round(samples[min(len(samples) - 1, int((len(samples) - 1) * fraction))], 3) if samples else None
        return {
            "frames": self.frames,
            "seconds": round(duration, 3),
            "render_fps": round((self.frames - 1) / duration, 2) if duration > 0 else 0,
            "frame_ms_p50": percentile(.5),
            "frame_ms_p95": percentile(.95),
            "frame_ms_p99": percentile(.99),
            "samples": len(samples),
        }
