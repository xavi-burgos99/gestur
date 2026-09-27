"""Decide which control ticks need a new image without slowing input polling."""


class RenderCadence:
    """Changed scenes draw immediately; unchanged scenes retain their last image.

    The occasional idle redraw covers exposure events from window systems that
    do not emit a property change.  It does not change framebuffer quality.
    """

    def __init__(self, target_fps=60, idle_fps=10, welcome_fps=30):
        self.target_fps = float(target_fps)
        self.idle_fps = float(idle_fps)
        self.welcome_fps = float(welcome_fps)
        self.last_draw = None
        self.next_refresh = None
        self.pending_frames = 2
        self.ticks = 0
        self.skipped = 0
        self.mode = "refresh"

    def invalidate(self, frames=1):
        self.pending_frames = max(self.pending_frames, frames)

    def due(self, now, *, welcome=False):
        self.ticks += 1
        fps = min(self.target_fps, self.welcome_fps if welcome else self.idle_fps)
        interval = 1.0 / fps
        if self.pending_frames:
            self.pending_frames -= 1
            self.mode = "refresh"
            self.last_draw = now
            self.next_refresh = now + interval
            return True
        self.mode = "welcome" if welcome else "idle"
        # Keep the refresh deadline anchored. Measuring each deadline from the
        # last draw accumulates clock jitter and drops a nominal 30 Hz scene to
        # 20-25 Hz under a 60 Hz task loop. A 1 ms tolerance avoids that extra
        # control tick without materially changing the refresh budget.
        tolerance = min(0.001, 0.1 / self.target_fps)
        if self.next_refresh is None or now + tolerance >= self.next_refresh:
            self.last_draw = now
            if self.next_refresh is None:
                self.next_refresh = now + interval
            else:
                periods = max(1, int((now - self.next_refresh) // interval) + 1)
                self.next_refresh += periods * interval
            return True
        self.skipped += 1
        return False
