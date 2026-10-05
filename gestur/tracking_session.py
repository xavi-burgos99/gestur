"""Model/camera lifecycle off the render thread, with one latest request."""

import logging
import threading
import time
from copy import deepcopy

LOG = logging.getLogger(__name__)


def tracking_request(config, *, has_model=True, no_camera=False):
    """Global switches permit recognition; enabled controls determine demand."""
    if no_camera or not has_model:
        return None
    tracking = config["tracking"]
    inputs = {
        mapping["input"]
        for mapping in config["controls"]["mappings"]
        if mapping["enabled"]
    }
    pose_parts = tuple(
        part
        for part in ("head", "torso")
        if tracking["use_pose"] and any(name.startswith(f"{part}_") for name in inputs)
    )
    pose = bool(pose_parts)
    hands = tracking["use_hands"] and any("hand" in name for name in inputs)
    if not pose and not hands:
        return None
    return dict(
        use_pose=pose,
        use_hands=hands,
        pose_parts=pose_parts,
        smoothing_time_ms=tracking["smoothing_ms"],
        mirror=tracking["mirror"],
        camera_index=tracking["camera_index"],
        inference_fps=tracking["inference_fps"],
        hand_fps=tracking["hand_fps"],
        width=tracking["width"],
        height=tracking["height"],
    )


def default_factory(settings):
    # Importing the inference runtime can be expensive, too.
    from gestur.pose_detector import PoseHandTracker

    return PoseHandTracker(**settings)


class TrackingSession:
    def __init__(
        self, publish, factory=default_factory, retry_seconds=15, clock=time.monotonic
    ):
        self.publish = publish
        self.factory, self.retry_seconds, self.clock = factory, retry_seconds, clock
        self._condition = threading.Condition(threading.RLock())
        self._desired = None
        self._generation = 0
        self._active_attempt = None
        self._closed = False
        self._thread = None
        self._tracker = None
        self._state = "stopped"
        self._error = None
        self._last_metrics = {}

    def request(self, settings):
        with self._condition:
            if self._closed or settings == self._desired:
                return
            self._desired = deepcopy(settings)
            self._generation += 1
            self._active_attempt = None
            self._error = None
            self._state = "starting" if settings else "stopped"
            # Revoke old callbacks before returning to the render thread.
            self.publish({})
            if self._thread is None and settings is not None:
                self._thread = threading.Thread(
                    target=self._work, name="gestur-tracking-lifecycle", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            tracker = self._tracker
            state = {
                "state": self._state,
                "error": self._error,
                "requested": deepcopy(self._desired),
                "metrics": dict(self._last_metrics),
            }
        if tracker is not None:
            state["metrics"] = tracker.get_metrics()
        return state

    def _publish(self, generation, attempt, data):
        with self._condition:
            if (
                generation == self._generation
                and attempt is self._active_attempt
                and not self._closed
            ):
                self.publish(data)

    @staticmethod
    def _resources_active(tracker):
        return tracker is not None and (
            tracker.running
            or any(
                thread is not None and thread.is_alive()
                for thread in (
                    getattr(tracker, "thread", None),
                    getattr(tracker, "capture_thread", None),
                )
            )
        )

    def _stop(self, tracker):
        if tracker is None:
            return True
        with self._condition:
            self._active_attempt = None
        try:
            tracker.stop()
            # stop() has bounded joins. A stuck capture driver must never be
            # overlapped by a replacement camera, even after another request.
            with self._condition:
                while self._resources_active(tracker):
                    self._tracker = tracker
                    self._state = "stopping"
                    if self._closed:
                        return False
                    self._condition.wait(timeout=0.25)
            metrics = tracker.get_metrics()
            with self._condition:
                self._last_metrics = metrics
            return True
        except Exception:
            LOG.exception("No se pudo cerrar el seguimiento correctamente")
            # A failed close cannot certify that the camera was released.
            with self._condition:
                self._tracker = tracker
                self._state = "stopping"
                self._error = "No se pudo liberar la cámara; reinicia el visor."
                while not self._closed:
                    self._condition.wait()
            return False

    def _work(self):
        tracker = None
        generation = -1
        next_attempt = 0.0
        try:
            while True:
                with self._condition:
                    if self._closed:
                        break
                    desired = deepcopy(self._desired)
                    requested_generation = self._generation
                if generation != requested_generation:
                    if not self._stop(tracker):
                        break
                    tracker = None
                    with self._condition:
                        self._tracker = None
                        if requested_generation == self._generation:
                            self._state = "starting" if desired else "stopped"
                    generation = requested_generation
                    next_attempt = 0.0
                    # The requested settings may have changed during a slow
                    # close. Re-read them before opening any replacement.
                    continue
                if tracker is not None and (tracker.last_error or not tracker.running):
                    error = tracker.last_error or RuntimeError(
                        "El seguimiento se ha detenido."
                    )
                    with self._condition:
                        self._active_attempt = None
                        if generation == self._generation:
                            self._state, self._error = (
                                "retrying",
                                f"Seguimiento detenido: {error}",
                            )
                            self.publish({})
                    if not self._stop(tracker):
                        break
                    tracker = None
                    next_attempt = self.clock() + self.retry_seconds
                    with self._condition:
                        self._tracker = None
                        if generation == self._generation:
                            self._state = "retrying"
                    continue
                if (
                    desired is not None
                    and tracker is None
                    and self.clock() >= next_attempt
                ):
                    next_attempt = self.clock() + self.retry_seconds
                    try:
                        with self._condition:
                            if generation != self._generation or self._closed:
                                continue
                            attempt = object()
                            self._active_attempt = attempt
                        tracker = self.factory(desired)
                        tracker.subscribe(
                            lambda data, tag=generation, token=attempt: self._publish(
                                tag, token, data
                            )
                        )
                        tracker.run()
                        with self._condition:
                            if generation == self._generation and not self._closed:
                                self._tracker = tracker
                                self._state, self._error = "running", None
                    except Exception as error:
                        if not self._stop(tracker):
                            break
                        tracker = None
                        next_attempt = self.clock() + self.retry_seconds
                        with self._condition:
                            if generation == self._generation and not self._closed:
                                self._state = "retrying"
                                self._error = (
                                    f"No se pudo iniciar el seguimiento: {error}"
                                )
                                self.publish({})
                                LOG.error(self._error)
                with self._condition:
                    if not self._closed and generation == self._generation:
                        self._condition.wait(
                            timeout=0.25 if desired is not None else None
                        )
        finally:
            released = self._stop(tracker)
            with self._condition:
                self._tracker = None if released else tracker
                self._state = "stopped" if released else "stopping"

    def close(self):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._active_attempt = None
            self.publish({})
            self._condition.notify_all()
            worker = self._thread
        if worker:
            worker.join(timeout=6)
            if worker.is_alive():
                LOG.warning(
                    "La cámara todavía está cerrándose; la salida del proceso liberará los recursos."
                )
                return False
        with self._condition:
            return self._state == "stopped"
