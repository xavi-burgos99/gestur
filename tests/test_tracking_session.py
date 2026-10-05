"""Camera lifecycle must never stall drawing or revive obsolete callbacks."""

import threading
import time

import pytest

from runtime_config import default_config
from tracking_session import TrackingSession, tracking_request


def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "lifecycle transition timed out"
        time.sleep(0.005)


class Tracker:
    def __init__(self):
        self.running = False
        self.last_error = None
        self.stopped = threading.Event()

    def subscribe(self, callback):
        self.callback = callback

    def run(self):
        self.running = True

    def stop(self):
        self.running = False
        self.stopped.set()

    def get_metrics(self):
        return {"frames": 17}


def test_demand_loads_only_enabled_detectors_actually_used_by_controls():
    config = default_config()
    config["tracking"]["use_hands"] = True
    assert tracking_request(config)["use_pose"] is True
    assert tracking_request(config)["use_hands"] is False
    assert tracking_request(config)["pose_parts"] == ("head",)
    mapping = config["controls"]["mappings"][0]
    config["controls"]["mappings"] = [
        {**mapping, "input": "right_hand_roll", "enabled": True}
    ]
    assert tracking_request(config)["use_pose"] is False
    assert tracking_request(config)["use_hands"] is True
    assert tracking_request(config, has_model=False) is None
    assert tracking_request(config, no_camera=True) is None
    config["tracking"]["use_hands"] = False
    assert tracking_request(config) is None
    config["tracking"]["use_hands"] = True
    config["controls"]["mappings"][0]["enabled"] = False
    assert tracking_request(config) is None


@pytest.mark.parametrize(
    "input_name",
    [f"torso_{field}" for field in ("x", "y", "scale", "pitch", "yaw", "roll")],
)
def test_body_only_controls_request_pose_and_torso_presence(input_name):
    config = default_config()
    mapping = config["controls"]["mappings"][0]
    config["controls"]["mappings"] = [{**mapping, "input": input_name, "enabled": True}]
    request = tracking_request(config)
    assert request["use_pose"] is True and request["use_hands"] is False
    assert request["pose_parts"] == ("torso",)
    config["tracking"]["use_pose"] = False
    assert tracking_request(config) is None
    config["tracking"]["use_pose"] = True
    config["controls"]["mappings"][0]["enabled"] = False
    assert tracking_request(config) is None


def test_mixed_body_head_and_hand_proximity_demand_remains_independent():
    config = default_config()
    config["tracking"]["use_hands"] = True
    mapping = config["controls"]["mappings"][0]
    config["controls"]["mappings"] = [
        {**mapping, "input": name, "enabled": True}
        for name in ("torso_scale", "head_x", "left_hand_scale", "right_hand_scale")
    ]
    request = tracking_request(config)
    assert request["pose_parts"] == ("head", "torso") and request["use_hands"]
    config["tracking"]["use_pose"] = False
    request = tracking_request(config)
    assert (
        not request["use_pose"] and request["pose_parts"] == () and request["use_hands"]
    )
    config["tracking"]["use_hands"] = False
    assert tracking_request(config) is None


def test_slow_model_load_does_not_block_request_and_stale_start_is_discarded():
    entered, release = threading.Event(), threading.Event()
    received, trackers = [], []
    owner_thread = threading.get_ident()

    def factory(settings):
        assert threading.get_ident() != owner_thread
        tracker = Tracker()
        trackers.append(tracker)
        entered.set()
        assert release.wait(3)
        return tracker

    session = TrackingSession(received.append, factory=factory)
    try:
        session.request({"camera": 0})
        assert entered.wait(3)
        session.request(None)
        assert session.snapshot()["state"] == "stopped"
        release.set()
        until(lambda: trackers[0].stopped.is_set())
        trackers[0].callback({"obsolete": True})
        assert all(not data for data in received)
    finally:
        release.set()
        session.close()


def test_live_change_replaces_tracker_and_revokes_previous_callbacks():
    trackers, received = [], []

    def factory(settings):
        tracker = Tracker()
        trackers.append(tracker)
        return tracker

    session = TrackingSession(received.append, factory=factory)
    try:
        settings = {"camera": 0}
        session.request(settings)
        until(lambda: session.snapshot()["state"] == "running")
        settings["camera"] = 99
        assert session.snapshot()["requested"] == {"camera": 0}
        session.request({"camera": 0})
        assert len(trackers) == 1
        trackers[0].callback({"head": 1})
        assert received[-1] == {"head": 1}
        session.request({"camera": 1})
        assert received[-1] == {}
        until(lambda: len(trackers) == 2 and session.snapshot()["state"] == "running")
        assert trackers[0].stopped.is_set()
        trackers[0].callback({"head": "stale"})
        assert received[-1] == {}
        trackers[1].callback({"head": 2})
        assert received[-1] == {"head": 2}
    finally:
        session.close()
    assert trackers[-1].stopped.is_set()
    trackers[-1].callback({"head": "closed"})
    assert received[-1] == {}
    assert session.snapshot()["metrics"] == {"frames": 17}


def test_failed_start_waits_full_backoff_after_failure_then_recovers():
    clock, attempts, received = [100.0], [], []

    def factory(settings):
        attempts.append(True)
        if len(attempts) == 1:
            clock[0] += 20  # Loading itself took longer than the retry delay.
            raise RuntimeError("camera disconnected")
        return Tracker()

    session = TrackingSession(received.append, factory=factory, clock=lambda: clock[0])
    try:
        session.request({"camera": 0})
        until(lambda: session.snapshot()["state"] == "retrying")
        assert "camera disconnected" in session.snapshot()["error"]
        clock[0] = 134.0
        time.sleep(0.3)
        assert len(attempts) == 1
        clock[0] = 135.0
        until(lambda: session.snapshot()["state"] == "running")
        assert len(attempts) == 2
        assert session.snapshot()["error"] is None
    finally:
        session.close()


def test_running_camera_failure_clears_input_then_empty_scene_cancels_retry():
    tracker, received, attempts = Tracker(), [], []

    def factory(settings):
        attempts.append(True)
        return tracker

    session = TrackingSession(received.append, factory=factory, retry_seconds=15)
    try:
        session.request({"camera": 0})
        until(lambda: session.snapshot()["state"] == "running")
        tracker.callback({"head": 1})
        tracker.last_error = RuntimeError("capture failed")
        until(lambda: session.snapshot()["state"] == "retrying")
        assert received[-1] == {}
        assert tracker.stopped.is_set()
        session.request(None)
        assert session.snapshot()["state"] == "stopped"
        assert session.snapshot()["error"] is None
        time.sleep(0.3)
        assert len(attempts) == 1
    finally:
        session.close()


def test_retry_of_same_configuration_revokes_callbacks_from_previous_attempt():
    trackers, received = [], []

    def factory(settings):
        trackers.append(Tracker())
        return trackers[-1]

    session = TrackingSession(received.append, factory=factory, retry_seconds=0)
    try:
        session.request({"camera": 0})
        until(lambda: session.snapshot()["state"] == "running")
        trackers[0].last_error = RuntimeError("disconnected")
        until(lambda: len(trackers) == 2 and session.snapshot()["state"] == "running")
        trackers[1].callback({"head": 0.8})
        trackers[0].callback({"head": 0.1})
        assert received[-1] == {"head": 0.8}
    finally:
        session.close()


def test_replacement_waits_for_old_camera_thread_even_when_stop_returns():
    entered, release = threading.Event(), threading.Event()
    trackers, received = [], []

    def factory(settings):
        tracker = Tracker()
        if not trackers:
            tracker.capture_thread = threading.Thread(
                target=lambda: (entered.set(), release.wait(3))
            )
            tracker.capture_thread.start()
        trackers.append(tracker)
        return tracker

    session = TrackingSession(received.append, factory=factory)
    try:
        session.request({"camera": 0})
        until(lambda: session.snapshot()["state"] == "running")
        session.request({"camera": 1})
        assert trackers[0].stopped.wait(3)
        time.sleep(0.3)
        assert len(trackers) == 1
        session.request({"camera": 2})  # Only the latest request should start.
        trackers[0].callback({"head": "stale"})
        assert received[-1] == {}
        release.set()
        until(lambda: session.snapshot()["state"] == "running")
        assert len(trackers) == 2
        assert session.snapshot()["requested"] == {"camera": 2}
    finally:
        release.set()
        session.close()


def test_shutdown_reports_unreleased_camera_so_controller_can_exit_process():
    release = threading.Event()
    tracker = Tracker()
    tracker.capture_thread = threading.Thread(target=lambda: release.wait(3))
    tracker.capture_thread.start()
    session = TrackingSession(lambda data: None, factory=lambda settings: tracker)
    try:
        session.request({"camera": 0})
        until(lambda: session.snapshot()["state"] == "running")
        assert session.close() is False
        assert session.snapshot()["state"] == "stopping"
    finally:
        release.set()
        tracker.capture_thread.join(3)
