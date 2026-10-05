import sys
import threading
import time
from types import SimpleNamespace

import pytest

from gestur.pose_detector import PoseHandTracker
from gestur.tracking_geometry import empty_part


class Camera:
    def __init__(self, fail=False):
        self.released = False
        self.count = 0
        self.fail = fail

    def read(self):
        time.sleep(0.002)
        self.count += 1
        if self.fail:
            return False, None
        return True, SimpleNamespace(shape=(480, 640, 3), number=self.count)

    def release(self):
        self.released = True


class Model:
    def __init__(self, fail=False, delay=0):
        self.timestamps = []
        self.frames = []
        self.fail = fail
        self.delay = delay

    def detect_for_video(self, frame, timestamp):
        if self.fail:
            raise RuntimeError("inference failure")
        time.sleep(self.delay)
        self.timestamps.append(timestamp)
        self.frames.append(frame.number)
        return SimpleNamespace(
            pose_landmarks=[],
            pose_world_landmarks=[],
            hand_landmarks=[],
            hand_world_landmarks=[],
            handedness=[],
        )


class Backend:
    def __init__(self, fail=False, delay=0):
        self.pose = Model(fail=fail, delay=delay)
        self.hands = Model()
        self.closed = False

    def image(self, frame, mirror):
        return frame

    def close(self):
        self.closed = True


def wait_until(condition, timeout=2):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert condition()


def setup(monkeypatch, **kwargs):
    tracker = PoseHandTracker(inference_fps=30, hand_fps=10, **kwargs)
    backends, cameras = [], []

    def backend():
        backends.append(Backend())
        return backends[-1]

    def camera():
        cameras.append(Camera())
        return cameras[-1]

    monkeypatch.setattr(tracker, "_build_backend", backend)
    monkeypatch.setattr(tracker, "_open_camera", camera)
    return tracker, backends, cameras


def test_restart_creates_fresh_models_and_closes_all_resources(monkeypatch):
    tracker, backends, cameras = setup(monkeypatch)
    tracker.stop()  # stop before run is harmless
    for iteration in range(2):
        tracker.run()
        wait_until(lambda: tracker.get_metrics()["pose_frames"] >= 3)
        tracker.stop()
        assert not tracker.running
        assert backends[iteration].closed and cameras[iteration].released
        assert not tracker.thread.is_alive() and not tracker.capture_thread.is_alive()
        timestamps = backends[iteration].pose.timestamps
        assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    tracker.stop()


def test_inference_uses_latest_frame_and_hands_have_independent_cadence(monkeypatch):
    tracker, backends, cameras = setup(monkeypatch)
    tracker.run()
    wait_until(lambda: tracker.get_metrics()["pose_frames"] >= 8)
    tracker.stop()
    pose = backends[0].pose
    assert len(pose.frames) >= 2 * len(backends[0].hands.frames)
    assert any(b - a > 2 for a, b in zip(pose.frames, pose.frames[1:]))
    assert tracker._latest_frame[0] == cameras[0].count
    assert (
        tracker.get_metrics()["captured_frames"] > tracker.get_metrics()["pose_frames"]
    )


def test_camera_start_failure_never_allocates_models(monkeypatch):
    tracker, backends, _ = setup(monkeypatch)

    def fail():
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(tracker, "_open_camera", fail)
    with pytest.raises(RuntimeError, match="camera unavailable"):
        tracker.run()
    assert backends == []
    assert not tracker.running
    tracker.stop()


def test_backend_start_failure_releases_open_camera(monkeypatch):
    tracker, _, cameras = setup(monkeypatch)

    def fail():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(tracker, "_build_backend", fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        tracker.run()
    assert len(cameras) == 1 and cameras[0].released
    assert not tracker.running
    tracker.stop()


def test_capture_failure_is_bounded_and_releases_resources(monkeypatch):
    tracker, backends, _ = setup(monkeypatch)
    camera = Camera(fail=True)
    monkeypatch.setattr(tracker, "_open_camera", lambda: camera)
    tracker.run()
    wait_until(lambda: not tracker.running)
    assert tracker.get_metrics()["capture_failures"] == 10
    assert tracker.last_error is not None
    assert backends[0].closed and camera.released


def test_model_failure_closes_capture_and_clears_detection(monkeypatch):
    tracker, _, cameras = setup(monkeypatch)
    backend = Backend(fail=True)
    monkeypatch.setattr(tracker, "_build_backend", lambda: backend)
    tracker.run()
    wait_until(lambda: not tracker.running)
    assert str(tracker.last_error) == "inference failure"
    assert backend.closed and cameras[0].released
    assert all(not part["detected"] for part in tracker.get_current_data().values())


def test_listener_mutation_is_isolated_and_stop_inside_callback_is_safe(monkeypatch):
    tracker, backends, cameras = setup(monkeypatch)
    observed = []

    def mutate(snapshot):
        snapshot["head"]["x"] = "corrupt"

    def observe(snapshot):
        observed.append(snapshot["head"]["x"])
        tracker.stop()

    tracker.subscribe(mutate)
    tracker.subscribe(observe)
    tracker.run()
    wait_until(lambda: bool(observed) and not tracker.running)
    assert all(value is None for value in observed)
    assert tracker.get_current_data()["head"]["x"] is None
    assert backends[0].closed and cameras[0].released


def test_disabled_tracking_never_opens_hardware(monkeypatch):
    tracker, backends, cameras = setup(monkeypatch, use_pose=False, use_hands=False)
    tracker.run()
    assert not tracker.running
    assert backends == cameras == []


def test_camera_rate_and_expiry_ignore_disabled_models(monkeypatch):
    settings = {}
    cap = SimpleNamespace(
        isOpened=lambda: True,
        release=lambda: None,
        set=lambda key, value: settings.update({key: value}),
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda index: cap,
            CAP_PROP_BUFFERSIZE="buffers",
            CAP_PROP_FRAME_WIDTH="width",
            CAP_PROP_FRAME_HEIGHT="height",
            CAP_PROP_FPS="fps",
        ),
    )
    hands = PoseHandTracker(
        use_pose=False, use_hands=True, inference_fps=60, hand_fps=5
    )
    assert hands._open_camera() is cap
    assert settings["fps"] == 5
    assert hands.detection_timeout == 0.4
    pose = PoseHandTracker(use_pose=True, use_hands=False, inference_fps=30, hand_fps=1)
    pose._open_camera()
    assert settings["fps"] == 30
    assert pose.detection_timeout == 0.25


def test_waiting_for_cadence_does_not_publish_duplicate_frames(monkeypatch):
    tracker, _, _ = setup(monkeypatch, use_hands=False)
    published = []
    tracker.subscribe(published.append)
    tracker.run()
    wait_until(lambda: tracker.get_metrics()["pose_frames"] >= 8)
    tracker.stop()
    metrics = tracker.get_metrics()
    assert metrics["captured_frames"] > 5 * metrics["pose_frames"]
    # One fresh publication per inference, plus the explicit cleared shutdown.
    assert len(published) == metrics["pose_frames"] + 1


def test_stalled_camera_publishes_expiry_once_without_refreshing_stale_data(
    monkeypatch,
):
    from gestur import pose_detector

    release_read = threading.Event()

    class ReadOnceCamera(Camera):
        def read(self):
            if self.count == 0:
                self.count += 1
                return True, SimpleNamespace(shape=(480, 640, 3), number=self.count)
            release_read.wait(2)
            return False, None

    sample = empty_part(head=True)
    sample.update(detected=True, x=0.7, y=0.4, scale=0.3)
    monkeypatch.setattr(
        pose_detector,
        "pose_features",
        lambda *args: {"head": sample, "torso": empty_part()},
    )
    # This expiry test supplies features directly, without landmark inference.
    monkeypatch.setattr(
        pose_detector.PrimaryPersonLock, "update_pose", lambda *args: True
    )
    tracker, _, _ = setup(monkeypatch, use_hands=False)
    monkeypatch.setattr(tracker, "_open_camera", ReadOnceCamera)
    publications = []
    tracker.subscribe(lambda data: publications.append(data["head"]["detected"]))
    tracker.run()
    try:
        wait_until(lambda: publications == [True, False])
        time.sleep(0.08)
        assert publications == [True, False]
        assert tracker.get_metrics()["pose_frames"] == 1
    finally:
        release_read.set()
        tracker.stop()


def test_shared_budget_limits_combined_pose_and_hand_work(monkeypatch):
    starts, ends = [], []

    class TimedModel(Model):
        def __init__(self, record_start=False, record_end=False):
            super().__init__(delay=0.015)
            self.record_start, self.record_end = record_start, record_end

        def detect_for_video(self, frame, timestamp):
            if self.record_start:
                starts.append(time.monotonic())
            result = super().detect_for_video(frame, timestamp)
            if self.record_end:
                ends.append(time.monotonic())
            return result

    tracker = PoseHandTracker(
        inference_fps=120, hand_fps=120, inference_duty=0.5, idle_after_seconds=100
    )
    backend = Backend()
    backend.pose, backend.hands = (
        TimedModel(record_start=True),
        TimedModel(record_end=True),
    )
    monkeypatch.setattr(tracker, "_build_backend", lambda: backend)
    monkeypatch.setattr(tracker, "_open_camera", Camera)
    tracker.run()
    wait_until(lambda: tracker.get_metrics()["hand_frames"] >= 6)
    tracker.stop()
    assert len(starts) == len(ends)
    for index in range(len(starts) - 1):
        combined_work = ends[index] - starts[index]
        assert starts[index + 1] - starts[index] >= combined_work / 0.5 - 0.003
    metrics = tracker.get_metrics()
    assert metrics["budget_pauses"] > 0
    assert metrics["budget_pause_seconds"] > 0.1
    assert metrics["inference_wall_seconds"] >= len(ends) * 0.03
    assert metrics["inference_duty_limit"] == 0.5


def test_each_model_enters_idle_probing_and_recovers_independently(monkeypatch):
    tracker = PoseHandTracker(
        inference_fps=24, hand_fps=15, idle_fps=3, idle_after_seconds=2
    )
    # Configure the scheduler without a camera or a running worker.
    tracker._metrics = {}
    due = {"pose": 0, "hand": 0}
    last = {"pose": 0, "hand": 0}
    tracker._schedule_model("pose", True, 2.1, 2.1, 2.12, due, last)
    tracker._schedule_model("hand", False, 2.1, 2.1, 2.12, due, last)
    assert tracker._scheduled_rates == {"pose": 24, "hand": 3}
    assert due["pose"] == pytest.approx(2.1 + 1 / 24)
    assert due["hand"] == pytest.approx(2.1 + 1 / 3)
    assert tracker.get_metrics()["hand_idle"] is True
    tracker._schedule_model("hand", True, 2.45, 2.45, 2.47, due, last)
    assert tracker._scheduled_rates == {"pose": 24, "hand": 15}
    assert due["hand"] == pytest.approx(2.45 + 1 / 15)
    assert tracker.get_metrics()["hand_idle"] is False


def test_idle_probe_is_bounded_and_empty_capture_keeps_no_phantom_detection(
    monkeypatch,
):
    tracker, _, _ = setup(
        monkeypatch, use_hands=False, idle_fps=3, idle_after_seconds=0
    )
    tracker.run()
    wait_until(lambda: tracker.get_metrics()["pose_frames"] >= 2)
    tracker.stop()
    metrics = tracker.get_metrics()
    assert metrics["pose_idle"] is True
    assert metrics["pose_scheduled_fps"] == 3
    assert metrics["pose_frames"] == 2
    assert 2 / 3 <= tracker._expiry_timeouts["pose"] <= 0.8
    assert not tracker.get_current_data()["head"]["detected"]


@pytest.mark.parametrize(
    "parts,head_present,torso_present,idle",
    [
        (("torso",), False, True, False),
        (("head", "torso"), False, True, False),
        (("head",), False, True, True),
        (("torso",), True, False, True),
    ],
)
def test_pose_idle_cadence_uses_only_parts_requested_by_controls(
    monkeypatch, parts, head_present, torso_present, idle
):
    from gestur import pose_detector

    head, torso = empty_part(head=True), empty_part()
    head.update(detected=head_present, x=0.5, y=0.4, scale=0.3)
    torso.update(detected=torso_present, x=0.5, y=0.6, scale=0.6)
    monkeypatch.setattr(
        pose_detector, "pose_features", lambda *args: {"head": head, "torso": torso}
    )
    monkeypatch.setattr(
        pose_detector.PrimaryPersonLock, "update_pose", lambda *args: True
    )
    tracker, _, _ = setup(
        monkeypatch, use_hands=False, pose_parts=parts, idle_after_seconds=0
    )
    tracker.run()
    try:
        wait_until(lambda: tracker.get_metrics()["pose_frames"] >= 2)
        metrics = tracker.get_metrics()
        assert metrics["pose_idle"] is idle
        assert metrics["pose_scheduled_fps"] == (3 if idle else 30)
        assert tracker.get_current_data()["torso"]["scale"] == (
            0.6 if torso_present else None
        )
    finally:
        tracker.stop()


def test_idle_hand_expiry_cannot_extend_head_detection():
    tracker = PoseHandTracker()
    tracker._expiry_timeouts = {"pose": 0.25, "hand": 2 / 3}
    for key in ("head", "torso", "left_hand"):
        sample = empty_part(head=key == "head", hand=key == "left_hand")
        sample.update(detected=True, x=0.5, y=0.5, scale=0.7)
        tracker._filter.update(key, sample, 0)
    assert tracker._expire_tracking(0.3)
    assert not tracker._filter.data["head"]["detected"]
    assert not tracker._filter.data["torso"]["detected"]
    assert tracker._filter.data["torso"]["scale"] is None
    assert tracker._filter.data["left_hand"]["detected"]
    assert tracker._filter.data["left_hand"]["scale"] == 0.7
    assert not tracker._expire_tracking(0.4)
    assert tracker._expire_tracking(0.7)
    assert tracker._filter.data["left_hand"]["scale"] is None


def test_fast_hand_cycle_cannot_shorten_expiry_of_budget_limited_head():
    tracker = PoseHandTracker(inference_fps=24, hand_fps=15, inference_duty=0.6)
    tracker._update_expiry_limits(["pose"], 1, 0.2)
    assert tracker._expiry_timeouts["pose"] == pytest.approx(2 / 3)
    tracker._update_expiry_limits(["hand"], 1.35, 0.01)
    assert tracker._expiry_timeouts["pose"] == pytest.approx(2 / 3)
    assert tracker._expiry_timeouts["hand"] == 0.25
    tracker._update_expiry_limits(["pose"], 1.5, 0.2)
    assert tracker._expiry_timeouts["pose"] == 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inference_fps": 0},
        {"width": 0},
        {"hand_fps": float("nan")},
        {"visibility_threshold": 2},
        {"head_scale_min": 1, "head_scale_max": 0},
        {"inference_duty": 0},
        {"inference_duty": 1.1},
        {"idle_fps": 0},
        {"idle_after_seconds": -1},
        {"idle_after_seconds": float("nan")},
        {"pose_parts": ()},
        {"pose_parts": "torso"},
        {"pose_parts": ("feet",)},
    ],
)
def test_invalid_configuration_fails_before_allocating_resources(kwargs):
    with pytest.raises(ValueError):
        PoseHandTracker(**kwargs)
