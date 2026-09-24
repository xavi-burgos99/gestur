import threading
import time
from types import SimpleNamespace

import pytest

from pose_detector import PoseHandTracker


class Camera:
    def __init__(self, fail=False):
        self.released = False
        self.count = 0
        self.fail = fail

    def read(self):
        time.sleep(.002)
        self.count += 1
        if self.fail:
            return False, None
        return True, SimpleNamespace(shape=(480,640,3), number=self.count)

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
            raise RuntimeError('inference failure')
        time.sleep(self.delay)
        self.timestamps.append(timestamp)
        self.frames.append(frame.number)
        return SimpleNamespace(pose_landmarks=[],pose_world_landmarks=[],
                               hand_landmarks=[],hand_world_landmarks=[],handedness=[])


class Backend:
    def __init__(self, fail=False, delay=0):
        self.pose = Model(fail=fail,delay=delay)
        self.hands = Model()
        self.closed = False

    def image(self, frame, mirror):
        return frame

    def close(self):
        self.closed = True


def wait_until(condition, timeout=2):
    deadline = time.monotonic()+timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(.005)
    assert condition()


def setup(monkeypatch, **kwargs):
    tracker = PoseHandTracker(inference_fps=30,hand_fps=10,**kwargs)
    backends, cameras = [], []
    def backend():
        backends.append(Backend());return backends[-1]
    def camera():
        cameras.append(Camera());return cameras[-1]
    monkeypatch.setattr(tracker,'_build_backend',backend)
    monkeypatch.setattr(tracker,'_open_camera',camera)
    return tracker,backends,cameras


def test_restart_creates_fresh_models_and_closes_all_resources(monkeypatch):
    tracker,backends,cameras = setup(monkeypatch)
    tracker.stop()  # stop before run is harmless
    for iteration in range(2):
        tracker.run()
        wait_until(lambda: tracker.get_metrics()['pose_frames'] >= 3)
        tracker.stop()
        assert not tracker.running
        assert backends[iteration].closed and cameras[iteration].released
        assert not tracker.thread.is_alive() and not tracker.capture_thread.is_alive()
        timestamps = backends[iteration].pose.timestamps
        assert all(b>a for a,b in zip(timestamps,timestamps[1:]))
    tracker.stop()


def test_inference_uses_latest_frame_and_hands_have_independent_cadence(monkeypatch):
    tracker,backends,cameras = setup(monkeypatch)
    tracker.run()
    wait_until(lambda:tracker.get_metrics()['pose_frames']>=8)
    tracker.stop()
    pose = backends[0].pose
    assert len(pose.frames) >= 2*len(backends[0].hands.frames)
    assert any(b-a>2 for a,b in zip(pose.frames,pose.frames[1:]))
    assert tracker._latest_frame[0] == cameras[0].count
    assert tracker.get_metrics()['captured_frames'] > tracker.get_metrics()['pose_frames']


def test_camera_start_failure_closes_allocated_models(monkeypatch):
    tracker,backends,_ = setup(monkeypatch)
    def fail():
        raise RuntimeError('camera unavailable')
    monkeypatch.setattr(tracker,'_open_camera',fail)
    with pytest.raises(RuntimeError,match='camera unavailable'):
        tracker.run()
    assert backends[0].closed
    assert not tracker.running
    tracker.stop()


def test_capture_failure_is_bounded_and_releases_resources(monkeypatch):
    tracker,backends,_ = setup(monkeypatch)
    camera = Camera(fail=True)
    monkeypatch.setattr(tracker,'_open_camera',lambda:camera)
    tracker.run()
    wait_until(lambda:not tracker.running)
    assert tracker.get_metrics()['capture_failures'] == 10
    assert tracker.last_error is not None
    assert backends[0].closed and camera.released


def test_model_failure_closes_capture_and_clears_detection(monkeypatch):
    tracker,_,cameras = setup(monkeypatch)
    backend = Backend(fail=True)
    monkeypatch.setattr(tracker,'_build_backend',lambda:backend)
    tracker.run()
    wait_until(lambda:not tracker.running)
    assert str(tracker.last_error) == 'inference failure'
    assert backend.closed and cameras[0].released
    assert all(not part['detected'] for part in tracker.get_current_data().values())


def test_listener_mutation_is_isolated_and_stop_inside_callback_is_safe(monkeypatch):
    tracker,backends,cameras = setup(monkeypatch)
    observed = []
    def mutate(snapshot):
        snapshot['head']['x'] = 'corrupt'
    def observe(snapshot):
        observed.append(snapshot['head']['x'])
        tracker.stop()
    tracker.subscribe(mutate)
    tracker.subscribe(observe)
    tracker.run()
    wait_until(lambda:bool(observed) and not tracker.running)
    assert all(value is None for value in observed)
    assert tracker.get_current_data()['head']['x'] is None
    assert backends[0].closed and cameras[0].released


def test_disabled_tracking_never_opens_hardware(monkeypatch):
    tracker,backends,cameras = setup(monkeypatch,use_pose=False,use_hands=False)
    tracker.run()
    assert not tracker.running
    assert backends == cameras == []


@pytest.mark.parametrize('kwargs',[{'inference_fps':0},{'width':0},{'hand_fps':float('nan')},
                                    {'visibility_threshold':2},{'head_scale_min':1,'head_scale_max':0}])
def test_invalid_configuration_fails_before_allocating_resources(kwargs):
    with pytest.raises(ValueError):
        PoseHandTracker(**kwargs)
