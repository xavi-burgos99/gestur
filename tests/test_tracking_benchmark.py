from types import SimpleNamespace

import pytest

import pose_hand_tracker as benchmark_module
from tracking_geometry import empty_data


def test_summary_uses_model_counters_not_number_of_validity_samples():
    report = benchmark_module.summarize(
        {'captured_frames':300,'pose_frames':120,'hand_frames':50,'capture_failures':2},
        10,15,dict(head=15,torso=10,left_hand=0,right_hand=5),20,[10,20,30],[12,24,36],
        startup_seconds=2)
    assert report['fps'] == {'capture':30,'pose':12,'hands':5}
    assert report['cpu_percent_one_core'] == 150
    assert report['validity_sample_fraction']['head'] == .75
    assert report['sampled_inference_ms'] == {'samples':3,'median':20,'p95':30,'maximum':30}


class FakeClock:
    def __init__(self):
        self.now = 0
    def monotonic(self):
        return self.now
    def process_time(self):
        return self.now / 2
    def sleep(self,seconds):
        self.now += seconds


class FakeTracker:
    def __init__(self,clock,fail_after=None):
        self.clock = clock
        self.fail_after = fail_after
        self.running = False
        self.last_error = None
        self.stopped = False
    def run(self):
        self.running = True
    def stop(self):
        self.stopped = True
        self.running = False
    def get_metrics(self):
        if self.fail_after is not None and self.clock.now >= self.fail_after:
            self.last_error = RuntimeError('camera disconnected')
        return {'pose_frames':int(self.clock.now*10),'hand_frames':int(self.clock.now*5),
                'captured_frames':int(self.clock.now*30),'inference_ms':10,'frame_age_ms':12}
    def get_current_data(self):
        return empty_data()


def test_benchmark_has_bounded_duration_without_hardware(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(benchmark_module,'time',clock)
    tracker = FakeTracker(clock)
    report = benchmark_module.benchmark(tracker,seconds=1)
    assert clock.now == pytest.approx(1)
    assert report['elapsed_seconds'] == pytest.approx(1)
    assert report['counts']['pose_frames'] == 10
    assert report['status'] == 'completed'
    assert tracker.stopped


def test_benchmark_stops_early_on_tracker_failure(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(benchmark_module,'time',clock)
    tracker = FakeTracker(clock,fail_after=.1)
    report = benchmark_module.benchmark(tracker,seconds=30)
    assert report['elapsed_seconds'] < 1
    assert report['status'] == 'error'
    assert report['error'] == 'camera disconnected'
    assert tracker.stopped
