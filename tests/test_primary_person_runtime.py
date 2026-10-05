"""The ownership gate runs before smoothing; masking preserves camera geometry."""

from types import SimpleNamespace

import numpy as np
import pytest

from gestur.pose_detector import PoseHandTracker, _MediaPipeBackend
from gestur.tracking_geometry import empty_part


def pose(center):
    points = [
        SimpleNamespace(x=center, y=0.5, z=0.0, visibility=0.0, presence=0.0)
        for _ in range(33)
    ]
    locations = {
        0: (0, 0.23),
        2: (-0.025, 0.21),
        5: (0.025, 0.21),
        7: (-0.055, 0.24),
        8: (0.055, 0.24),
        11: (-0.09, 0.38),
        12: (0.09, 0.38),
        23: (-0.065, 0.7),
        24: (0.065, 0.7),
        13: (-0.12, 0.5),
        14: (0.12, 0.5),
        15: (-0.15, 0.6),
        16: (0.15, 0.6),
    }
    for index, (dx, y) in locations.items():
        points[index] = SimpleNamespace(
            x=center + dx, y=y, z=0.0, visibility=1.0, presence=1.0
        )
    world = [
        SimpleNamespace(
            x=p.x - center,
            y=p.y - 0.5,
            z=0.0,
            visibility=p.visibility,
            presence=p.presence,
        )
        for p in points
    ]
    world[0].z = -0.1
    return SimpleNamespace(pose_landmarks=[points], pose_world_landmarks=[world])


def test_intruder_never_reaches_filters_or_moves_region_and_handoff_clears_history():
    tracker = PoseHandTracker(smoothing_time_ms=500)
    first = tracker._update_pose(pose(0.24), 1.0, 4 / 3)
    assert first["head"]["detected"]
    assert tracker._filter.data["head"]["x"] == pytest.approx(0.24)
    region = tracker._primary.region(1.0)
    generation = tracker._primary.generation
    hand = empty_part(hand=True)
    hand.update(detected=True, x=0.3, y=0.5)
    tracker._filter.update("left_hand", hand, 1.0)
    intruder = tracker._update_pose(pose(0.78), 1.1, 4 / 3)
    assert not intruder["head"]["detected"]
    assert not tracker._filter.data["head"]["detected"]
    assert not tracker._filter.data["torso"]["detected"]
    # An already measured hand stays valid until its independent next result
    # or expiry. A lost body cannot erase another detector's current sample.
    assert tracker._filter.data["left_hand"]["detected"]
    assert tracker._primary.region(1.1) == region
    assert tracker._primary.generation == generation
    assert tracker._metrics["rejected_pose_frames"] == 1
    returned = tracker._update_pose(pose(0.26), 1.2, 4 / 3)
    assert returned["head"]["detected"]
    assert tracker._primary.generation == generation
    # After the grace period a new participant starts with their own values,
    # without the former participant's long smoothing history or hand output.
    assert tracker._primary.region(3.0) is None
    replacement = tracker._update_pose(pose(0.78), 3.0, 4 / 3)
    assert replacement["head"]["detected"]
    assert tracker._primary.generation > generation
    assert tracker._filter.data["head"]["x"] == pytest.approx(0.78)
    assert not tracker._filter.data["left_hand"]["detected"]


@pytest.mark.parametrize("mirror", [False, True])
def test_mask_keeps_canvas_coordinates_and_source_frame_unchanged(mirror):
    cv2 = pytest.importorskip("cv2")
    backend = _MediaPipeBackend.__new__(_MediaPipeBackend)
    backend.cv2 = cv2
    backend.mp = SimpleNamespace(
        Image=lambda **kwargs: kwargs["data"], ImageFormat=SimpleNamespace(SRGB="srgb")
    )
    frame = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    original = frame.copy()
    expected = cv2.cvtColor(cv2.flip(frame, 1) if mirror else frame, cv2.COLOR_BGR2RGB)
    result = backend.image(frame, mirror, (0.25, 0.25, 0.75, 0.75))
    assert result.shape == frame.shape
    assert np.array_equal(result[2:6, 3:9], expected[2:6, 3:9])
    assert not result[:2].any() and not result[6:].any()
    assert not result[:, :3].any() and not result[:, 9:].any()
    assert np.array_equal(frame, original)
    assert np.array_equal(backend.image(frame, mirror), expected)


def test_new_hands_cannot_control_without_a_pose_owner_or_during_its_loss(monkeypatch):
    tracker = PoseHandTracker()
    # A candidate with valid anatomical label but no selected person must be
    # rejected before gesture extraction, regardless of its classification score.
    points = [
        SimpleNamespace(x=0.3 + i * 0.002, y=0.5 + i * 0.001, z=0.0) for i in range(21)
    ]
    result = SimpleNamespace(
        hand_landmarks=[points],
        hand_world_landmarks=[points],
        handedness=[[SimpleNamespace(category_name="Left", score=0.99)]],
    )
    monkeypatch.setattr(
        "gestur.pose_detector.hand_features",
        lambda *args: pytest.fail("unowned hand was processed"),
    )
    tracker._update_hands(result, 1.0, 4 / 3)
    assert not tracker._filter.data["left_hand"]["detected"]
    tracker._update_pose(pose(0.24), 1.1, 4 / 3)
    tracker._update_pose(
        SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[]), 1.2, 4 / 3
    )
    tracker._update_hands(result, 1.2, 4 / 3)
    assert not tracker._filter.data["left_hand"]["detected"]
    assert tracker._metrics["rejected_hand_candidates"] == 2
