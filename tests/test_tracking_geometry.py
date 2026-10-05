import math
from types import SimpleNamespace as Point

import pytest

from tracking_geometry import (
    TrackingFilter,
    anatomical_hand,
    empty_part,
    hand_features,
    pose_features,
    smooth_value,
    wrap_angle,
)


def point(x=0, y=0, z=0, visibility=1, presence=1):
    return Point(x=x, y=y, z=z, visibility=visibility, presence=presence)


def hand():
    points = [point() for _ in range(21)]
    points[0] = point(0, 0.04)
    points[1], points[2], points[3], points[4] = [
        point(-0.07, v) for v in (0.03, 0.01, -0.01, -0.04)
    ]
    for base, x, tip_y in (
        (5, -0.035, -0.095),
        (9, 0, -0.11),
        (13, 0.02, -0.09),
        (17, 0.04, -0.065),
    ):
        for offset in range(4):
            points[base + offset] = point(x, -0.005 + (tip_y + 0.005) * offset / 3)
    return points


def project(points, magnification=1, aspect=4 / 3):
    """Weak-perspective camera; z translation is absent from object-centred data."""
    return [
        point(0.5 + magnification * p.x, 0.5 + magnification * p.y * aspect, p.z)
        for p in points
    ]


def torso_pose():
    points = pose()
    for index, x, y in (
        (11, -0.2, -0.4),
        (12, 0.2, -0.4),
        (23, -0.15, 0),
        (24, 0.15, 0),
    ):
        points[index] = point(x, y)
    return points


def rotate(points, axis, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    result = []
    for p in points:
        if axis == "z":
            result.append(point(c * p.x - s * p.y, s * p.x + c * p.y, p.z))
        elif axis == "y":
            result.append(point(c * p.x + s * p.z, p.y, -s * p.x + c * p.z))
        else:
            result.append(point(p.x, c * p.y - s * p.z, s * p.y + c * p.z))
    return result


def test_open_hand_has_neutral_orientation():
    features = hand_features(hand(), hand(), "Right", aspect=1)
    assert features["detected"]
    assert features["pitch"] == pytest.approx(0)
    assert features["yaw"] == pytest.approx(0)
    assert features["rotation"] == pytest.approx(0)
    assert features["gesture"] == "open"
    assert features["openness"] == 1


@pytest.mark.parametrize("degrees", [-179, -90, -30, 30, 90, 179])
def test_signed_image_rotation_is_not_limited_to_90_degrees(degrees):
    world = rotate(hand(), "z", degrees)
    features = hand_features(world, world, "Right", aspect=1)
    assert features["rotation"] == pytest.approx(degrees)
    assert features["gesture"] == "open"


@pytest.mark.parametrize("axis,field,angle", [("x", "pitch", 40), ("y", "yaw", -50)])
def test_world_palm_rotation_survives_non_square_image(axis, field, angle):
    world = rotate(hand(), axis, angle)
    # Changing image aspect/coordinates cannot change metric palm normal.
    normalized = [point(p.x / 2, p.y, p.z) for p in world]
    features = hand_features(normalized, world, "Right", aspect=2)
    assert features[field] == pytest.approx(angle if axis == "x" else -angle)


def rigid_orientation(points, pitch, yaw, roll):
    """Apply the documented Rz(roll) @ Ry(-yaw) @ Rx(pitch) convention."""
    return rotate(rotate(rotate(points, "x", pitch), "y", -yaw), "z", roll)


@pytest.mark.parametrize(
    "pitch,yaw,roll",
    [
        (0, 0, 0),
        (40, 0, 0),
        (0, -50, 0),
        (0, 0, 120),
        (35, 45, 60),
        (-45, 35, -100),
        (70, -55, 160),
        (-120, 70, -175),
        (12, 89, 30),
        (-40, -89, -120),
    ],
)
@pytest.mark.parametrize("label", ["Left", "Right"])
def test_palm_frame_separates_all_three_axes_for_rigid_rotations(
    pitch, yaw, roll, label
):
    world = hand()
    if label == "Left":
        world = [point(-p.x, p.y, p.z) for p in world]
    world = rigid_orientation(world, pitch, yaw, roll)
    normalized = [point(p.x / 2 + 0.5, p.y + 0.5, p.z) for p in world]
    features = hand_features(normalized, world, label, aspect=2)
    assert features["detected"]
    assert [features[name] for name in ("pitch", "yaw", "roll")] == pytest.approx(
        [pitch, yaw, roll]
    )
    assert features["gesture"] == "open"


def test_mirror_and_model_handedness_have_consistent_orientation_signs():
    world = rigid_orientation(hand(), 30, 45, 50)
    original = hand_features(world, world, "Right", aspect=1)
    mirrored = [point(-p.x, p.y, p.z) for p in world]
    changed = hand_features(mirrored, mirrored, "Left", aspect=1)
    assert changed["pitch"] == pytest.approx(original["pitch"])
    for field in ("yaw", "roll", "rotation"):
        assert changed[field] == pytest.approx(-original[field])
    assert changed["pinch"] == pytest.approx(original["pinch"])
    assert changed["scale"] == pytest.approx(original["scale"])


def test_knuckle_skew_along_fingers_does_not_change_palm_frame():
    world = hand()
    world[17].y -= 0.01
    world = rigid_orientation(world, 35, 40, -60)
    features = hand_features(world, world, "Right", aspect=1)
    assert [features[name] for name in ("pitch", "yaw", "roll")] == pytest.approx(
        [35, 40, -60]
    )


@pytest.mark.parametrize("yaw", [-90, 90])
def test_edge_on_palm_uses_reproducible_euler_singularity_convention(yaw):
    pitch, roll = 30, 20
    world = rigid_orientation(hand(), pitch, yaw, roll)
    features = hand_features(world, world, "Right", aspect=1)
    assert features["detected"]
    assert features["yaw"] == pytest.approx(yaw)
    assert features["roll"] == 0
    expected_pitch = pitch + (roll if yaw > 0 else -roll)
    assert features["pitch"] == pytest.approx(expected_pitch)
    recovered = rigid_orientation(
        hand(), features["pitch"], features["yaw"], features["roll"]
    )
    for actual, expected in zip(recovered, world):
        assert (actual.x, actual.y, actual.z) == pytest.approx(
            (expected.x, expected.y, expected.z)
        )


def test_fingers_toward_camera_only_invalidate_the_projected_rotation():
    world = rigid_orientation(hand(), 90, 0, 0)
    features = hand_features(world, world, "Right", aspect=1)
    assert features["detected"]
    assert features["pitch"] == pytest.approx(90)
    assert features["yaw"] == pytest.approx(0)
    assert features["roll"] == pytest.approx(0)
    assert features["rotation"] is None
    assert features["gesture"] == "open"
    assert features["pinch"] is not None


def test_image_rotation_keeps_legacy_behavior_and_is_distinct_from_3d_roll():
    world = rigid_orientation(hand(), 45, 50, 30)
    features = hand_features(world, world, "Right", aspect=1)
    dx, dy = world[9].x - world[0].x, world[9].y - world[0].y
    assert features["rotation"] == pytest.approx(math.degrees(math.atan2(dx, -dy)))
    assert features["roll"] == pytest.approx(30)
    assert abs(features["rotation"] - features["roll"]) > 20


def test_nearly_collinear_palm_and_unknown_handedness_are_rejected():
    world = hand()
    world[5], world[17] = point(0, -0.03), point(1e-8, -0.09)
    assert not hand_features(world, world, "Right", aspect=1)["detected"]
    assert not hand_features(hand(), hand(), "Unknown", aspect=1)["detected"]


def test_3d_roll_smoothing_crosses_wrap_without_neutral_sample_then_clears_loss():
    state = TrackingFilter(smoothing_time=0.1)
    first = rotate(hand(), "z", 179)
    second = rotate(hand(), "z", -179)
    state.update("right_hand", hand_features(first, first, "Right", 1), 1)
    state.update("right_hand", hand_features(second, second, "Right", 1), 1.1)
    assert abs(state.snapshot()["right_hand"]["roll"]) > 179
    state.update("right_hand", empty_part(hand=True), 1.2)
    assert state.snapshot()["right_hand"]["roll"] is None
    assert state.snapshot()["right_hand"]["detected"] is False


def test_pinch_is_scale_and_rotation_invariant():
    world = hand()
    world[4] = point(world[8].x + 0.001, world[8].y, world[8].z)
    features = hand_features(world, world, "Right", aspect=1)
    rotated = rotate(world, "z", 100)
    scaled = [point(p.x * 5, p.y * 5, p.z * 5) for p in rotated]
    changed = hand_features(scaled, scaled, "Right", aspect=1)
    assert features["gesture"] == changed["gesture"] == "pinch"
    assert features["pinch"] == pytest.approx(changed["pinch"])


def test_folded_fingers_are_fist():
    points = hand()
    for base in (5, 9, 13, 17):
        points[base + 2] = point(points[base].x, 0.005, -0.015)
        points[base + 3] = point(points[base].x, 0.025, -0.01)
    result = hand_features(points, points, "Right", aspect=1)
    assert result["gesture"] == "fist"
    assert result["openness"] == 0


@pytest.mark.parametrize(
    "mirror,invert,result",
    [
        (True, False, "Right"),
        (False, False, "Left"),
        (True, True, "Left"),
        (False, True, "Right"),
    ],
)
def test_tasks_handedness_accounts_for_mirror_and_explicit_inversion(
    mirror, invert, result
):
    assert anatomical_hand("Left", mirror, invert) == result


def test_missing_or_degenerate_landmarks_never_detect_hand():
    assert not hand_features([], [], "Right")["detected"]
    points = [point() for _ in range(21)]
    assert not hand_features(points, points, "Right")["detected"]
    points = hand()
    points[8].x = float("nan")
    assert not hand_features(points, points, "Right")["detected"]


def test_circular_smoothing_crosses_wrap_via_180_not_zero():
    result = smooth_value(179, -179, 0.1, 0.1, circular=True)
    assert result < -179 or result > 179
    assert smooth_value(0, 10, 0.1, 0.1) == pytest.approx(6.3212056)
    assert smooth_value(0.4, 0, 0.1, 0.1) < 0.4


def test_zero_smoothing_detects_real_hand_on_first_frame_then_clears_loss():
    state = TrackingFilter(smoothing_time=0)
    state.update("left_hand", hand_features(hand(), hand(), "Right", 1), 1)
    assert state.snapshot()["left_hand"]["detected"]
    assert state.snapshot()["left_hand"]["rotation"] == 0
    state.update("left_hand", empty_part(hand=True), 1.02)
    assert state.snapshot()["left_hand"]["detected"] is False
    assert state.snapshot()["left_hand"]["rotation"] is None


def test_no_phantom_hands_on_start_and_stale_results_expire():
    state = TrackingFilter()
    state.expire(10)
    assert not state.snapshot()["left_hand"]["detected"]
    state.update("left_hand", hand_features(hand(), hand(), "Right", 1), 10)
    state.expire(10.1)
    assert state.snapshot()["left_hand"]["detected"]
    state.expire(10.3)
    assert not state.snapshot()["left_hand"]["detected"]


def test_filter_is_independent_of_sampling_rate():
    final = []
    for fps in (10, 30, 60):
        state = TrackingFilter(smoothing_time=0.3, timeout=2)
        sample = empty_part()
        sample.update(detected=True, x=0, y=0, pitch=0, yaw=0, roll=0)
        state.update("torso", sample, 0)
        sample["x"] = 1
        for i in range(1, fps + 1):
            state.update("torso", sample, i / fps)
        final.append(state.snapshot()["torso"]["x"])
    assert final == pytest.approx([final[0]] * 3)


def pose():
    points = [point(0.5, 0.5, 0) for _ in range(33)]
    points[0] = point(0.5, 0.25, -0.1)
    points[1] = point(0.46, 0.22, 0)
    points[2] = point(0.47, 0.22, 0)
    points[5] = point(0.53, 0.22, 0)
    points[7] = point(0.42, 0.25, 0)
    points[8] = point(0.58, 0.25, 0)
    points[11] = point(0.3, 0.4, 0)
    points[12] = point(0.7, 0.4, 0)
    points[23] = point(0.4, 0.8, 0)
    points[24] = point(0.6, 0.8, 0)
    return points


def test_head_scale_uses_both_eye_centres_and_preserves_zero_to_one_range():
    points = pose()
    result = pose_features(points, points)
    assert result["head"]["scale"] == pytest.approx((0.06 - 0.02) / 0.18)
    points[1].x = 0.4
    assert pose_features(points, points)["head"]["scale"] == result["head"]["scale"]
    points[5].x = 0.9
    assert pose_features(points, points)["head"]["scale"] == 1


def test_head_visibility_is_independent_of_torso_visibility():
    points = pose()
    points[23].visibility = 0.1
    result = pose_features(points, points)
    assert result["head"]["detected"]
    assert not result["torso"]["detected"]
    points[2].presence = 0.1
    assert not pose_features(points, points)["head"]["detected"]


@pytest.mark.parametrize("part", ["torso", "hand"])
@pytest.mark.parametrize("aspect", [1, 4 / 3, 16 / 9])
def test_proximity_increases_with_projected_size_without_using_world_translation(
    part, aspect
):
    world = torso_pose() if part == "torso" else hand()
    minimum, maximum, span = (
        (0.12, 0.80, 0.4) if part == "torso" else (0.025, 0.25, 0.075)
    )

    def extract(image, landmarks):
        return (
            pose_features(image, landmarks, aspect=aspect)["torso"]
            if part == "torso"
            else hand_features(image, landmarks, "Right", aspect=aspect)
        )

    scales = []
    for magnification in (0.1, 0.75, 1, 1.5, 5):
        image = project(world, magnification, aspect)
        value = extract(image, world)["scale"]
        assert value == pytest.approx(
            max(0, min(1, (span * magnification - minimum) / (maximum - minimum)))
        )
        # Landmark origins and estimated metric body/hand size are not camera depth.
        translated = [point(5 * p.x + 3, 5 * p.y - 4, 5 * p.z + 100) for p in world]
        assert extract(image, translated)["scale"] == pytest.approx(value)
        scales.append(value)
    assert scales == sorted(scales) and scales[0] == 0 and scales[-1] == 1


@pytest.mark.parametrize("part", ["torso", "hand"])
@pytest.mark.parametrize(
    "pitch,yaw,roll", [(0, 70, 0), (65, 0, 0), (0, 0, 120), (35, 60, -40)]
)
def test_proximity_compensates_rigid_foreshortening_and_roll(part, pitch, yaw, roll):
    base = torso_pose() if part == "torso" else hand()
    turned = rigid_orientation(base, pitch, yaw, roll)

    def extract(world):
        image = project(world, 1, 16 / 9)
        return (
            pose_features(image, world, aspect=16 / 9)["torso"]
            if part == "torso"
            else hand_features(image, world, "Right", aspect=16 / 9)
        )

    assert extract(turned)["detected"]
    assert extract(turned)["scale"] == pytest.approx(extract(base)["scale"])


def test_hand_proximity_uses_palm_not_finger_opening_or_pinch():
    opened = hand()
    closed = hand()
    for index in (4, 8, 12, 16, 20):
        closed[index] = point(0, 0)
    first = hand_features(project(opened), opened, "Right")
    second = hand_features(project(closed), closed, "Right")
    assert first["pinch"] != second["pinch"]
    assert first["openness"] != second["openness"]
    assert first["scale"] == pytest.approx(second["scale"])


@pytest.mark.parametrize("part", ["torso", "hand"])
def test_unobservable_proximity_does_not_discard_other_channels(part):
    world = torso_pose() if part == "torso" else hand()
    collapsed = [point(0.5, 0.5, p.z) for p in world]
    result = (
        pose_features(collapsed, world)["torso"]
        if part == "torso"
        else hand_features(collapsed, world, "Right")
    )
    assert result["detected"] and result["scale"] is None
    assert result["x"] == result["y"] == 0.5
    assert all(math.isfinite(result[key]) for key in ("pitch", "yaw", "roll"))
    if part == "hand":
        assert result["rotation"] is None
        assert result["gesture"] == "open" and result["pinch"] is not None
    if part == "torso":
        for index in (11, 12, 23, 24):
            world[index] = point()
        invalid = pose_features(collapsed, world)["torso"]
        assert not invalid["detected"] and invalid["scale"] is None


def test_torso_remains_available_with_occluded_head_and_signed_rotations():
    for axis, field, expected in (
        ("x", "pitch", 35),
        ("y", "yaw", -35),
        ("z", "roll", 35),
    ):
        world = rotate(torso_pose(), axis, 35)
        image = project(world)
        image[0].visibility = 0.1
        result = pose_features(image, world)
        assert not result["head"]["detected"] and result["torso"]["detected"]
        assert result["torso"][field] == pytest.approx(expected)
        assert result["torso"]["scale"] is not None


@pytest.mark.parametrize("part", ["torso", "left_hand", "right_hand"])
def test_proximity_filter_loss_expiry_and_reacquisition_are_independent(part):
    state = TrackingFilter(smoothing_time=0.06, timeout=0.25)
    sample = empty_part(hand="hand" in part)
    sample.update(detected=True, x=0.4, y=0.6, pitch=30, yaw=10, roll=20, scale=0.2)
    state.update(part, sample, 0)
    state.update(part, {**sample, "scale": 0.8}, 0.06)
    assert state.snapshot()[part]["scale"] == pytest.approx(
        0.2 + 0.6 * (1 - math.exp(-1))
    )
    state.update(part, {**sample, "scale": None}, 0.12)
    current = state.snapshot()[part]
    assert current["detected"] and current["scale"] is None and current["pitch"] == 30
    state.update(part, {**sample, "scale": 0.8}, 0.18)
    assert state.snapshot()[part]["scale"] == 0.8
    state.expire(0.5)
    assert (
        not state.snapshot()[part]["detected"]
        and state.snapshot()[part]["scale"] is None
    )
    state.update(part, {**sample, "scale": 0.7}, 0.6)
    assert state.snapshot()[part]["scale"] == 0.7


def test_torso_roll_filter_uses_unoriented_shoulder_line():
    state = TrackingFilter(smoothing_time=0.06)
    samples = []
    for angle in (89, 91):
        world = rotate(torso_pose(), "z", angle)
        samples.append(pose_features(project(world), world)["torso"])
    assert [s["roll"] for s in samples] == pytest.approx([89, -89])
    state.update("torso", samples[0], 0)
    state.update("torso", samples[1], 0.06)
    delta = wrap_angle(state.snapshot()["torso"]["roll"] - 89, 180)
    assert delta == pytest.approx(2 * (1 - math.exp(-1)))


def tilted_head(degrees):
    world = rotate(pose(), "z", degrees)
    return pose_features(world, world, aspect=1)["head"]


def test_head_roll_filter_respects_unoriented_eye_line_and_clears_loss():
    first, second = tilted_head(89), tilted_head(91)
    assert first["roll"] == pytest.approx(89)
    assert second["roll"] == pytest.approx(-89)
    state = TrackingFilter(smoothing_time=0.1)
    state.update("head", first, 0)
    state.update("head", second, 0.1)
    # The filter crosses the vertical line, not the upright pose at zero.
    assert -90 < state.snapshot()["head"]["roll"] < -89
    state.expire(0.4)
    assert state.snapshot()["head"]["detected"] is False
    assert state.snapshot()["head"]["roll"] is None
    state.update("head", tilted_head(0), 0.5)
    assert state.snapshot()["head"]["roll"] == pytest.approx(0)


def test_head_roll_control_crosses_eye_line_wrap_and_returns_upright_on_loss():
    from control_system import create_control_system
    from runtime_config import default_config

    now = 0.0
    config = default_config()
    config["controls"]["mappings"] = [
        {
            "id": "tilt",
            "input": "head_roll",
            "output": "rotation_roll",
            "mode": "absolute",
            "enabled": True,
            "scale": 180,
            "invert": False,
            "center": 0.5,
        }
    ]
    controls = create_control_system(config, clock=lambda: now)
    tracking = TrackingFilter(smoothing_time=0.1)
    tracking.update("head", tilted_head(89), now)
    first = controls.process_input(tracking.snapshot())["rotation"][2]
    now = 0.1
    tracking.update("head", tilted_head(91), now)
    second = controls.process_input(tracking.snapshot())["rotation"][2]
    assert first == pytest.approx(89)
    assert first < second < 91
    now = 0.4
    tracking.expire(now)
    after_loss = controls.process_input(tracking.snapshot())["rotation"][2]
    assert 0 < after_loss < second
    now = 5
    neutral = controls.process_input(tracking.snapshot())["rotation"][2]
    assert neutral == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("invert", [False, True])
def test_head_roll_reacquisition_preserves_output_turns_at_nondivisor_sensitivity(
    invert,
):
    from control_system import create_control_system
    from runtime_config import default_config

    now = 0.0
    config = default_config()
    config["controls"]["smoothing_ms"] = 0
    config["controls"]["mappings"] = [
        {
            "id": "tilt",
            "input": "head_roll",
            "output": "rotation_roll",
            "mode": "absolute",
            "enabled": True,
            "scale": 70,
            "invert": invert,
            "center": 0.5,
        }
    ]
    controls = create_control_system(config, clock=lambda: now)
    tracking = TrackingFilter(smoothing_time=0)
    # Each measured half-turn crosses the eye-line +/-90 degree boundary.
    for degrees in [0, 45, 89, 91, 135, 179] * 20 + [0]:
        now += 0.05
        tracking.update("head", tilted_head(degrees), now)
        previous = controls.process_input(tracking.snapshot())["rotation"][2]
    assert abs(previous) > 1000
    target = previous + wrap_angle(-previous)
    now += 5
    tracking.expire(now)
    controls.process_input(tracking.snapshot())
    now += 0.1
    neutral = controls.process_input(tracking.snapshot())["rotation"][2]
    assert neutral == pytest.approx(target)
    assert wrap_angle(neutral) == pytest.approx(0)
    now += 0.1
    tracking.update("head", tilted_head(0), now)
    acquired = controls.process_input(tracking.snapshot())["rotation"][2]
    assert acquired == pytest.approx(neutral)
    now += 0.1
    tracking.update("head", tilted_head(10), now)
    moved = controls.process_input(tracking.snapshot())["rotation"][2]
    assert moved == pytest.approx(neutral + 10 * 70 / 180 * (-1 if invert else 1))
