"""Deterministic tracking-loss, idle motion and re-acquisition checks."""

import math

import pytest

from control_system import create_control_system
from runtime_config import default_config


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def configuration(mode, mappings=None):
    config = default_config()
    config["controls"].update(idle_mode=mode, smoothing_ms=0, reset_timeout_seconds=2)
    if mappings is not None:
        config["controls"]["mappings"] = mappings
    return config


def absolute(input_name, output, scale=1, **kwargs):
    return dict(
        id=input_name,
        input=input_name,
        output=output,
        mode="absolute",
        enabled=True,
        scale=scale,
        invert=False,
        center=0.5,
        **kwargs,
    )


def head(x=0.8, y=0.7, scale=0.8):
    return {"head": {"detected": True, "x": x, "y": y, "scale": scale}}


def test_hold_freezes_every_channel_and_an_unfinished_zoom_transition():
    clock = Clock()
    config = configuration("hold")
    config["controls"]["mappings"].append(absolute("left_hand_x", "position_x", 8))
    system = create_control_system(config, clock)
    frame = {**head(), "left_hand": {"detected": True, "x": 0.8}}
    system.process_input(frame)
    clock.now = 0.375
    visible = system.process_input(frame)
    assert 1 < visible["scale"] < 1.75
    assert visible["position"][0] > 0
    for clock.now in (0.4, 1, 3, 10000):
        assert system.process_input({}) == visible
        assert system.get_idle_status() == {"mode": "hold", "active": True}
    # Reacquisition may complete the old stepped target internally, but the
    # first visible output must remain exactly the frozen intermediate zoom.
    clock.now += 0.01
    assert system.process_input(frame) == visible
    clock.now += 0.1
    resumed = system.process_input(frame)
    assert visible["scale"] < resumed["scale"] < 1.75


@pytest.mark.parametrize("mode", ["hold", "float"])
def test_partial_loss_freezes_only_missing_channel_and_never_starts_float(mode):
    clock = Clock()
    system = create_control_system(
        configuration(
            mode,
            [
                absolute("head_x", "position_x", 8),
                absolute("left_hand_scale", "scale_uniform", 2),
            ],
        ),
        clock,
    )
    visible = system.process_input(
        {**head(x=0.8), "left_hand": {"detected": True, "scale": 0.9}}
    )
    for clock.now in (1, 3, 10):
        output = system.process_input(head(x=0.7))
        assert output["position"][0] == pytest.approx(1.6)
        assert output["scale"] == visible["scale"]
        assert output["rotation"] == [0, 0, 0]
        assert system.get_idle_status()["active"] is False


@pytest.mark.parametrize("fps", [15, 30, 60, 120])
def test_float_delay_motion_and_amplitude_depend_on_elapsed_time_not_fps(fps):
    clock = Clock()
    system = create_control_system(configuration("float", []), clock)
    baseline = system.process_input({})
    for step in range(1, 10 * fps + 1):
        clock.now = step / fps
        output = system.process_input({})
        if clock.now < 2:
            assert output == baseline
            assert system.get_idle_status()["active"] is False
        assert abs(output["position"][1]) <= 0.12
        assert output["position"][0] == output["position"][2] == 0
        assert output["rotation"][0] == output["rotation"][1] == 0
        assert 0 <= output["rotation"][2] < 360
        assert output["scale"] == 1
    phase = 8
    assert output["rotation"][2] == pytest.approx(7 * (phase - 1 + math.exp(-phase)))
    assert output["position"][1] == pytest.approx(
        0.12 * (1 - math.exp(-phase)) * math.sin(phase * 0.7)
    )
    assert system.get_idle_status() == {"mode": "float", "active": True}
    clock.now = 20 * 86400
    output = system.process_input({})
    assert 0 <= output["rotation"][2] < 360 and abs(output["position"][1]) <= 0.12


def test_disabled_or_unusable_gesture_data_does_not_keep_float_awake():
    clock = Clock()
    disabled = absolute("head_x", "position_x")
    disabled["enabled"] = False
    system = create_control_system(
        configuration("float", [disabled, absolute("left_hand_x", "position_y")]), clock
    )
    for invalid in (None, float("nan"), float("inf"), True, "0.5"):
        clock.now += 3
        system.process_input({**head(), "left_hand": {"detected": True, "x": invalid}})
        assert system.get_idle_status()["active"] is True


@pytest.mark.parametrize("timeout", [0, 2])
def test_repeated_float_episodes_never_accumulate_vertical_drift(timeout):
    clock = Clock()
    config = configuration("float", [absolute("head_x", "position_x")])
    config["controls"]["reset_timeout_seconds"] = timeout
    system = create_control_system(config, clock)
    system.process_input(head())
    for _ in range(30):
        clock.now += timeout + math.pi / 1.4
        floating = system.process_input({})
        assert abs(floating["position"][1]) <= 0.12
        clock.now += 0.01
        reacquired = system.process_input(head())
        assert reacquired == floating
        clock.now += 0.8
        assert system.process_input(head())["position"][1] == pytest.approx(0)
    # Lose tracking again before the decorative return has settled, including
    # a zero-delay float. The old offset must not become the next baseline.
    for _ in range(100):
        clock.now += timeout + 2
        assert abs(system.process_input({})["position"][1]) <= 0.12
        clock.now += 0.01
        system.process_input(head())
        clock.now += 0.05
        assert abs(system.process_input(head())["position"][1]) <= 0.12


@pytest.mark.parametrize("timeout", [0, 2])
def test_flickering_vertical_tracking_keeps_decoration_separate_from_gesture_position(
    timeout,
):
    clock = Clock()
    config = configuration("float", [absolute("head_y", "position_y", 8)])
    config["controls"]["reset_timeout_seconds"] = timeout
    system = create_control_system(config, clock)
    system.process_input(head(y=0.5))
    for _ in range(100):
        clock.now += timeout + math.pi / 1.4
        floating = system.process_input({})
        assert abs(floating["position"][1]) <= 0.12
        clock.now += 0.01
        assert system.process_input(head(y=0.5)) == floating
        clock.now += 0.01
        assert abs(system.process_input(head(y=0.5))["position"][1]) <= 0.12
    # A sustained real gesture can still move beyond the decorative amplitude.
    clock.now += 1
    assert system.process_input(head(y=0.75))["position"][1] == pytest.approx(2)
    clock.now += timeout + 2
    assert 1.88 <= system.process_input({})["position"][1] <= 2.12


def test_float_hybrid_reacquisition_rebases_at_the_visible_angle():
    clock = Clock()
    config = configuration("float")
    config["controls"]["mappings"] = [config["controls"]["mappings"][0]]
    system = create_control_system(config, clock)
    system.process_input(head(x=1))
    for step in range(1, 61):
        clock.now = step / 60
        visible = system.process_input(head(x=1))
    clock.now = 1.1
    assert system.process_input({}) == visible
    clock.now = 15
    floating = system.process_input({})
    assert floating["rotation"][2] != visible["rotation"][2]
    clock.now += 1 / 60
    assert system.process_input(head(x=0.5)) == floating
    assert system.get_idle_status()["active"] is False
    for step in range(1, 61):
        clock.now += 1 / 60
        output = system.process_input(head(x=0.5))
        assert output["rotation"][2] == pytest.approx(floating["rotation"][2])
    clock.now += 0.05
    moved = system.process_input(head(x=1))
    assert abs(moved["rotation"][2] - floating["rotation"][2]) < 40


@pytest.mark.parametrize("mode", ["hold", "float"])
def test_circular_reacquisition_uses_the_short_arc_without_jump(mode):
    clock = Clock()
    system = create_control_system(
        configuration(
            mode,
            [
                absolute("left_hand_yaw", "rotation_yaw", 180),
            ],
        ),
        clock,
    )
    first = system.process_input({"left_hand": {"detected": True, "yaw": 179}})
    clock.now = 3
    missing = system.process_input({})
    assert missing["rotation"][0] == first["rotation"][0]
    clock.now = 3.1
    first_reacquired = system.process_input(
        {"left_hand": {"detected": True, "yaw": -179}}
    )
    assert first_reacquired == missing
    previous = first_reacquired["rotation"][0]
    for step in range(1, 31):
        clock.now = 3.1 + step / 60
        current = system.process_input({"left_hand": {"detected": True, "yaw": -179}})[
            "rotation"
        ][0]
        assert abs(current - previous) < 0.2
        previous = current
    assert previous == pytest.approx(181)


@pytest.mark.parametrize(
    "old_mode,new_mode",
    [("float", "hold"), ("hold", "float"), ("hold", "return"), ("return", "hold")],
)
def test_settings_changes_adopt_the_visible_pose_before_applying_new_controls(
    old_mode, new_mode
):
    clock = Clock()
    old = create_control_system(configuration(old_mode), clock)
    old.process_input(head())
    clock.now = 0.5
    visible = old.process_input(head())
    clock.now = 5
    visible = old.process_input({})
    new = create_control_system(configuration(new_mode), clock)
    new.adopt_output_state(old)
    assert new.process_input({}) == visible
    clock.now += 0.1
    if new_mode in ("hold", "float"):
        assert new.process_input({}) == visible
    # No mutable vector is shared between the old output and the replacement.
    visible["rotation"][0] = 999
    assert new.process_input({})["rotation"][0] != 999


def test_default_return_retains_legacy_delayed_hybrid_reset():
    clock = Clock()
    config = configuration("return")
    system = create_control_system(config, clock)
    system.process_input(head(x=1))
    clock.now = 0.1
    visible = system.process_input(head(x=1))["rotation"][2]
    clock.now = 1
    assert system.process_input({})["rotation"][2] == visible
    clock.now = 2.1
    assert system.process_input({})["rotation"][2] == visible
    clock.now = 2.6
    halfway = system.process_input({})["rotation"][2]
    assert abs(halfway) < abs(visible)
    clock.now = 3.1
    assert system.process_input({})["rotation"][2] == pytest.approx(0)


def test_adopt_accepts_visible_output_from_a_custom_control_system():
    clock = Clock()
    system = create_control_system(configuration("hold", []), clock)
    output = {"position": [1, 2, 3], "rotation": [10, 20, 30], "scale": 1.5}
    system.adopt_output_state(output)
    assert system.process_input({}) == output
    output["position"][1] = 999
    assert system.process_input({})["position"][1] == 2
