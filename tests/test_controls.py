import math
import unittest

from gestur.control_system import (
    DataProcessor,
    ExponentialSmoother,
    HybridRotationController,
    create_appliers,
    create_control_system,
    create_extractors,
)
from gestur.runtime_config import default_config as current_defaults


def neutral_config():
    """Use return-to-origin when testing recovery rather than decorative motion."""
    config = current_defaults()
    config["controls"]["idle_mode"] = "return"
    return config


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def head(x=0.5, y=0.5, scale=0.3):
    return {"head": {"detected": True, "x": x, "y": y, "scale": scale}}


class ControlTests(unittest.TestCase):
    def test_smoothing_response_is_independent_of_render_fps(self):
        results = []
        for fps in (24, 30, 60, 120):
            clock = Clock()
            smoother = ExponentialSmoother(smoothing_ms=250, clock=clock)
            smoother.update(0)
            for _ in range(fps):
                clock.advance(1 / fps)
                value = smoother.update(1)
            results.append(value)
        for result in results:
            self.assertAlmostEqual(result, 1 - math.exp(-4), places=10)

    def test_continuous_rotation_is_independent_of_render_fps(self):
        results = []
        for fps in (24, 30, 60, 120):
            clock = Clock()
            controller = HybridRotationController(
                continuous_speed_degrees_per_second=100, clock=clock
            )
            controller.update(1)
            for _ in range(2 * fps):
                clock.advance(1 / fps)
                value = controller.update(1)
            results.append(value)
        for result in results:
            self.assertAlmostEqual(result, 215, places=8)

    def test_default_capitell_channels_and_initial_acquisition(self):
        clock = Clock()
        system = create_control_system(neutral_config(), clock=clock)
        self.assertEqual(system.process_input({})["rotation"], [0, 0, 0])
        result = system.process_input(head(x=0.75, y=0.75, scale=0.1))
        self.assertEqual(result["rotation"], [-5, 15, -15])
        self.assertEqual(result["scale"], 1)
        self.assertEqual(result["position"], [0, 0, 0])

    def test_lost_tracking_stops_motion_then_resets(self):
        clock = Clock()
        system = create_control_system(neutral_config(), clock=clock)
        system.process_input(head(x=1))
        for _ in range(60):
            clock.advance(1 / 60)
            before = system.process_input(head(x=1))["rotation"][2]
        clock.advance(0.1)
        self.assertAlmostEqual(system.process_input({})["rotation"][2], before)
        clock.advance(1)
        self.assertAlmostEqual(system.process_input({})["rotation"][2], before)
        clock.advance(2)
        self.assertAlmostEqual(system.process_input({})["rotation"][2], before)
        clock.advance(0.5)
        middle = system.process_input({})["rotation"][2]
        self.assertLess(abs(middle), abs(before))
        clock.advance(0.5)
        self.assertAlmostEqual(system.process_input({})["rotation"][2], 0)

    def test_reset_takes_shortest_path_and_cancellation_has_no_jump(self):
        clock = Clock()
        controller = HybridRotationController(
            max_degrees=0, continuous_speed_degrees_per_second=350, clock=clock
        )
        controller.update(1)
        for _ in range(10):
            clock.advance(0.1)
            controller.update(1)
        self.assertAlmostEqual(controller.current_rotation, 350)
        controller.update(None)
        clock.advance(0.5)
        self.assertAlmostEqual(controller.update(None), 355)
        clock.advance(0.01)
        self.assertAlmostEqual(controller.update(0.5), 355)
        clock.advance(0.01)
        self.assertAlmostEqual(controller.update(0.5), 355)

    def test_zoom_hysteresis_and_time_based_transition(self):
        clock = Clock()
        apply = create_appliers(clock)["scale_stepped"](
            threshold=0.4,
            small_scale=1,
            large_scale=1.75,
            transition_time_ms=750,
            hysteresis=0.02,
        )
        output = {}
        apply(0.5, output)
        clock.advance(0.375)
        apply(0.5, output)
        self.assertAlmostEqual(output["scale"], 1.375)
        clock.advance(0.375)
        apply(0.39, output)  # Noise within the deadband keeps the current zoom.
        self.assertAlmostEqual(output["scale"], 1.75)
        apply(0.3, output)
        clock.advance(0.75)
        apply(0.3, output)
        self.assertEqual(output["scale"], 1)

    def test_circular_hand_roll_crosses_wrap_without_spinning(self):
        clock = Clock()
        config = neutral_config()
        config["controls"]["mappings"] = [
            {
                "id": "palm",
                "input": "left_hand_rotation",
                "output": "rotation_roll",
                "mode": "absolute",
                "enabled": True,
                "scale": 180,
                "invert": False,
                "center": 0.5,
            }
        ]
        system = create_control_system(config, clock)
        first = system.process_input(
            {"left_hand": {"detected": True, "rotation": 179}}
        )["rotation"][2]
        clock.advance(0.05)
        second = system.process_input(
            {"left_hand": {"detected": True, "rotation": -179}}
        )["rotation"][2]
        self.assertAlmostEqual(first, 179)
        self.assertGreater(second, first)
        self.assertLess(second - first, 2)

    def test_pinch_can_be_mapped_to_zoom_and_disabled(self):
        config = neutral_config()
        config["controls"]["mappings"] = [
            {
                "id": "pinch",
                "input": "right_hand_pinch",
                "output": "scale_uniform",
                "mode": "absolute",
                "enabled": True,
                "scale": 2,
                "invert": True,
                "center": 0.5,
            }
        ]
        system = create_control_system(config)
        data = {"right_hand": {"detected": True, "pinch": 0.0}}
        self.assertEqual(system.process_input(data)["scale"], 2)
        system.enable_mapping("pinch", False)
        self.assertEqual(system.process_input(data)["scale"], 1)

    def test_hand_pitch_and_yaw_are_independent_circular_inputs(self):
        clock = Clock()
        config = neutral_config()
        config["controls"]["mappings"] = [
            {
                "id": "pitch",
                "input": "left_hand_pitch",
                "output": "rotation_pitch",
                "mode": "absolute",
                "enabled": True,
                "scale": 180,
                "invert": False,
                "center": 0.5,
            },
            {
                "id": "yaw",
                "input": "right_hand_yaw",
                "output": "rotation_yaw",
                "mode": "absolute",
                "enabled": True,
                "scale": 180,
                "invert": True,
                "center": 0.5,
            },
        ]
        system = create_control_system(config, clock)
        first = system.process_input(
            {
                "left_hand": {"detected": True, "pitch": 179},
                "right_hand": {"detected": True, "yaw": 45},
            }
        )
        self.assertAlmostEqual(first["rotation"][1], 179)
        self.assertAlmostEqual(first["rotation"][0], -45)
        clock.advance(0.05)
        second = system.process_input(
            {
                "left_hand": {"detected": True, "pitch": -179},
                "right_hand": {"detected": True, "yaw": 45},
            }
        )
        self.assertGreater(second["rotation"][1], 179)
        self.assertLess(second["rotation"][1], 181)
        self.assertAlmostEqual(second["rotation"][0], -45)

    def test_head_angles_control_each_rotation_independently(self):
        config = neutral_config()
        config["controls"]["mappings"] = [
            {
                "id": angle,
                "input": f"head_{angle}",
                "output": f"rotation_{angle}",
                "mode": "absolute",
                "enabled": True,
                "scale": 180,
                "invert": False,
                "center": 0.5,
            }
            for angle in ("yaw", "pitch", "roll")
        ]
        result = create_control_system(config).process_input(
            {
                "head": {
                    "detected": True,
                    "x": 0.1,
                    "y": 0.9,
                    "pitch": -20,
                    "yaw": 30,
                    "roll": 45,
                }
            }
        )
        for actual, expected in zip(result["rotation"], (30, -20, 45)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["position"], [0, 0, 0])

    def test_head_yaw_wrap_and_tracking_loss_keep_shortest_rotation(self):
        clock = Clock()
        config = neutral_config()
        config["controls"]["mappings"] = [
            {
                "id": "head_turn",
                "input": "head_yaw",
                "output": "rotation_yaw",
                "mode": "absolute",
                "enabled": True,
                "scale": 180,
                "invert": False,
                "center": 0.5,
            }
        ]
        system = create_control_system(config, clock)
        first = system.process_input({"head": {"detected": True, "yaw": 179}})[
            "rotation"
        ][0]
        clock.advance(0.05)
        second = system.process_input({"head": {"detected": True, "yaw": -179}})[
            "rotation"
        ][0]
        self.assertAlmostEqual(first, 179)
        self.assertGreater(second, first)
        self.assertLess(second - first, 2)
        # An undetected stale value must not continue commanding a turn.
        clock.advance(5)
        neutral = system.process_input({"head": {"detected": False, "yaw": -179}})[
            "rotation"
        ][0]
        self.assertAlmostEqual(neutral % 360, 0, places=8)

    def test_individual_hand_positions_and_openness_are_independent(self):
        config = neutral_config()
        config["controls"]["mappings"] = [
            {
                "id": input_name,
                "input": input_name,
                "output": output_name,
                "mode": "absolute",
                "enabled": True,
                "scale": 2,
                "invert": False,
                "center": 0.5,
            }
            for input_name, output_name in (
                ("left_hand_x", "position_x"),
                ("right_hand_y", "position_y"),
                ("left_hand_openness", "scale_uniform"),
            )
        ]
        system = create_control_system(config)
        result = system.process_input(
            {
                "left_hand": {"detected": True, "x": 0.25, "y": 0.1, "openness": 0.75},
                "right_hand": {"detected": True, "x": 0.9, "y": 0.8, "openness": 0},
            }
        )
        self.assertAlmostEqual(result["position"][0], -0.5)
        self.assertAlmostEqual(result["position"][1], 0.6)
        self.assertEqual(result["scale"], 1.5)

    def test_new_inputs_require_detected_finite_measurements(self):
        extractors = create_extractors()
        inputs = [
            (f"head_{angle}", "head", angle, 90, 0.75)
            for angle in ("pitch", "yaw", "roll")
        ]
        inputs += [
            (f"torso_{angle}", "torso", angle, 45, 0.625)
            for angle in ("pitch", "yaw", "roll")
        ]
        inputs += [
            (f"torso_{field}", "torso", field, 0.75, 0.75)
            for field in ("x", "y", "scale")
        ]
        inputs += [
            (f"{side}_hand_{field}", f"{side}_hand", field, 0, 0)
            for side in ("left", "right")
            for field in ("x", "y", "scale", "openness")
        ]
        for name, part, field, value, expected in inputs:
            with self.subTest(input=name):
                extractor = extractors[name]
                self.assertEqual(
                    extractor({part: {"detected": True, field: value}}), expected
                )
                self.assertIsNone(extractor({}))
                self.assertIsNone(extractor({part: None}))
                self.assertIsNone(extractor({part: {"detected": False, field: value}}))
                self.assertIsNone(extractor({part: {"detected": True}}))
                for invalid in (None, True, "0.5", float("nan"), float("inf")):
                    self.assertIsNone(
                        extractor({part: {"detected": True, field: invalid}})
                    )

    def test_lost_head_angle_stops_hybrid_rotation_before_reset(self):
        clock = Clock()
        config = neutral_config()
        config["controls"]["mappings"] = [
            {
                "id": "head_turn",
                "input": "head_yaw",
                "output": "rotation_yaw",
                "mode": "hybrid",
                "enabled": True,
                "scale": 30,
                "invert": False,
                "center": 0.5,
                "left_threshold": 0.4,
                "right_threshold": 0.6,
                "continuous_speed": 100,
            }
        ]
        system = create_control_system(config, clock)
        frame = {"head": {"detected": True, "yaw": 90}}
        system.process_input(frame)
        for _ in range(10):
            clock.advance(0.1)
            before = system.process_input(frame)["rotation"][0]
        self.assertGreater(before, 6)
        clock.advance(0.1)
        self.assertAlmostEqual(
            system.process_input({"head": {"detected": False, "yaw": 90}})["rotation"][
                0
            ],
            before,
        )
        clock.advance(3)
        system.process_input({})
        clock.advance(1)
        self.assertAlmostEqual(system.process_input({})["rotation"][0], 0)

    def test_partial_or_invalid_detection_cannot_produce_nan(self):
        data = DataProcessor().process_hands(
            {"detected": True, "x": 0.5}, {"detected": True}
        )
        self.assertIsNone(data["distance"])
        system = create_control_system()
        result = system.process_input(head(x=float("nan"), y=float("inf")))
        self.assertTrue(all(math.isfinite(value) for value in result["rotation"]))

    def test_angular_loss_returns_to_real_neutral_for_each_sensitivity(self):
        for scale in (30, 70, 90, 180):
            for invert in (False, True):
                with self.subTest(scale=scale, invert=invert):
                    clock = Clock()
                    config = neutral_config()
                    config["controls"]["mappings"] = [
                        {
                            "id": "palm",
                            "input": "left_hand_rotation",
                            "output": "rotation_roll",
                            "mode": "absolute",
                            "enabled": True,
                            "scale": scale,
                            "invert": invert,
                            "center": 0.5,
                        }
                    ]
                    system = create_control_system(config, clock)
                    system.process_input(
                        {"left_hand": {"detected": True, "rotation": 179}}
                    )
                    for _ in range(10):
                        clock.advance(0.05)
                        previous = system.process_input(
                            {"left_hand": {"detected": True, "rotation": -179}}
                        )["rotation"][2]
                    target = previous + (0 - previous + 180) % 360 - 180
                    clock.advance(5)
                    neutral = system.process_input({})["rotation"][2]
                    self.assertAlmostEqual(neutral, target)
                    clock.advance(0.1)
                    system.process_input({})
                    clock.advance(0.1)
                    acquired = system.process_input(
                        {"left_hand": {"detected": True, "rotation": 0}}
                    )["rotation"][2]
                    self.assertAlmostEqual(acquired, neutral)

    def test_angular_reacquisition_preserves_completed_turns_with_nondivisor_sensitivity(
        self,
    ):
        clock = Clock()
        config = neutral_config()
        config["controls"]["smoothing_ms"] = 0
        config["controls"]["mappings"] = [
            {
                "id": "palm",
                "input": "right_hand_roll",
                "output": "rotation_roll",
                "mode": "absolute",
                "enabled": True,
                "scale": 70,
                "invert": False,
                "center": 0.5,
            }
        ]
        system = create_control_system(config, clock)
        for angle in [0, 90, 179, -90] * 8 + [0]:
            clock.advance(0.05)
            previous = system.process_input(
                {"right_hand": {"detected": True, "roll": angle}}
            )["rotation"][2]
        self.assertGreater(previous, 1000)
        target = previous + (0 - previous + 180) % 360 - 180
        clock.advance(5)
        system.process_input({})
        clock.advance(0.1)
        neutral = system.process_input({})["rotation"][2]
        self.assertAlmostEqual(neutral, target)
        clock.advance(0.1)
        acquired = system.process_input({"right_hand": {"detected": True, "roll": 0}})[
            "rotation"
        ][2]
        self.assertAlmostEqual(acquired, neutral)
        clock.advance(0.1)
        moved = system.process_input({"right_hand": {"detected": True, "roll": 10}})[
            "rotation"
        ][2]
        self.assertAlmostEqual(moved, neutral + 10 * 70 / 180)

    def test_angular_position_and_zoom_return_to_numeric_center_after_wrap_and_loss(
        self,
    ):
        for input_name, limit in (
            ("head_roll", 90),
            ("head_pitch", 180),
            ("head_yaw", 180),
            ("torso_roll", 90),
            ("torso_pitch", 180),
            ("torso_yaw", 180),
            ("left_hand_rotation", 180),
            ("right_hand_roll", 180),
        ):
            part, field = input_name.rsplit("_", 1)
            for output_name in (
                "position_x",
                "position_y",
                "position_z",
                "scale_uniform",
            ):
                for invert in (False, True):
                    with self.subTest(
                        input=input_name, output=output_name, invert=invert
                    ):
                        clock = Clock()
                        config = neutral_config()
                        config["controls"]["smoothing_ms"] = 0
                        config["controls"]["mappings"] = [
                            {
                                "id": "scalar",
                                "input": input_name,
                                "output": output_name,
                                "mode": "absolute",
                                "enabled": True,
                                "scale": 2,
                                "invert": invert,
                                "center": 0.5,
                            }
                        ]
                        system = create_control_system(config, clock)

                        def value(data):
                            result = system.process_input(data)
                            if output_name == "scale_uniform":
                                return result["scale"]
                            return result["position"]["xyz".index(output_name[-1])]

                        value({part: {"detected": True, field: limit - 1}})
                        clock.advance(0.05)
                        value({part: {"detected": True, field: 1 - limit}})
                        clock.advance(5)
                        value({})
                        clock.advance(0.1)
                        neutral = 1 if output_name == "scale_uniform" else 0
                        self.assertAlmostEqual(value({}), neutral)
                        clock.advance(0.1)
                        self.assertAlmostEqual(
                            value({part: {"detected": True, field: 0}}), neutral
                        )
                        clock.advance(0.1)
                        moved = value({part: {"detected": True, field: 10}})
                        self.assertAlmostEqual(
                            moved, neutral + 10 / 180 * (-1 if invert else 1)
                        )

    def test_missing_projected_rotation_preserves_other_valid_hand_channels(self):
        clock = Clock()
        config = neutral_config()
        config["controls"]["smoothing_ms"] = 0
        config["controls"]["mappings"] = [
            {
                "id": input_name,
                "input": input_name,
                "output": output_name,
                "mode": "absolute",
                "enabled": True,
                "scale": scale,
                "invert": False,
                "center": 0.5,
            }
            for input_name, output_name, scale in (
                ("left_hand_rotation", "rotation_roll", 180),
                ("left_hand_roll", "rotation_yaw", 180),
                ("left_hand_pitch", "rotation_pitch", 180),
                ("hands_center_x", "position_x", 2),
                ("left_hand_yaw", "position_y", 360),
                ("left_hand_pinch", "scale_uniform", 2),
            )
        ]
        system = create_control_system(config, clock)
        first = system.process_input(
            {
                "left_hand": {
                    "detected": True,
                    "rotation": 45,
                    "roll": 30,
                    "pitch": 10,
                    "yaw": 20,
                    "x": 0.2,
                    "pinch": 0.5,
                }
            }
        )
        for actual, expected in zip(first["rotation"], (30, 10, 45)):
            self.assertAlmostEqual(actual, expected)
        clock.advance(1 / 60)
        result = system.process_input(
            {
                "left_hand": {
                    "detected": True,
                    "rotation": None,
                    "roll": 60,
                    "pitch": 80,
                    "yaw": -35,
                    "x": 0.8,
                    "pinch": 0.8,
                }
            }
        )
        self.assertGreater(result["rotation"][2], 0)
        self.assertLess(result["rotation"][2], 45)
        self.assertAlmostEqual(result["rotation"][0], 60)
        self.assertAlmostEqual(result["rotation"][1], 80)
        self.assertAlmostEqual(result["position"][0], 0.6)
        self.assertAlmostEqual(result["position"][1], -35)
        self.assertAlmostEqual(result["scale"], 1.6)

    def test_body_rotation_and_hand_depth_mapping_preserve_other_channels_on_depth_loss(
        self,
    ):
        clock = Clock()
        config = neutral_config()
        config["controls"]["smoothing_ms"] = 0
        config["controls"]["mappings"] = [
            {
                "id": name,
                "input": name,
                "output": output,
                "mode": "absolute",
                "enabled": True,
                "scale": scale,
                "center": 0.5,
                "invert": False,
            }
            for name, output, scale in (
                ("torso_roll", "rotation_roll", 70),
                ("torso_scale", "position_z", 2),
                ("left_hand_scale", "position_x", 2),
                ("right_hand_scale", "scale_uniform", 2),
                ("torso_y", "position_y", 2),
            )
        ]
        system = create_control_system(config, clock)
        data = {
            "torso": {"detected": True, "roll": 89, "scale": 0.75, "y": 0.6},
            "left_hand": {"detected": True, "scale": 0.8},
            "right_hand": {"detected": True, "scale": 0.7},
        }
        first = system.process_input(data)
        self.assertAlmostEqual(first["position"][0], 0.6)
        self.assertAlmostEqual(first["position"][2], 0.5)
        self.assertAlmostEqual(first["scale"], 1.4)
        clock.advance(0.05)
        data["torso"].update(roll=-89, scale=None)
        data["left_hand"]["scale"] = None
        result = system.process_input(data)
        self.assertAlmostEqual(
            result["rotation"][2] - first["rotation"][2], 2 * 70 / 180
        )
        self.assertGreater(result["position"][2], 0)
        self.assertLess(result["position"][2], 0.5)
        self.assertAlmostEqual(result["position"][1], 0.2)
        self.assertAlmostEqual(result["scale"], 1.4)
        clock.advance(5)
        neutral = system.process_input({})
        self.assertAlmostEqual(neutral["rotation"][2], 0, places=7)
        self.assertAlmostEqual(neutral["position"][2], 0, places=7)

    def test_proximity_accepts_existing_hybrid_and_stepped_modes(self):
        clock = Clock()
        config = neutral_config()
        config["controls"]["smoothing_ms"] = 0
        hybrid = dict(config["controls"]["mappings"][0])
        hybrid.update(input="torso_scale", output="rotation_yaw", invert=False)
        stepped = next(
            dict(m) for m in config["controls"]["mappings"] if m["mode"] == "stepped"
        )
        stepped.update(input="left_hand_scale", transition_ms=0, scale=1)
        config["controls"]["mappings"] = [hybrid, stepped]
        system = create_control_system(config, clock)
        far = system.process_input(
            {
                "torso": {"detected": True, "scale": 0.5},
                "left_hand": {"detected": True, "scale": 0},
            }
        )
        clock.advance(0.05)
        near = system.process_input(
            {
                "torso": {"detected": True, "scale": 1},
                "left_hand": {"detected": True, "scale": 1},
            }
        )
        self.assertGreater(near["rotation"][0], far["rotation"][0])
        self.assertEqual(far["scale"], stepped["small_scale"])
        self.assertEqual(near["scale"], stepped["large_scale"])


if __name__ == "__main__":
    unittest.main()
