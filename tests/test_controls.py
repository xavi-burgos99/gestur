import math
import unittest

from control_system import (DataProcessor, ExponentialSmoother,
                            HybridRotationController, create_appliers,
                            create_control_system)
from runtime_config import default_config


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
            controller = HybridRotationController(continuous_speed_degrees_per_second=100, clock=clock)
            controller.update(1)
            for _ in range(2 * fps):
                clock.advance(1 / fps)
                value = controller.update(1)
            results.append(value)
        for result in results:
            self.assertAlmostEqual(result, 215, places=8)

    def test_default_capitell_channels_and_initial_acquisition(self):
        clock = Clock()
        system = create_control_system(clock=clock)
        self.assertEqual(system.process_input({})["rotation"], [0, 0, 0])
        result = system.process_input(head(x=0.75, y=0.75))
        self.assertEqual(result["rotation"], [-5, 15, -15])
        self.assertEqual(result["scale"], 1)
        self.assertEqual(result["position"], [0, 0, 0])

    def test_lost_tracking_stops_motion_then_resets(self):
        clock = Clock()
        system = create_control_system(clock=clock)
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
        controller = HybridRotationController(max_degrees=0,
            continuous_speed_degrees_per_second=350, clock=clock)
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
        apply = create_appliers(clock)["scale_stepped"](threshold=0.4,
            small_scale=1, large_scale=1.75, transition_time_ms=750, hysteresis=0.02)
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
        config = default_config()
        config["controls"]["mappings"] = [{"id": "palm", "input": "left_hand_rotation",
            "output": "rotation_roll", "mode": "absolute", "enabled": True,
            "scale": 180, "invert": False, "center": 0.5}]
        system = create_control_system(config, clock)
        first = system.process_input({"left_hand": {"detected": True, "rotation": 179}})["rotation"][2]
        clock.advance(0.05)
        second = system.process_input({"left_hand": {"detected": True, "rotation": -179}})["rotation"][2]
        self.assertAlmostEqual(first, 179)
        self.assertGreater(second, first)
        self.assertLess(second - first, 2)

    def test_pinch_can_be_mapped_to_zoom_and_disabled(self):
        config = default_config()
        config["controls"]["mappings"] = [{"id": "pinch", "input": "right_hand_pinch",
            "output": "scale_uniform", "mode": "absolute", "enabled": True,
            "scale": 2, "invert": True, "center": 0.5}]
        system = create_control_system(config)
        data = {"right_hand": {"detected": True, "pinch": 0.0}}
        self.assertEqual(system.process_input(data)["scale"], 2)
        system.enable_mapping("pinch", False)
        self.assertEqual(system.process_input(data)["scale"], 1)

    def test_hand_pitch_and_yaw_are_independent_circular_inputs(self):
        clock = Clock()
        config = default_config()
        config["controls"]["mappings"] = [
            {"id": "pitch", "input": "left_hand_pitch", "output": "rotation_pitch",
             "mode": "absolute", "enabled": True, "scale": 180, "invert": False, "center": 0.5},
            {"id": "yaw", "input": "right_hand_yaw", "output": "rotation_yaw",
             "mode": "absolute", "enabled": True, "scale": 180, "invert": True, "center": 0.5},
        ]
        system = create_control_system(config, clock)
        first = system.process_input({"left_hand": {"detected": True, "pitch": 179},
                                      "right_hand": {"detected": True, "yaw": 45}})
        self.assertAlmostEqual(first["rotation"][1], 179)
        self.assertAlmostEqual(first["rotation"][0], -45)
        clock.advance(0.05)
        second = system.process_input({"left_hand": {"detected": True, "pitch": -179},
                                       "right_hand": {"detected": True, "yaw": 45}})
        self.assertGreater(second["rotation"][1], 179)
        self.assertLess(second["rotation"][1], 181)
        self.assertAlmostEqual(second["rotation"][0], -45)

    def test_partial_or_invalid_detection_cannot_produce_nan(self):
        data = DataProcessor().process_hands({"detected": True, "x": 0.5}, {"detected": True})
        self.assertIsNone(data["distance"])
        system = create_control_system()
        result = system.process_input(head(x=float("nan"), y=float("inf")))
        self.assertTrue(all(math.isfinite(value) for value in result["rotation"]))

    def test_angular_loss_returns_to_real_neutral_for_each_sensitivity(self):
        for scale in (30, 70, 90, 180):
            for invert in (False, True):
                with self.subTest(scale=scale, invert=invert):
                    clock = Clock()
                    config = default_config()
                    config["controls"]["mappings"] = [{"id": "palm", "input": "left_hand_rotation",
                        "output": "rotation_roll", "mode": "absolute", "enabled": True,
                        "scale": scale, "invert": invert, "center": .5}]
                    system = create_control_system(config, clock)
                    system.process_input({"left_hand": {"detected": True, "rotation": 179}})
                    for _ in range(10):
                        clock.advance(.05)
                        previous = system.process_input({"left_hand": {"detected": True, "rotation": -179}})["rotation"][2]
                    target = previous + (0 - previous + 180) % 360 - 180
                    clock.advance(5)
                    neutral = system.process_input({})["rotation"][2]
                    self.assertAlmostEqual(neutral, target)
                    clock.advance(.1)
                    system.process_input({})
                    clock.advance(.1)
                    acquired = system.process_input({"left_hand": {"detected": True, "rotation": 0}})["rotation"][2]
                    self.assertAlmostEqual(acquired, neutral)

    def test_angular_reacquisition_preserves_completed_turns_with_nondivisor_sensitivity(self):
        clock = Clock()
        config = default_config()
        config["controls"]["smoothing_ms"] = 0
        config["controls"]["mappings"] = [{"id": "palm", "input": "right_hand_roll",
            "output": "rotation_roll", "mode": "absolute", "enabled": True,
            "scale": 70, "invert": False, "center": .5}]
        system = create_control_system(config, clock)
        for angle in [0, 90, 179, -90] * 8 + [0]:
            clock.advance(.05)
            previous = system.process_input({"right_hand": {"detected": True, "roll": angle}})["rotation"][2]
        self.assertGreater(previous, 1000)
        target = previous + (0 - previous + 180) % 360 - 180
        clock.advance(5)
        system.process_input({})
        clock.advance(.1)
        neutral = system.process_input({})["rotation"][2]
        self.assertAlmostEqual(neutral, target)
        clock.advance(.1)
        acquired = system.process_input({"right_hand": {"detected": True, "roll": 0}})["rotation"][2]
        self.assertAlmostEqual(acquired, neutral)
        clock.advance(.1)
        moved = system.process_input({"right_hand": {"detected": True, "roll": 10}})["rotation"][2]
        self.assertAlmostEqual(moved, neutral + 10 * 70 / 180)

    def test_missing_projected_rotation_preserves_other_valid_hand_channels(self):
        clock = Clock()
        config = default_config()
        config["controls"]["smoothing_ms"] = 0
        config["controls"]["mappings"] = [
            {"id": input_name, "input": input_name, "output": output_name,
             "mode": "absolute", "enabled": True, "scale": scale, "invert": False, "center": .5}
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
        first = system.process_input({"left_hand": {"detected": True, "rotation": 45,
            "roll": 30, "pitch": 10, "yaw": 20, "x": .2, "pinch": .5}})
        for actual, expected in zip(first["rotation"], (30, 10, 45)):
            self.assertAlmostEqual(actual, expected)
        clock.advance(1/60)
        result = system.process_input({"left_hand": {"detected": True, "rotation": None,
            "roll": 60, "pitch": 80, "yaw": -35, "x": .8, "pinch": .8}})
        self.assertGreater(result["rotation"][2], 0)
        self.assertLess(result["rotation"][2], 45)
        self.assertAlmostEqual(result["rotation"][0], 60)
        self.assertAlmostEqual(result["rotation"][1], 80)
        self.assertAlmostEqual(result["position"][0], .6)
        self.assertAlmostEqual(result["position"][1], -35)
        self.assertAlmostEqual(result["scale"], 1.6)


if __name__ == "__main__":
    unittest.main()
