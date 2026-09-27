import json
from pathlib import Path
import tempfile
import unittest

from runtime_config import (ConfigReloader, ConfigurationError, default_config,
                            load_config, save_config, validate_config)
from control_system import create_control_system


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid_independent_copies(self):
        original = default_config()
        validated = validate_config(original)
        validated["tracking"]["use_hands"] = True
        self.assertFalse(original["tracking"]["use_hands"])
        self.assertFalse(default_config()["tracking"]["use_hands"])
        self.assertIsNone(default_config()["active_model"])

    def test_missing_file_uses_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_config(Path(directory) / "absent.json"), default_config())

    def test_malformed_unsafe_or_unsupported_configuration_is_rejected(self):
        cases = [("active_model", "../secret.obj"), ("active_model", "/tmp/test.obj"),
                 ("active_model", "a/../../x.obj"), ("schema_version", 2), ("schema_version", True)]
        for key, value in cases:
            with self.subTest(value=value):
                config = default_config()
                config[key] = value
                with self.assertRaises(ConfigurationError):
                    validate_config(config)
        for value in (float("nan"), float("inf"), 0, 10000):
            config = default_config()
            config["tracking"]["inference_fps"] = value
            with self.assertRaises(ConfigurationError):
                validate_config(config)

    def test_safe_archive_entrypoints_are_supported(self):
        for model in (None, "a123/model.obj", "a123/sculpture/Capital.gltf", "a123/Capital.glb"):
            config = default_config()
            config["active_model"] = model
            self.assertEqual(validate_config(config)["active_model"], model)

    def test_legacy_bundled_selection_becomes_empty_but_imports_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for selected, expected in (("capitell.obj", None), ("package/capitell.obj", "package/capitell.obj")):
                config = default_config()
                config["active_model"] = selected
                path.write_text(json.dumps(config), encoding="utf-8")
                self.assertEqual(load_config(path)["active_model"], expected)

    def test_mapping_modes_and_relationships_are_validated(self):
        for mutate in (
            lambda mappings: mappings[0].update(left_threshold=0.6),
            lambda mappings: mappings[0].update(output="scale_uniform"),
            lambda mappings: mappings[0].pop("continuous_speed"),
            lambda mappings: mappings[1].update(id=mappings[0]["id"]),
            lambda mappings: mappings[3].update(small_scale=3, large_scale=1),
            lambda mappings: mappings[3].update(threshold=0.99, hysteresis=0.1),
        ):
            config = default_config()
            mutate(config["controls"]["mappings"])
            with self.assertRaises(ConfigurationError):
                validate_config(config)

    def test_atomic_save_and_reload_preserve_last_good_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(default_config(), path)
            reloader = ConfigReloader(path)
            self.assertIsNone(reloader.reload_if_changed())
            changed = default_config()
            changed["render"]["target_fps"] = 30
            save_config(changed, path)
            self.assertEqual(reloader.reload_if_changed()["render"]["target_fps"], 30)
            path.write_text('{"invalid": true}', encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                reloader.reload_if_changed()
            self.assertEqual(reloader.current["render"]["target_fps"], 30)
            save_config(default_config(), path)
            self.assertEqual(reloader.reload_if_changed()["render"]["target_fps"], 60)
            self.assertEqual(list(path.parent.glob(".config.json.*")), [])

    def test_invalid_save_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(default_config(), path)
            with self.assertRaises(ConfigurationError):
                save_config({}, path)
            self.assertEqual(json.loads(path.read_text()), default_config())

    def test_new_gestures_save_reload_and_build_runtime_controls(self):
        inputs = ("head_pitch", "head_yaw", "head_roll", "left_hand_x", "left_hand_y",
                  "right_hand_x", "right_hand_y", "left_hand_openness", "right_hand_openness")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(default_config(), path)
            reloader = ConfigReloader(path)
            for input_name in inputs:
                with self.subTest(input=input_name):
                    config = default_config()
                    config["controls"]["mappings"] = [{"id": "new_input", "input": input_name,
                        "output": "position_x", "mode": "absolute", "enabled": True,
                        "scale": 2, "invert": False, "center": 0.5}]
                    save_config(config, path)
                    reloaded = reloader.reload_if_changed()
                    self.assertEqual(reloaded, config)
                    part, field = input_name.rsplit("_", 1)
                    value = 90 if part == "head" else 0.75
                    runtime = create_control_system(reloaded)
                    output = runtime.process_input({part: {"detected": True, field: value}})
                    self.assertAlmostEqual(output["position"][0], 0.5)

    def test_unmeasured_depth_inputs_are_rejected(self):
        for input_name in ("head_z", "left_hand_z", "right_hand_z"):
            with self.subTest(input=input_name):
                config = default_config()
                config["controls"]["mappings"][0]["input"] = input_name
                with self.assertRaises(ConfigurationError):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()


def test_duplicate_enabled_outputs_match_portal_validation():
    from runtime_config import ConfigurationError, default_config, validate_config
    config = default_config()
    config['controls']['mappings'][1]['output'] = config['controls']['mappings'][0]['output']
    try:
        validate_config(config)
    except ConfigurationError:
        pass
    else:
        raise AssertionError('Duplicate enabled movements must be rejected')
    config['controls']['mappings'][1]['enabled'] = False
    validate_config(config)
