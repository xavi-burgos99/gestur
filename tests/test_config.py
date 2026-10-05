import json
import tempfile
import unittest
from pathlib import Path

from control_system import create_control_system
from runtime_config import (
    ConfigReloader,
    ConfigurationError,
    default_config,
    load_config,
    save_config,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid_independent_copies(self):
        original = default_config()
        validated = validate_config(original)
        validated["tracking"]["use_hands"] = True
        self.assertFalse(original["tracking"]["use_hands"])
        self.assertFalse(default_config()["tracking"]["use_hands"])
        self.assertIsNone(default_config()["active_model"])
        self.assertEqual(default_config()["render"]["ambient_light"], "none")
        self.assertEqual(default_config()["render"]["exposure"], 50)
        self.assertEqual(default_config()["controls"]["idle_mode"], "float")

    def test_legacy_v1_lighting_and_idle_defaults_preserve_user_parameters(self):
        legacy = default_config()
        del legacy["render"]["ambient_light"]
        del legacy["render"]["exposure"]
        del legacy["controls"]["idle_mode"]
        legacy["render"]["target_fps"] = 30
        legacy["controls"]["smoothing_ms"] = 237
        legacy["controls"]["mappings"][0]["scale"] = 44
        legacy["tracking"]["use_hands"] = True
        snapshot = json.loads(json.dumps(legacy))
        migrated = validate_config(legacy)
        expected = json.loads(json.dumps(legacy))
        expected["render"]["ambient_light"] = "none"
        expected["render"]["exposure"] = 50
        expected["controls"]["idle_mode"] = "return"
        self.assertEqual(migrated, expected)
        self.assertEqual(legacy, snapshot)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            self.assertEqual(load_config(path), expected)
            save_config(load_config(path), path)
            self.assertEqual(json.loads(path.read_text()), expected)
        # Partial migration must retain a deliberately selected preset/mode.
        legacy["render"]["ambient_light"] = "sunset"
        self.assertEqual(validate_config(legacy)["render"]["ambient_light"], "sunset")
        del legacy["render"]["ambient_light"]
        legacy["controls"]["idle_mode"] = "hold"
        self.assertEqual(validate_config(legacy)["controls"]["idle_mode"], "hold")

    def test_lighting_and_idle_modes_validate_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for light in ("none", "studio", "gallery", "sunset", "rim"):
                for mode in ("hold", "return", "float"):
                    with self.subTest(light=light, mode=mode):
                        config = default_config()
                        config["render"]["ambient_light"] = light
                        config["controls"]["idle_mode"] = mode
                        save_config(config, path)
                        self.assertEqual(load_config(path), config)
            valid_bytes = path.read_bytes()
            for section, field in (
                ("render", "ambient_light"),
                ("controls", "idle_mode"),
            ):
                for value in (None, "", "automatic", 1, False, [], {}):
                    with self.subTest(field=field, value=value):
                        config = default_config()
                        config[section][field] = value
                        with self.assertRaises(ConfigurationError):
                            save_config(config, path)
                        self.assertEqual(path.read_bytes(), valid_bytes)

    def test_legacy_lighting_presets_migrate_without_changing_user_exposure(self):
        for old, new in (
            ("soft", "studio"),
            ("warm", "sunset"),
            ("cool", "gallery"),
            ("contrast", "rim"),
        ):
            with self.subTest(old=old):
                legacy = default_config()
                legacy["render"]["ambient_light"] = old
                legacy["render"]["target_fps"] = 30
                legacy["controls"]["idle_mode"] = "hold"
                del legacy["render"]["exposure"]
                expected = json.loads(json.dumps(legacy))
                expected["render"].update(ambient_light=new, exposure=50)
                self.assertEqual(validate_config(legacy), expected)
                self.assertEqual(legacy["render"]["ambient_light"], old)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(legacy), encoding="utf-8")
                    self.assertEqual(load_config(path), expected)
                    legacy["render"]["exposure"] = 73
                    expected["render"]["exposure"] = 73
                    self.assertEqual(save_config(legacy, path), expected)
                    self.assertEqual(load_config(path), expected)

    def test_exposure_bounds_and_integer_validation_preserve_saved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for exposure in (0, 5, 10, 37, 50, 73, 100):
                config = default_config()
                config["render"].update(exposure=exposure, ambient_light="studio")
                self.assertEqual(save_config(config, path), config)
                self.assertEqual(load_config(path), config)
            before = path.read_bytes()
            for exposure in (None, -1, 101, 49.5, "50", True, False, [], {}):
                with self.subTest(exposure=exposure):
                    config = default_config()
                    config["render"]["exposure"] = exposure
                    with self.assertRaises(ConfigurationError):
                        save_config(config, path)
                    self.assertEqual(path.read_bytes(), before)

    def test_missing_file_uses_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                load_config(Path(directory) / "absent.json"), default_config()
            )

    def test_malformed_unsafe_or_unsupported_configuration_is_rejected(self):
        cases = [
            ("active_model", "../secret.obj"),
            ("active_model", "/tmp/test.obj"),
            ("active_model", "a/../../x.obj"),
            ("schema_version", 2),
            ("schema_version", True),
        ]
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
        for model in (
            None,
            "a123/model.obj",
            "a123/sculpture/Capital.gltf",
            "a123/Capital.glb",
        ):
            config = default_config()
            config["active_model"] = model
            self.assertEqual(validate_config(config)["active_model"], model)

    def test_legacy_bundled_selection_becomes_empty_but_imports_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for selected, expected in (
                ("capitell.obj", None),
                ("package/capitell.obj", "package/capitell.obj"),
            ):
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

    def test_true_hand_roll_can_be_saved_and_reloaded_independently_of_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(default_config(), path)
            reloader = ConfigReloader(path)
            config = default_config()
            config["controls"]["mappings"] = [
                {
                    "id": side,
                    "input": f"{side}_hand_roll",
                    "output": output,
                    "mode": "absolute",
                    "enabled": True,
                    "scale": 180,
                    "center": 0.5,
                    "invert": False,
                }
                for side, output in (
                    ("left", "rotation_roll"),
                    ("right", "rotation_yaw"),
                )
            ]
            save_config(config, path)
            self.assertEqual(reloader.reload_if_changed(), config)
            self.assertEqual(load_config(path), config)

    def test_new_gestures_save_reload_and_build_runtime_controls(self):
        inputs = (
            "head_pitch",
            "head_yaw",
            "head_roll",
            "left_hand_x",
            "left_hand_y",
            "right_hand_x",
            "right_hand_y",
            "left_hand_openness",
            "right_hand_openness",
            "torso_x",
            "torso_y",
            "torso_scale",
            "torso_pitch",
            "torso_yaw",
            "torso_roll",
            "left_hand_scale",
            "right_hand_scale",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(default_config(), path)
            reloader = ConfigReloader(path)
            for input_name in inputs:
                with self.subTest(input=input_name):
                    config = default_config()
                    config["controls"]["mappings"] = [
                        {
                            "id": "new_input",
                            "input": input_name,
                            "output": "position_x",
                            "mode": "absolute",
                            "enabled": True,
                            "scale": 2,
                            "invert": False,
                            "center": 0.5,
                        }
                    ]
                    save_config(config, path)
                    reloaded = reloader.reload_if_changed()
                    self.assertEqual(reloaded, config)
                    part, field = input_name.rsplit("_", 1)
                    value = 90 if field in ("pitch", "yaw", "roll") else 0.75
                    runtime = create_control_system(reloaded)
                    output = runtime.process_input(
                        {part: {"detected": True, field: value}}
                    )
                    self.assertAlmostEqual(output["position"][0], 0.5)

    def test_unmeasured_depth_inputs_are_rejected(self):
        for input_name in ("head_z", "torso_z", "left_hand_z", "right_hand_z"):
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
    config["controls"]["mappings"][1]["output"] = config["controls"]["mappings"][0][
        "output"
    ]
    try:
        validate_config(config)
    except ConfigurationError:
        pass
    else:
        raise AssertionError("Duplicate enabled movements must be rejected")
    config["controls"]["mappings"][1]["enabled"] = False
    validate_config(config)
