import json
from pathlib import Path
import tempfile
import unittest

from runtime_config import (ConfigReloader, ConfigurationError, default_config,
                            load_config, save_config, validate_config)


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid_independent_copies(self):
        original = default_config()
        validated = validate_config(original)
        validated["tracking"]["use_hands"] = True
        self.assertFalse(original["tracking"]["use_hands"])
        self.assertFalse(default_config()["tracking"]["use_hands"])

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
        for model in ("capitell.obj", "a123/model.obj", "a123/sculpture/Capital.gltf", "a123/Capital.glb"):
            config = default_config()
            config["active_model"] = model
            self.assertEqual(validate_config(config)["active_model"], model)

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


if __name__ == "__main__":
    unittest.main()
