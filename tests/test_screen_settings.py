import subprocess
from types import SimpleNamespace

import pytest

from runtime_config import ConfigurationError, default_config, validate_config
from screen_settings import DEFAULT_SCREEN, rotate_display


def test_legacy_screen_defaults_and_invalid_screen_never_silently_reset():
    config = default_config()
    del config["screen"]
    assert validate_config(config)["screen"] == DEFAULT_SCREEN
    for screen in (
        None,
        {**DEFAULT_SCREEN, "orientation": 45},
        {**DEFAULT_SCREEN, "content_size": "huge"},
    ):
        config["screen"] = screen
        with pytest.raises(ConfigurationError):
            validate_config(config)


def test_native_rotation_uses_only_active_outputs_and_reports_rotated_size(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--query":
            return SimpleNamespace(
                stdout="HDMI-1 connected primary 1920x1080+0+0 (normal left inverted right)\nHDMI-2 disconnected\n"
            )
        return SimpleNamespace(
            stdout="Screen 0: minimum 320 x 200, current 1080 x 1920, maximum 16384 x 16384"
        )

    assert rotate_display(90, platform="linux", run=run) == (1080, 1920)
    assert calls[1] == ["xrandr", "--output", "HDMI-1", "--rotate", "right"]
    assert len(calls) == 3


def test_rotation_failure_is_reported_without_spawning_other_programs(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")

    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(RuntimeError):
        rotate_display(180, platform="linux", run=run)
