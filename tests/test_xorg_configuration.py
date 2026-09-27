"""Regression coverage for Pi 5's separate display and render DRM devices."""

import shlex
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/configure-xorg.sh"


def configure(action, image_root):
    # First-boot provisioning uses a private umask. Xorg configuration must
    # still be readable by the unprivileged kiosk user after installation.
    return subprocess.run(
        ["bash", "-c", 'umask 077; bash "$1" "$2" "$3"', "bash", str(SCRIPT), action, str(image_root)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_install_reinstall_and_uninstall_preserve_other_xorg_configuration(tmp_path):
    directory = tmp_path / "etc/X11/xorg.conf.d"
    directory.mkdir(parents=True)
    vendor = directory / "00-glamor.conf"
    vendor.write_text('Section "ServerFlags"\nEndSection\n')
    original = vendor.read_bytes()
    target = directory / "99-gestur-vc4.conf"

    configure("install", tmp_path)
    first = target.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert first == (ROOT / "deployment/99-gestur-vc4.conf").read_bytes()

    configure("install", tmp_path)
    assert target.read_bytes() == first
    assert vendor.read_bytes() == original
    assert not list(directory.glob(".gestur-vc4.*"))

    configure("uninstall", tmp_path)
    configure("uninstall", tmp_path)
    assert not target.exists()
    assert vendor.read_bytes() == original


@pytest.mark.parametrize("drm_devices", [
    [("card0", "v3d"), ("card1", "vc4")],
    [("card0", "vc4"), ("card1", "v3d")],
    [("card0", "simpledrm"), ("card1", "v3d"), ("card2", "vc4")],
])
def test_display_selection_matches_drm_driver_independently_of_card_order(tmp_path, drm_devices):
    configure("install", tmp_path)
    text = (tmp_path / "etc/X11/xorg.conf.d/99-gestur-vc4.conf").read_text()
    directives = [shlex.split(line, comments=True) for line in text.splitlines()]
    directives = [line for line in directives if line]
    assert directives[0] == ["Section", "OutputClass"]
    assert directives[-1] == ["EndSection"]
    options = {line[0]: line[1:] for line in directives[1:-1]}
    assert options["Driver"] == ["modesetting"]
    assert options["Option"] == ["PrimaryGPU", "true"]
    assert set(options) == {"Identifier", "MatchDriver", "Driver", "Option"}

    # Xorg obtains this name from drmGetVersion, not the sysfs platform driver
    # (vc4-drm). Its OutputClass MatchDriver compares this string exactly.
    selected = [card for card, driver in drm_devices if driver == options["MatchDriver"][0]]
    assert selected == [card for card, driver in drm_devices if driver == "vc4"]
    assert len(selected) == 1


def test_unknown_action_does_not_create_configuration(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        configure("unexpected", tmp_path)
    assert not (tmp_path / "etc").exists()
