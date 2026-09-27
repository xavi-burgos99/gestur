"""Deployment contracts that prevent a usable HTTP server with broken imports."""

import configparser
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("node", ["/usr/bin/node", "/opt/gestur-node/bin/node"])
def test_installer_uses_selected_node_for_server_and_preflight(tmp_path, node):
    """Run the installer's actual template command, without running installation."""
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "gestur-portal.service").write_bytes(
        (ROOT / "deployment/gestur-portal.service").read_bytes()
    )
    commands = [
        line
        for line in (ROOT / "scripts/install-portal.sh").read_text().splitlines()
        if line.startswith("sed ") and "gestur-portal.service" in line
    ]
    assert len(commands) == 1
    command, separator, destination = commands[0].partition(" > ")
    assert separator and destination == "/etc/systemd/system/gestur-portal.service"
    # Capture the generated unit; never write to the host's /etc.
    rendered = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "NODE_BIN": node, "INSTALL_ROOT": str(tmp_path)},
    ).stdout
    unit = configparser.ConfigParser(strict=False, interpolation=None)
    unit.read_string(rendered)
    service = unit["Service"]
    assert service["ExecStart"] == f"{node} /opt/gestur/portal/server/index.mjs"
    # No '-' to ignore failure and no '+' to bypass the unit's credentials.
    assert service["ExecStartPre"] == f"{node} /opt/gestur/scripts/check-importer.mjs"


def test_native_preflight_keeps_service_credentials_and_compatible_mounts():
    unit = configparser.ConfigParser(strict=False, interpolation=None)
    unit.read(ROOT / "deployment/gestur-portal.service")
    service = unit["Service"]
    assert service["User"] == "gestur-portal"
    assert service["Group"] == "gestur"
    assert service["AmbientCapabilities"] == "CAP_NET_BIND_SERVICE"
    assert not service.getboolean("PermissionsStartOnly", fallback=False)
    assert not service.getboolean("RootDirectoryStartOnly", fallback=False)
    # The converter must work without relaxing the outer service restrictions.
    assert service.getboolean("ProtectKernelTunables")
    assert service["ProtectSystem"] == "strict"
    assert service.getboolean("ProtectHome")
    assert service.getboolean("PrivateTmp")
    assert service.getboolean("ProtectKernelModules")
    assert service.getboolean("ProtectControlGroups")
    assert service["ReadWritePaths"].split() == [
        "/var/lib/gestur",
        "/etc/gestur",
        "/etc/hosts",
    ]
    assert (
        "Environment=GESTUR_DEVICE_STATE=/etc/gestur/device.json"
        in (ROOT / "deployment/gestur-portal.service").read_text()
    )
    # The helpers still require their explicitly scoped sudo elevation.
    assert not service.getboolean("NoNewPrivileges", fallback=False)
    assert "CapabilityBoundingSet" not in service
