"""First-boot failure recovery without root, networking, installation or reboot."""

import importlib.util
import io
import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "gestur_first_boot",
    Path(__file__).resolve().parents[1] / "scripts" / "first-boot.py",
)
first_boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(first_boot)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    revision = "a" * 40
    (bootstrap / ".source-revision").write_text(revision + "\n")
    config = tmp_path / "first-boot.json"
    config.write_text(
        json.dumps(
            {
                "revision": revision,
                "wifi_country": "ES",
                "admin_user": "operator",
            }
        )
    )
    state = tmp_path / "state"
    marker = state / "complete.json"
    events = []
    commands_seen = []
    fail_once = set()

    def host_check(settings):
        events.append("host")
        if "host" in fail_once:
            fail_once.remove("host")
            raise RuntimeError("Unsupported host")

    def run(args, **kwargs):
        # Only these commands may be requested, and none are executed.
        commands_seen.append(args)
        settings = json.loads(config.read_text())
        hostname_args = (
            ("--hostname", settings["hostname"]) if "hostname" in settings else ()
        )
        commands = {
            ("/usr/bin/raspi-config", "nonint", "do_wifi_country", "ES"): "country",
            ("/usr/bin/dpkg", "--configure", "--pending"): "packages",
            (
                "/bin/bash",
                str(bootstrap / "gestur.sh"),
                "install",
                *hostname_args,
            ): "install",
            (
                "/usr/bin/systemctl",
                "is-active",
                "--quiet",
                "gestur-portal.service",
            ): "service",
            ("/usr/bin/systemctl", "--no-block", "reboot"): "reboot",
        }
        action = commands[tuple(args)]
        events.append(action)
        assert kwargs["check"] is True
        assert kwargs["env"]["DEBIAN_FRONTEND"] == "noninteractive"
        assert kwargs["env"]["GESTUR_UNATTENDED"] == "1"
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["umask"] == 0o022
        if action == "reboot":
            assert marker.exists()
            assert events.index("sync") < events.index("reboot")
        else:
            assert not marker.exists()
        if action in fail_once:
            fail_once.remove(action)
            raise subprocess.CalledProcessError(1, args)
        kwargs["stdout"].write(f"Completed {action}\n")
        return subprocess.CompletedProcess(args, 0)

    def ready():
        events.append("http")
        assert not marker.exists()
        if "http" in fail_once:
            fail_once.remove("http")
            raise RuntimeError("Portal unavailable")

    def sync():
        # Installed files must reach storage before recording completion.
        assert not marker.exists()
        events.append("sync")
        if "sync" in fail_once:
            fail_once.remove("sync")
            raise OSError("Storage synchronization failed")

    original_atomic_json = first_boot.atomic_json

    def atomic_json(path, content):
        assert events[-1] == "sync"
        original_atomic_json(path, content)
        events.append("marker")

    monkeypatch.setattr(first_boot, "atomic_json", atomic_json)
    options = dict(
        bootstrap=bootstrap,
        config=config,
        state=state,
        log_path=tmp_path / "logs" / "first-boot.log",
        run=run,
        host_check=host_check,
        ready=ready,
        sync=sync,
    )
    return SimpleNamespace(
        **options,
        options=options,
        revision=revision,
        marker=marker,
        events=events,
        fail_once=fail_once,
        commands_seen=commands_seen,
    )


def test_success_verifies_portal_and_durable_marker_before_reboot(prepared):
    first_boot.provision(**prepared.options)
    assert prepared.events == [
        "host",
        "packages",
        "country",
        "install",
        "service",
        "http",
        "sync",
        "marker",
        "reboot",
    ]
    result = json.loads(prepared.marker.read_text())
    assert result["revision"] == prepared.revision
    assert result["completed_at"]
    assert not prepared.marker.with_suffix(".tmp").exists()
    assert "Instalación completada" in prepared.log_path.read_text()


def test_second_boot_never_installs_or_reboots_again(prepared):
    first_boot.provision(**prepared.options)
    contents = prepared.marker.read_bytes(), prepared.log_path.read_bytes()
    prepared.events.clear()
    first_boot.provision(**prepared.options)
    assert prepared.events == ["host"]
    assert contents == (prepared.marker.read_bytes(), prepared.log_path.read_bytes())


@pytest.mark.parametrize("hostname", ["a", "sala-2", "a" * 63])
def test_hostname_is_passed_as_a_literal_installer_option(prepared, hostname):
    settings = json.loads(prepared.config.read_text())
    settings["hostname"] = hostname
    prepared.config.write_text(json.dumps(settings))
    first_boot.provision(**prepared.options)
    assert [
        "/bin/bash",
        str(prepared.bootstrap / "gestur.sh"),
        "install",
        "--hostname",
        hostname,
    ] in prepared.commands_seen
    assert "portal-token" not in prepared.log_path.read_text()


@pytest.mark.parametrize(
    "failure", ["country", "packages", "install", "service", "http"]
)
def test_failed_attempt_remains_pending_and_next_attempt_succeeds(prepared, failure):
    prepared.fail_once.add(failure)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        first_boot.provision(**prepared.options)
    assert not prepared.marker.exists()
    assert "reboot" not in prepared.events
    assert "sync" not in prepared.events
    assert "Instalación pendiente" in prepared.log_path.read_text()
    prepared.events.clear()
    first_boot.provision(**prepared.options)
    assert prepared.events == [
        "host",
        "packages",
        "country",
        "install",
        "service",
        "http",
        "sync",
        "marker",
        "reboot",
    ]
    assert prepared.marker.exists()


def test_storage_failure_never_marks_installation_complete_or_reboots(prepared):
    prepared.fail_once.add("sync")
    with pytest.raises(OSError, match="Storage synchronization failed"):
        first_boot.provision(**prepared.options)
    assert not prepared.marker.exists()
    assert "reboot" not in prepared.events
    prepared.events.clear()
    first_boot.provision(**prepared.options)
    assert prepared.marker.exists()
    assert prepared.events[-3:] == ["sync", "marker", "reboot"]


def test_host_rejection_does_not_create_state_or_run_commands(prepared):
    prepared.fail_once.add("host")
    with pytest.raises(RuntimeError, match="Unsupported host"):
        first_boot.provision(**prepared.options)
    assert prepared.events == ["host"]
    assert not prepared.state.exists()
    assert not prepared.log_path.exists()


def test_source_revision_mismatch_does_not_install(prepared):
    (prepared.bootstrap / ".source-revision").write_text("b" * 40)
    with pytest.raises(RuntimeError, match="no coincide"):
        first_boot.provision(**prepared.options)
    assert prepared.events == ["host"]
    assert not prepared.state.exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("revision", "web-portal"),
        ("revision", "a" * 39),
        ("admin_user", "root"),
        ("admin_user", "gestur"),
        ("admin_user", "gestur-portal"),
        ("admin_user", "bad/user"),
        ("admin_user", ""),
        ("wifi_country", ""),
        ("wifi_country", "Spain"),
        *(
            ("hostname", value)
            for value in (
                "",
                "Sala",
                "sala.local",
                "-sala",
                "sala-",
                "a" * 64,
                "sala;id",
                "sala\n",
                None,
                42,
            )
        ),
    ],
)
def test_invalid_image_settings_are_rejected_before_commands(prepared, key, value):
    settings = json.loads(prepared.config.read_text())
    settings[key] = value
    prepared.config.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        first_boot.provision(**prepared.options)
    assert prepared.events == []
    assert not prepared.state.exists()


def test_state_and_log_are_private_even_if_preexisting_modes_were_permissive(prepared):
    prepared.state.mkdir(mode=0o755)
    prepared.state.chmod(0o755)
    prepared.log_path.parent.mkdir()
    prepared.log_path.write_text("Earlier attempt\n")
    prepared.log_path.chmod(0o644)
    first_boot.provision(**prepared.options)
    assert stat.S_IMODE(prepared.state.stat().st_mode) == 0o700
    assert stat.S_IMODE(prepared.marker.stat().st_mode) == 0o600
    assert stat.S_IMODE(prepared.log_path.stat().st_mode) == 0o600
    assert prepared.log_path.read_text().startswith("Earlier attempt\n")


def test_log_symlink_is_rejected_without_modifying_target(prepared, tmp_path):
    target = tmp_path / "unrelated-file"
    target.write_text("Preserve this data")
    prepared.log_path.parent.mkdir()
    prepared.log_path.symlink_to(target)
    with pytest.raises(OSError):
        first_boot.provision(**prepared.options)
    assert target.read_text() == "Preserve this data"
    assert prepared.events == ["host"]
    assert not prepared.marker.exists()


def test_reboot_failure_preserves_completed_installation(prepared, capsys):
    prepared.fail_once.add("reboot")
    first_boot.provision(**prepared.options)
    assert prepared.marker.exists()
    assert "Ejecuta sudo reboot" in prepared.log_path.read_text()
    assert "Ejecuta sudo reboot" in capsys.readouterr().out
    prepared.events.clear()
    first_boot.provision(**prepared.options)
    assert prepared.events == ["host"]


def test_portal_healthcheck_uses_port_80_without_proxy_and_retries_invalid_response(
    monkeypatch,
):
    calls, delays, proxies = [], [], []
    replies = [
        OSError("Starting"),
        "[]",
        '{"authenticated": "false"}',
        '{"authenticated": false}',
    ]

    class Opener:
        def open(self, url, timeout):
            calls.append((url, timeout))
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            response = io.StringIO(reply)
            response.status = 200
            return response

    def build_opener(handler):
        proxies.append(handler.proxies)
        return Opener()

    monkeypatch.setattr(first_boot.urllib.request, "build_opener", build_opener)
    monkeypatch.setattr(first_boot.time, "sleep", delays.append)
    first_boot.portal_ready()
    assert proxies == [{}]
    assert calls == [("http://127.0.0.1/api/session", 2)] * 4
    assert delays == [1, 1, 1]


def test_portal_healthcheck_fails_instead_of_accepting_a_started_service(monkeypatch):
    calls = []

    class Opener:
        def open(self, url, timeout):
            calls.append(url)
            raise OSError("Connection refused")

    monkeypatch.setattr(
        first_boot.urllib.request, "build_opener", lambda *args: Opener()
    )
    monkeypatch.setattr(first_boot.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="puerto 80"):
        first_boot.portal_ready()
    assert len(calls) == 30


@pytest.fixture
def pi_host(monkeypatch):
    """Provide OS/account facts without consulting the machine running pytest."""
    import grp
    import pwd

    host = SimpleNamespace(
        uid=0,
        system="Linux",
        machine="aarch64",
        model="Raspberry Pi 5 Model B Rev 1.0\x00",
        admin=SimpleNamespace(
            pw_name="operator", pw_uid=1000, pw_gid=1000, pw_shell="/bin/bash"
        ),
        sudo=SimpleNamespace(gr_gid=27, gr_mem=["operator"]),
    )
    monkeypatch.setattr(first_boot.os, "geteuid", lambda: host.uid)
    monkeypatch.setattr(first_boot.platform, "system", lambda: host.system)
    monkeypatch.setattr(first_boot.platform, "machine", lambda: host.machine)
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path("/proc/device-tree/model"):
            return host.model
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: host.admin)
    monkeypatch.setattr(grp, "getgrnam", lambda name: host.sudo)
    return host


def test_real_host_validation_accepts_pi5_with_existing_sudo_admin(pi_host):
    first_boot.check_host({"admin_user": "operator"})


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("uid", 1000),
        ("system", "Darwin"),
        ("machine", "armv7l"),
        ("model", "Raspberry Pi 4 Model B Rev 1.5\x00"),
    ],
)
def test_real_host_validation_rejects_wrong_platform(pi_host, attribute, value):
    setattr(pi_host, attribute, value)
    with pytest.raises(RuntimeError):
        first_boot.check_host({"admin_user": "operator"})


@pytest.mark.parametrize("invalid_account", ["system", "no_sudo", "nologin", "false"])
def test_real_host_validation_requires_existing_accessible_admin(
    pi_host, invalid_account
):
    if invalid_account == "system":
        pi_host.admin.pw_uid = 999
    elif invalid_account == "no_sudo":
        pi_host.sudo.gr_mem = []
    else:
        pi_host.admin.pw_shell = "/usr/sbin/" + invalid_account
    with pytest.raises(RuntimeError):
        first_boot.check_host({"admin_user": "operator"})
