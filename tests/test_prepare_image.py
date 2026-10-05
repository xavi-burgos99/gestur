"""Offline image preparation never runs target binaries or touches host systemd."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_image", PROJECT / "scripts/prepare-image.py"
)
prepare_image = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_image)


@pytest.fixture
def image(tmp_path):
    root = tmp_path / "rootfs"
    (root / "etc/systemd/system").mkdir(parents=True)
    (root / "etc/rpi-issue").write_text("Raspberry Pi reference image\n")
    binary = root / "usr/lib/systemd/systemd"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELF\x02\x01" + bytes(12) + bytes([183, 0]))
    return root


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True
        )

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Image test")
    # Real entrypoints/units, minimal other source. All work occurs in a temp repo.
    files = [
        "gestur.sh",
        "requirements.txt",
        "scripts/first-boot.py",
        "scripts/install-portal.sh",
        "scripts/gestur-device.py",
        "config/default.json",
        "portal/package-lock.json",
        *(f"deployment/{name}" for name in prepare_image.UNITS),
    ]
    for name in files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PROJECT / name).read_bytes())
    (repo / "gestur.sh").chmod(0o755)
    # Even accidentally tracked development data is excluded from the payload.
    for name in [
        "models/private.glb",
        "portal/.dev-data/config.json",
        ".env",
        "config/test.local.json",
        "tracking_models/detector.task",
        "node_modules/secret.txt",
    ]:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("DO NOT SHIP")
    (repo / ".gitignore").write_text("local-cache/\n")
    (repo / "local-cache").mkdir()
    (repo / "local-cache/token").write_text("ignored admin token")
    git("add", ".")
    git("commit", "-qm", "Fixture revision")
    return repo


def prepare(image, source, **kwargs):
    return prepare_image.prepare(
        image,
        source,
        kwargs.get("country", "ES"),
        kwargs.get("admin", "expositor"),
        kwargs.get("hostname"),
    )


def test_prepares_only_committed_code_and_enables_timer_offline(image, source):
    revision, changed = prepare(image, source)
    assert changed
    payload = image / "opt/gestur-bootstrap"
    assert (payload / "gestur.sh").read_bytes() == (source / "gestur.sh").read_bytes()
    assert (payload / "gestur.sh").stat().st_mode & 0o777 == 0o755
    assert (payload / ".source-revision").read_text().strip() == revision
    assert not (payload / ".git").exists()
    assert not (payload / "local-cache").exists()
    for file in payload.rglob("*"):
        if file.is_file():
            assert b"DO NOT SHIP" not in file.read_bytes()
    config = json.loads((image / "etc/gestur/first-boot.json").read_text())
    assert config == {
        "revision": revision,
        "wifi_country": "ES",
        "admin_user": "expositor",
    }
    link = image / "etc/systemd/system/timers.target.wants/gestur-first-boot.timer"
    assert (
        link.is_symlink()
        and link.resolve() == image / "etc/systemd/system/gestur-first-boot.timer"
    )
    assert not (
        image / "etc/systemd/system/multi-user.target.wants/gestur-first-boot.service"
    ).exists()
    assert not (image / "var/lib/gestur-first-boot/complete.json").exists()
    assert prepare(image, source) == (revision, False)


@pytest.mark.parametrize(
    "existing",
    [
        "opt/gestur",
        "etc/gestur/device.json",
        "etc/gestur/portal-token",
        "var/lib/gestur/config.json",
        "var/lib/gestur-first-boot/complete.json",
    ],
)
def test_rejects_already_provisioned_images(image, source, existing):
    path = image / existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("keep")
    with pytest.raises(ValueError, match="ya contiene"):
        prepare(image, source)
    assert path.read_text() == "keep"
    assert not (image / "opt/gestur-bootstrap").exists()


def test_different_settings_do_not_overwrite_image(image, source):
    prepare(image, source)
    with pytest.raises(ValueError, match="otra configuración"):
        prepare(image, source, country="FR")
    assert (
        json.loads((image / "etc/gestur/first-boot.json").read_text())["wifi_country"]
        == "ES"
    )


def test_hostname_is_staged_explicitly_and_never_overwrites_another_image_identity(
    image, source
):
    revision, changed = prepare(image, source, hostname="sala-2")
    assert changed
    settings = json.loads((image / "etc/gestur/first-boot.json").read_text())
    assert settings["hostname"] == "sala-2"
    assert (image / "opt/gestur-bootstrap/scripts/gestur-device.py").is_file()
    assert prepare(image, source, hostname="sala-2") == (revision, False)
    with pytest.raises(ValueError, match="otra configuración"):
        prepare(image, source, hostname="sala-3")
    assert json.loads((image / "etc/gestur/first-boot.json").read_text()) == settings


def test_refuses_host_root_and_32bit_image(image, source):
    with pytest.raises(ValueError):
        prepare(Path("/"), source)
    (image / "usr/lib/systemd/systemd").write_bytes(b"\x7fELF\x01\x01" + bytes(14))
    with pytest.raises(ValueError, match="64 bits"):
        prepare(image, source)


def test_refuses_desktop_and_boot_partition(image, source, tmp_path):
    (image / "etc/systemd/system/display-manager.service").symlink_to(
        "/lib/systemd/system/lightdm.service"
    )
    with pytest.raises(ValueError, match="Lite"):
        prepare(image, source)
    with pytest.raises(ValueError, match="rootfs"):
        prepare(tmp_path / "bootfs", source)


@pytest.mark.parametrize(
    "relative", ["opt", "etc/gestur", "etc/systemd/system/timers.target.wants"]
)
def test_never_writes_through_image_symlinks(image, source, tmp_path, relative):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = image / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="enlaces"):
        prepare(image, source)
    assert list(outside.iterdir()) == []


def test_rejects_source_symlinks(image, source):
    (source / "escape").symlink_to("/etc/passwd")
    subprocess.run(["git", "-C", str(source), "add", "escape"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "Symlink"], check=True)
    with pytest.raises(ValueError, match="enlaces"):
        prepare(image, source)


@pytest.mark.parametrize("dirty", ["gestur.sh", "new-file.py"])
def test_requires_committed_source(image, source, dirty):
    (source / dirty).write_text("uncommitted")
    with pytest.raises(ValueError, match="commit"):
        prepare(image, source)
    assert not (image / "opt").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"country": "ES;reboot"},
        {"admin": "gestur"},
        {"admin": "../root"},
        {"admin": "root"},
        *(
            {"hostname": value}
            for value in (
                "",
                "Sala",
                "sala.local",
                "-sala",
                "sala-",
                "a" * 64,
                "sala;id",
                "sala\n",
                42,
            )
        ),
    ],
)
def test_rejects_invalid_parameters(image, source, kwargs):
    with pytest.raises(ValueError):
        prepare(image, source, **kwargs)
    assert not (image / "opt").exists()
