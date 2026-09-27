#!/usr/bin/env python3
"""Stage a committed Gestur revision into an offline Raspberry Pi OS rootfs.

Does not mount/flash disks, run target binaries, or execute the installer here.
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

EXCLUDED = {
    ".git",
    ".github",
    ".venv",
    ".bootstrap",
    ".python",
    "__pycache__",
    ".pytest_cache",
    ".cache",
    ".dev-data",
    "node_modules",
    "dist",
    "artifacts",
    "data",
    "models",
    "models_compressed",
    "tests",
    "test",
}
UNITS = ("gestur-first-boot.service", "gestur-first-boot.timer")


def git(source, *args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def image_path(root, relative):
    """Writing an offline image must never follow its links into the host OS."""
    target = root
    for part in PurePosixPath(relative).parts:
        if part in ("..", "/"):
            raise ValueError("Ruta de imagen no válida.")
        target /= part
        if target.is_symlink():
            raise ValueError(f"No se escribe a través de enlaces: {target}")
    return target


def validate_root(root):
    if root == Path("/"):
        raise ValueError("Indica la raíz de una imagen offline, nunca / del ordenador.")
    if not root.is_dir() or not (root / "etc/rpi-issue").is_file():
        raise ValueError(
            "No parece una raíz de Raspberry Pi OS; monta la partición rootfs, no bootfs."
        )
    binary = root / "usr/lib/systemd/systemd"
    # Inspect, never execute, the target architecture (ELF64, little-endian AArch64).
    if not binary.resolve().is_relative_to(root):
        raise ValueError("El binario systemd apunta fuera de la imagen.")
    with binary.open("rb") as stream:
        header = stream.read(20)
    if (
        header[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(header[18:20], "little") != 183
    ):
        raise ValueError("La imagen debe ser Raspberry Pi OS Lite de 64 bits (ARM64).")
    # An enabled display manager owns the screen needed by our kiosk.
    if os.path.lexists(root / "etc/systemd/system/display-manager.service"):
        raise ValueError("Usa la imagen Lite, sin gestor de escritorio habilitado.")
    for existing in (
        "opt/gestur",
        "etc/gestur/device.json",
        "etc/gestur/portal-token",
        "etc/gestur/wifi-profile.json",
        "var/lib/gestur/config.json",
        "var/lib/gestur-first-boot/complete.json",
    ):
        if os.path.lexists(root / existing):
            raise ValueError(
                f"La imagen ya contiene una instalación o datos: {existing}. Usa una imagen limpia."
            )


def include(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Ruta no válida en el código: {name}")
    return not (
        set(path.parts) & EXCLUDED
        or path.name.startswith((".env", "capitell."))
        or path.name.endswith((".local.json", ".task", ".tflite", ".pem", ".key"))
    )


def prepare(root, source, country, admin, hostname=None):
    root, source = root.resolve(), source.resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError(
            "El código y la raíz de la imagen deben estar en carpetas separadas."
        )
    if not re.fullmatch(r"[A-Z]{2}", country):
        raise ValueError(
            "Indica el código de país Wi-Fi de dos letras, por ejemplo ES."
        )
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", admin) or admin in (
        "root",
        "gestur",
        "gestur-portal",
    ):
        raise ValueError(
            "Elige el usuario administrador de Imager; root y las cuentas gestur están reservadas."
        )
    if hostname is not None and (
        not isinstance(hostname, str)
        or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", hostname)
    ):
        raise ValueError(
            "El hostname debe tener 1–63 letras minúsculas, números o guiones, sin .local ni guiones en los extremos."
        )
    validate_root(root)
    # Export a reproducible snapshot; ignored files, local models and dev tokens
    # cannot enter the image. Never clone/pull a moving branch during first boot.
    if git(source, "status", "--porcelain"):
        raise ValueError(
            "Guarda los cambios del código en un commit antes de preparar la imagen."
        )
    revision = git(source, "rev-parse", "HEAD").decode().strip()
    settings = {"revision": revision, "wifi_country": country, "admin_user": admin}
    if hostname is not None:
        settings["hostname"] = hostname
    payload = image_path(root, "opt/gestur-bootstrap")
    config = image_path(root, "etc/gestur/first-boot.json")
    unit_dir = image_path(root, "etc/systemd/system")
    wants = image_path(root, "etc/systemd/system/timers.target.wants")
    link = wants / "gestur-first-boot.timer"
    for name in UNITS:
        image_path(root, f"etc/systemd/system/{name}")

    with tarfile.open(
        fileobj=io.BytesIO(git(source, "archive", "--format=tar", "HEAD"))
    ) as archive:
        members = [member for member in archive.getmembers() if include(member.name)]
        if any(not (member.isdir() or member.isfile()) for member in members):
            raise ValueError(
                "El paquete no puede contener enlaces ni archivos especiales."
            )
        names = {member.name for member in members}
        required = {
            "gestur.sh",
            "requirements.txt",
            "scripts/first-boot.py",
            "scripts/install-portal.sh",
            "scripts/gestur-device.py",
            "config/default.json",
            "portal/package-lock.json",
            *(f"deployment/{name}" for name in UNITS),
        }
        if not required <= names:
            raise ValueError(
                "La revisión no incluye todos los archivos del primer arranque."
            )
        unit_bytes = {
            name: archive.extractfile(f"deployment/{name}").read() for name in UNITS
        }
        if payload.exists():
            if (
                config.is_file()
                and json.loads(config.read_text()) == settings
                and (payload / ".source-revision").read_text().strip() == revision
                and all(
                    (unit_dir / name).read_bytes() == data
                    for name, data in unit_bytes.items()
                )
                and link.is_symlink()
                and os.readlink(link) == "../gestur-first-boot.timer"
            ):
                return revision, False
            raise ValueError(
                "La imagen ya está preparada con otra configuración o está incompleta. Usa una imagen limpia."
            )
        if (
            config.exists()
            or any((unit_dir / name).exists() for name in UNITS)
            or os.path.lexists(link)
        ):
            raise ValueError(
                "Hay archivos de primer arranque previos; usa una imagen limpia."
            )

        payload.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=".gestur-bootstrap-", dir=payload.parent)
        )
        try:
            for member in members:
                target = staging / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.extractfile(member) as input_file,
                        target.open("wb") as output,
                    ):
                        shutil.copyfileobj(input_file, output)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
            (staging / ".source-revision").write_text(revision + "\n")
            for directory, _, _ in os.walk(staging):
                Path(directory).chmod(0o755)
            staging.rename(payload)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(settings, indent=2) + "\n")
    config.chmod(0o644)
    unit_dir.mkdir(parents=True, exist_ok=True)
    for name, data in unit_bytes.items():
        (unit_dir / name).write_bytes(data)
        (unit_dir / name).chmod(0o644)
    wants.mkdir(parents=True, exist_ok=True)
    link.symlink_to(
        "../gestur-first-boot.timer"
    )  # Enable offline; do not start host systemd.
    return revision, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Raíz montada de Raspberry Pi OS Lite ARM64, antes de arrancar",
    )
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument(
        "--wifi-country",
        required=True,
        help="País donde se usará la Raspberry, por ejemplo ES",
    )
    parser.add_argument(
        "--admin-user",
        required=True,
        help="Usuario administrador configurado en Raspberry Pi Imager",
    )
    parser.add_argument(
        "--hostname",
        help="Nombre del dispositivo sin .local; por defecto gestur-XXXX según su MAC Wi-Fi",
    )
    args = parser.parse_args()
    try:
        if os.geteuid() != 0:
            raise ValueError(
                "Ejecuta con sudo para que el código de la imagen pertenezca a root."
            )
        os.umask(0o022)
        revision, changed = prepare(
            args.root, args.source, args.wifi_country, args.admin_user, args.hostname
        )
        print(f"Imagen {'preparada' if changed else 'ya preparada'}: {revision}")
        print(
            "Desmonta la tarjeta. El primer arranque requiere el usuario de Imager y Ethernet con Internet."
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"No se pudo preparar la imagen: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
