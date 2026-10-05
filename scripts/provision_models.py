#!/usr/bin/env python3
"""Download verified official Lite assets and assemble the MediaPipe hand task.

Run after installing requirements/runtime.txt, never at inference startup. MediaPipe
adds the image normalization metadata missing from the Lite source models.
--check uses only the standard library and performs no network I/O.
"""

import argparse
import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tracking_models" / "manifest.json"


def verified(data, asset):
    return (
        len(data) == asset["bytes"]
        and hashlib.sha256(data).hexdigest() == asset["sha256"]
    )


def atomic_write(path, data):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            handle.close()
            temporary.chmod(0o644)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def with_normalization_metadata(data, source):
    # Official legacy Lite models use RGB [0, 1]. Tasks requires this input
    # normalization declared in metadata; weights/tensor order stay unchanged.
    from mediapipe.tasks.python.metadata.metadata_writers.metadata_writer import (
        MetadataWriter,
    )

    writer = (
        MetadataWriter.create(bytearray(data))
        .add_general_info(source.removesuffix(".tflite"))
        .add_image_input(norm_mean=[0.0], norm_std=[255.0])
    )
    names = (
        ["palm detection", "scores"]
        if source.startswith("palm_")
        else ["landmarks", "presence", "handedness", "world landmarks"]
    )
    for name in names:
        writer.add_feature_output(name=name)
    return bytes(writer.populate()[0])


def provision(destination, check=False):
    manifest = json.loads(MANIFEST.read_text())
    destination = Path(destination)
    if not check:
        destination.mkdir(parents=True, exist_ok=True)
    for name, asset in manifest["assets"].items():
        path = destination / name
        if path.is_file() and verified(path.read_bytes(), asset):
            continue
        if check:
            raise RuntimeError(f"Modelo ausente o checksum incorrecto: {path}")
        print(f"Descargando {name}…", flush=True)
        with urllib.request.urlopen(asset["url"], timeout=60) as response:
            data = response.read(asset["bytes"] + 1)
        if not verified(data, asset):
            raise RuntimeError(
                f"Checksum/tamaño incorrecto para {name}; no se instaló el archivo."
            )
        atomic_write(path, data)
    bundle = manifest["hand_bundle"]
    path = destination / bundle["filename"]
    if check:
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != set(bundle["members"]):
                raise RuntimeError(f"Contenido inesperado en {path}")
            for member, source in bundle["members"].items():
                if not verified(
                    archive.read(member), bundle["processed_assets"][member]
                ):
                    raise RuntimeError(f"Checksum incorrecto en {path}: {member}")
    else:
        import io

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            for member, source in bundle["members"].items():
                info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o644 << 16
                data = with_normalization_metadata(
                    (destination / source).read_bytes(), source
                )
                if not verified(data, bundle["processed_assets"][member]):
                    raise RuntimeError(
                        f"Metadatos generados no reproducibles para {member}; usa requirements/runtime.txt."
                    )
                archive.writestr(info, data)
        atomic_write(path, output.getvalue())
    print(f"Modelos Lite verificados en {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "tracking_models")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    provision(args.model_dir, check=args.check)


if __name__ == "__main__":
    main()
