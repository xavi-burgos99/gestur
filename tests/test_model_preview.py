"""Catalog previews must remain static and reflect model orientation."""

import subprocess
import sys
from pathlib import Path


def test_preview_is_small_and_changes_with_orientation(tmp_path):
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "triangle.obj"
    model.write_text("v -2 0 0\nv 1 0 0\nv 0 0 2\nf 1 2 3\nf 3 2 1\n")
    images = []
    for angle in (0, 90):
        output = tmp_path / f"{angle}.png"
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/render_preview.py"),
                str(model),
                str(output),
                f'{{"x":{angle},"y":0,"z":0}}',
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        with Image.open(output) as image:
            assert image.size == (640, 640)
            assert image.convert("RGBA").getpixel((0, 0))[3] == 0
            images.append(image.convert("RGB").tobytes())
    assert images[0] != images[1]
    assert any(images[1])
