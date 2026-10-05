"""Persistent display settings and native Xorg rotation for the kiosk session."""

import os
import re
import subprocess
import sys

SIZE_FACTORS = {
    "very_small": 0.7,
    "small": 0.85,
    "default": 1.0,
    "large": 1.15,
    "very_large": 1.3,
}
DEFAULT_SCREEN = {"orientation": 0, "content_size": "default", "model_size": "default"}


def rotate_display(orientation, *, run=subprocess.run, platform=None):
    """Use the viewer's own X session; rotation affects the whole output once."""
    if (platform or sys.platform) != "linux" or not os.environ.get("DISPLAY"):
        return None

    def invoke(command):
        try:
            return run(command, capture_output=True, text=True, check=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                "La pantalla no permite aplicar la orientación solicitada"
            ) from error

    rotation = {0: "normal", 90: "right", 180: "inverted", 270: "left"}[orientation]
    result = invoke(["xrandr", "--query"])
    active = re.findall(
        r"^(\S+) connected (?:primary )?\d+x\d+[-+]\d+[-+]\d+", result.stdout, re.M
    )
    if not active:
        raise RuntimeError("No hay una pantalla activa para aplicar la orientación")
    for output in active:
        invoke(["xrandr", "--output", output, "--rotate", rotation])
    result = invoke(["xrandr", "--current"])
    size = re.search(r"current (\d+) x (\d+)", result.stdout)
    return tuple(map(int, size.groups())) if size else None
