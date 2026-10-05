#!/usr/bin/env python3
"""Render a small, static catalog preview without a camera or X session."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("orientation")
    args = parser.parse_args()
    if sys.platform == "linux":
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    from panda3d.core import Filename, loadPrcFileData

    engine = "p3headlessgl" if sys.platform == "linux" else "p3tinydisplay"
    loadPrcFileData(
        "preview",
        f"load-display {engine}\nwin-size 640 360\naudio-library-name null",
    )
    from visualizer import ControlledObjViewer

    viewer = ControlledObjViewer(
        args.model,
        window_type="offscreen",
        window_size=(640, 360),
        fullscreen=False,
        antialias_samples=0,
        model_orientation=json.loads(args.orientation),
    )
    try:
        viewer.author_credit.hide()
        viewer.adjustWindowAspectRatio(16 / 9)
        for _ in range(3):
            viewer.graphicsEngine.render_frame()
        if not viewer.win.save_screenshot(Filename.from_os_specific(args.output)):
            raise RuntimeError("Preview could not be saved")
    finally:
        viewer.destroy()


if __name__ == "__main__":
    main()
