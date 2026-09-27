#!/usr/bin/env python3
"""Temporary onscreen capitel render-only trial, without camera or inference.
Uses benchmark_render.measure; overrides its offscreen/synthetic assumptions.
Run in the Pi display session, after the operator stops the normal kiosk.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/opt/gestur"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    if not args.model.is_file():
        parser.error("--model must be an existing local model")
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    if args.output and args.output.exists():
        parser.error("--output must not already exist")
    root = args.project_root.expanduser().resolve()
    if not (root / "scripts/benchmark_render.py").is_file():
        parser.error("--project-root must contain scripts/benchmark_render.py")
    sys.path.insert(0, str(root))

    from panda3d.core import ConfigVariableBool, ConfigVariableString, PandaSystem

    from scripts import benchmark_render as benchmark
    from scripts.benchmark_pi_system import model_inventory
    from visualizer import ControlledObjViewer

    observations = {}

    def onscreen_viewer(asset, **kwargs):
        kwargs.update(
            window_type=None, fullscreen=True, target_fps=60, antialias_samples=2
        )
        viewer = ControlledObjViewer(asset, **kwargs)
        try:
            dimensions = [viewer.win.get_x_size(), viewer.win.get_y_size()]
            samples = viewer.win.get_fb_properties().get_multisamples()
            if dimensions != [1920, 1080] or samples != 4:
                raise RuntimeError(
                    f"Non-comparable framebuffer: {dimensions}, MSAA={samples}; expected 1920x1080, MSAA=4"
                )
            observations.update(
                model=model_inventory(viewer.model, args.model, synthetic=False),
                sync_video_requested=ConfigVariableBool("sync-video").get_value(),
                threading_model=ConfigVariableString("threading-model").get_value(),
            )
            original_update = viewer.update_model

            def move_like_integrated_trial(**transform):
                # measure() supplies task.time * 30 as yaw. Use that same time
                # for the three-axis trajectory in benchmark_pi_system.
                elapsed = transform["rotation"][0] / 30
                return original_update(
                    position=[0, 0, 0],
                    scale=1,
                    rotation=[elapsed * 30, 15 * math.sin(elapsed / 3), elapsed * 12],
                )

            viewer.update_model = move_like_integrated_trial
            original_destroy = viewer.destroy

            def record_and_destroy():
                observations["draw_intervals_including_half_second_warmup"] = (
                    viewer.render_metrics.summary()
                )
                original_destroy()

            viewer.destroy = record_and_destroy
            return viewer
        except BaseException:
            viewer.destroy()
            raise

    with patch.object(benchmark, "ControlledObjViewer", onscreen_viewer):
        result = benchmark.measure(
            "capitel-render-only-onscreen",
            args.model,
            args.seconds,
            continuous=True,
            moving=True,
            precision=0.004,
        )
    # These two original fields describe its synthetic fixture, not our model.
    result.pop("texture", None)
    result["triangles"] = observations["model"]["triangles"]
    result.update(observations)
    result.update(
        panda3d=PandaSystem.get_version_string(),
        scope={
            "camera_or_inference_started": False,
            "production_files_changed": False,
            "onscreen": True,
            "continuous_draw": True,
            "clock_mode": "limited",
            "clock_limit_fps": 60,
            "motion": "Same three-axis formula as benchmark_pi_system; task-time phase includes 0.5 s warmup.",
            "quality": "1920x1080, requested MSAA2/verified MSAA4, original model and textures.",
            "cpu_and_fps_exclude_warmup": True,
            "gpu_busy_percent_or_gpu_time_measured": False,
        },
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
