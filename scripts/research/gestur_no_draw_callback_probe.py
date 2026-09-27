#!/usr/bin/env python3
"""Temporary comparison without the Python callback around Panda's GL draw.

Usage: /opt/gestur/.venv/bin/python /tmp/gestur_no_draw_callback_probe.py \
  --source replay --image IMAGE --model CAPITEL --motion continuous \
  --duration 20 --output-dir /tmp/no-draw-callback

Existing benchmark fields named render_fps/frames become SUBMISSION PROXIES,
not actual draw callbacks or hardware presentations. The manifest records this.
No production files or visual-quality settings are changed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", type=Path, default=Path("/opt/gestur"))
    options, remaining = parser.parse_known_args(argv)
    root = options.project_root.expanduser().resolve()
    if not (root / "scripts/benchmark_pi_system.py").is_file():
        parser.error("--project-root must contain scripts/benchmark_pi_system.py")
    sys.path.insert(0, str(root))

    from panda3d.core import ClockObject, ConfigVariableBool, ConfigVariableDouble

    from runtime_state import FrameMetrics
    from scripts import benchmark_pi_system as benchmark
    from visualizer import ControlledObjViewer

    args = benchmark.arguments(remaining)
    if args.motion != "continuous" or args.offscreen or args.duration > 60:
        parser.error("requires onscreen --motion continuous and --duration at most 60")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    original_init = ControlledObjViewer.__init__
    observed = {}

    def initialize(viewer, *positional, **keywords):
        original_init(viewer, *positional, **keywords)
        if viewer._draw_region is not None:
            viewer._draw_region.clear_draw_callback()
        viewer._draw_callback = None
        viewer.render_metrics = FrameMetrics()
        # The normal viewer sets MLimited 60. Keep it and all framebuffer / GL
        # settings intact; only remove the Python callback enclosing upcall().
        clock = ClockObject.get_global_clock()
        observed.update(
            clock_mode="limited"
            if clock.get_mode() == ClockObject.MLimited
            else "other",
            viewer_target_fps=viewer._render_cadence.target_fps,
            sync_video_requested=ConfigVariableBool("sync-video").get_value(),
            sleep_precision_seconds=ConfigVariableDouble("sleep-precision").get_value(),
        )

        def count_submissions(task):
            window = viewer.win
            if window and window.is_active() and window.is_valid():
                viewer.render_metrics.tick(viewer._render_clock())
            return task.cont

        # ShowBase's igLoop is sort 50; the normal cadence runs at sort 49.
        # This records an active valid window after igLoop, without entering GL.
        viewer.taskMgr.add(count_submissions, "probe-submitted-frame", sort=51)

    try:
        with patch.object(ControlledObjViewer, "__init__", initialize):
            return benchmark.run_trial(args)
    finally:
        path = args.output_dir / "manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            manifest["no_draw_callback_probe"] = {
                "observed_settings": observed,
                "python_draw_callback_removed": True,
                "metric_source": "main-thread task sort51 after igLoop; window active and valid",
                "metric_meaning": "submitted frame proxy",
                "applies_to": "all render_fps, frames and frame_ms_* fields in runtime, telemetry, final metrics and summary",
                "hardware_presented_frames_measured": False,
                "draw_callback_frames_measured": False,
                "limitations": [
                    "A main-loop submission is not proof of successful GPU execution or display presentation.",
                    "These FPS values must not be described as measured physical display FPS.",
                    "No GPU load, GPU execution time or optical latency measurement.",
                ],
                "production_files_changed": False,
                "visual_quality_or_vsync_changed": False,
            }
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "measurement_notice": "submitted frame proxy only; no hardware presentation counter",
                        "output_dir": str(args.output_dir),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
