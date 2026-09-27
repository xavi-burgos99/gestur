#!/usr/bin/env python3
"""Temporary clock-only wrapper for scripts/benchmark_pi_system.py.

Example on the Pi (keep the same model, replay and duration across runs):
  /opt/gestur/.venv/bin/python /tmp/gestur_clock_probe.py \
    --clock-case normal --source replay --image /tmp/replay.jpg \
    --model /tmp/model.glb --motion continuous --duration 20 \
    --output-dir /tmp/clock-normal

This changes the clock only within this process. No production files/config,
VSync, framebuffer, antialiasing, geometry or texture settings are changed.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch

CASES = ('normal', 'limited60', 'limited61', 'limited120')


def set_trial_clock(clock, target_fps, case):
    """Called after the viewer's normal apply_settings; preserves all its work."""
    if case == 'normal':
        clock.set_mode(clock.MNormal)
        return None
    frame_rate = {'limited60': float(target_fps),
                  'limited61': float(target_fps) + 1.0,
                  'limited120': 120.0}[case]
    clock.set_mode(clock.MLimited)
    clock.set_frame_rate(frame_rate)
    return frame_rate


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--clock-case', choices=CASES, default='limited60')
    parser.add_argument('--project-root', type=Path, default=Path('/opt/gestur'))
    options, remaining = parser.parse_known_args(argv)
    root = options.project_root.expanduser().resolve()
    if not (root / 'scripts/benchmark_pi_system.py').is_file():
        parser.error('--project-root must contain scripts/benchmark_pi_system.py')
    sys.path.insert(0, str(root))

    from panda3d.core import ClockObject, ConfigVariableBool, ConfigVariableDouble
    from scripts import benchmark_pi_system as benchmark
    from visualizer import ControlledObjViewer

    args = benchmark.arguments(remaining)
    # Keeping a continuous draw workload isolates clock/presentation pacing;
    # normal mode with skipped idle draws can spin the control loop uncapped.
    if args.motion != 'continuous' or args.offscreen:
        parser.error('clock probe requires --motion continuous and an onscreen window')
    if args.duration > 60:
        parser.error('temporary clock probe is limited to 60 seconds per case')
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
    original = ControlledObjViewer.apply_settings
    observed = []

    def apply_settings(viewer, *, target_fps=60, hide_cursor=True):
        original(viewer, target_fps=target_fps, hide_cursor=hide_cursor)
        clock = ClockObject.get_global_clock()
        limit = set_trial_clock(clock, target_fps, options.clock_case)
        observed.append({
            'viewer_target_fps': target_fps,
            'clock_mode': 'normal' if limit is None else 'limited',
            'clock_limit_fps': limit,
            'sync_video_requested': ConfigVariableBool('sync-video').get_value(),
            'sleep_precision_seconds': ConfigVariableDouble('sleep-precision').get_value(),
        })

    code = 1
    try:
        with patch.object(ControlledObjViewer, 'apply_settings', apply_settings):
            code = benchmark.run_trial(args)
        return code
    finally:
        # The benchmark creates its own manifest and checks actual MSAA/size.
        # Add provenance even if the run fails, without touching production state.
        manifest_path = args.output_dir / 'manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest['clock_probe'] = {
                'case': options.clock_case,
                'apply_settings_observed': observed,
                'vsync_or_quality_changed': False,
                'production_files_changed': False,
                'interpretation': 'Compare the same onscreen continuously moving scene; normal is an experiment, not an idle policy.',
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    sys.exit(main())
