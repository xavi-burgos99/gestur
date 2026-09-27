#!/usr/bin/env python3
"""Bounded integrated render/inference trial; no production configuration writes.

Run on the Pi's display session, with the normal kiosk stopped by its operator.
Replay runs the real installed models on local images, not a mocked detector.
This measures computational load and temperature, not recognition accuracy or
camera-to-display optical latency. No packages, models or images are downloaded.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMPERATURE_LIMIT = 85.0
ERROR_TIMEOUT = 15.0
STARTUP_GRACE = 30.0


def read_json(path):
    try:
        value = path.read_text(encoding='utf-8')
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except (OSError, ValueError):
        return None


def rss_metrics():
    current = None
    try:
        pages = int(Path('/proc/self/statm').read_text().split()[1])
        current = pages * os.sysconf('SC_PAGE_SIZE') / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak /= 1024 ** 2 if sys.platform == 'darwin' else 1024
    return {'process_rss_mib': round(current, 3) if current is not None else None,
            'process_peak_rss_mib': round(peak, 3)}


def parse_throttled(value):
    # Official bit definitions: raspberrypi.com/documentation/computers/os.html#get_throttled
    try:
        mask = int(value.strip().removeprefix('throttled='), 16)
    except (ValueError, AttributeError):
        return None
    names = ('undervoltage', 'frequency_capped', 'throttled', 'soft_temperature_limit')
    return {'raw': hex(mask),
            'now': {name: bool(mask & (1 << bit)) for bit, name in enumerate(names)},
            'since_boot': {name: bool(mask & (1 << (bit + 16))) for bit, name in enumerate(names)}}


class ReplayCapture:
    """OpenCV-compatible source paced by interruptible waits, with no catch-up burst."""
    def __init__(self, frames, stop_event, clock=time.monotonic):
        if not frames:
            raise ValueError('Replay needs at least one image.')
        self.frames, self.stop_event, self.clock = frames, stop_event, clock
        self.index, self.next_frame, self.closed = 0, clock(), False

    def read(self):
        if self.closed or self.stop_event.wait(max(0, self.next_frame - self.clock())):
            return False, None
        frame = self.frames[self.index % len(self.frames)].copy()
        self.index += 1
        self.next_frame = self.clock() + 1 / 24
        return True, frame

    def release(self):
        self.closed = True

    def isOpened(self):
        return not self.closed and not self.stop_event.is_set()


class TrialGuard:
    """Only the render task acts on a stop request; this class touches no UI."""
    def __init__(self):
        self.error_since = None
        self.progress = {}

    def check(self, elapsed, hardware, runtime, runtime_age, sampler_error=None):
        temperature = hardware.get('cpu_temperature_c')
        if temperature is not None and temperature > TEMPERATURE_LIMIT:
            return f'temperature_exceeded_{TEMPERATURE_LIMIT:g}C'
        tracking = runtime.get('tracking', {}) if runtime else {}
        problem = sampler_error or (runtime.get('error') if runtime else None) or tracking.get('error')
        if elapsed >= STARTUP_GRACE:
            if runtime is None or runtime_age is None or runtime_age > ERROR_TIMEOUT:
                problem = problem or 'runtime_status_missing_or_stale'
            if tracking.get('state') != 'running':
                problem = problem or 'tracking_not_running'
        metrics = tracking.get('metrics', {})
        for key in ('captured_frames', 'pose_frames', 'hand_frames'):
            count = metrics.get(key, 0)
            previous, changed = self.progress.get(key, (0, 0.0))
            if count != previous:
                changed = elapsed
            self.progress[key] = count, changed
            if elapsed >= STARTUP_GRACE and elapsed - changed >= ERROR_TIMEOUT:
                problem = problem or f'{key}_not_progressing'
        if problem:
            self.error_since = elapsed if self.error_since is None else self.error_since
            if elapsed - self.error_since >= ERROR_TIMEOUT:
                return f'persistent_error: {problem}'
        else:
            self.error_since = None
        return None


class TelemetrySampler(threading.Thread):
    def __init__(self, output, status_path, started):
        super().__init__(name='gestur-trial-telemetry', daemon=True)
        from device_metrics import DeviceMetrics
        self.device = DeviceMetrics()
        self.output, self.status_path, self.started = output, status_path, started
        self.finished, self.abort = threading.Event(), threading.Event()
        self.reason = None
        self.guard = TrialGuard()
        self.vcgencmd = shutil.which('vcgencmd')

    def stop_with_reason(self, reason):
        if not self.abort.is_set():
            self.reason = reason
            self.abort.set()

    def throttled(self):
        if not self.vcgencmd:
            return None
        try:
            result = subprocess.run([self.vcgencmd, 'get_throttled'], capture_output=True,
                                    text=True, timeout=.75, check=False)
            return parse_throttled(result.stdout) if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def run(self):
        try:
            with self.output.open('x', encoding='utf-8', buffering=1) as stream:
                while not self.finished.is_set():
                    tick = time.monotonic()
                    elapsed = tick - self.started
                    error = None
                    try:
                        hardware = {**self.device.sample(), **rss_metrics(), 'throttling': self.throttled()}
                    except Exception as exc:
                        hardware, error = {}, f'{type(exc).__name__}: {exc}'
                    runtime = read_json(self.status_path)
                    updated = runtime.get('updated_at') if runtime else None
                    age = max(0.0, time.time() - updated) if isinstance(updated, (int, float)) else None
                    reason = self.guard.check(elapsed, hardware, runtime, age, error)
                    sample = {'elapsed_seconds': round(elapsed, 3), 'timestamp': time.time(),
                              'hardware': hardware, 'runtime_status_age_seconds': age,
                              'runtime': runtime, 'sampler_error': error, 'abort_reason': reason}
                    stream.write(json.dumps(sample) + '\n')
                    if reason:
                        self.stop_with_reason(reason)
                        break
                    self.finished.wait(max(0, 1 - (time.monotonic() - tick)))
        except Exception as exc:
            self.stop_with_reason(f'telemetry_failed: {type(exc).__name__}: {exc}')


def trial_config(windowed, camera_index):
    from runtime_config import default_config, validate_config
    config = default_config()
    config['tracking'].update(use_pose=True, use_hands=True, width=640, height=480,
                              inference_fps=24, hand_fps=15, camera_index=camera_index)
    config['render'].update(fullscreen=not windowed, target_fps=60, antialias_samples=2)
    config['controls']['mappings'] = [
        {'id': source, 'input': source, 'output': target, 'mode': 'absolute',
         'enabled': True, 'scale': scale, 'invert': False, 'center': .5}
        for source, target, scale in (
            ('head_yaw', 'rotation_yaw', 30), ('head_pitch', 'rotation_pitch', 30),
            ('left_hand_roll', 'rotation_roll', 70), ('right_hand_x', 'position_x', 1),
            ('left_hand_pinch', 'scale_uniform', 1),
        )
    ]
    return validate_config(config)


def load_replay(paths):
    import cv2
    import numpy as np
    frames, inputs = [], []
    for path in paths:
        source = cv2.imread(str(path))
        if source is None:
            raise ValueError(f'Cannot decode replay image: {path}')
        height, width = source.shape[:2]
        ratio = min(640 / width, 480 / height)
        w, h = max(1, round(width * ratio)), max(1, round(height * ratio))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[(480-h)//2:(480-h)//2+h, (640-w)//2:(640-w)//2+w] = cv2.resize(source, (w, h))
        frames.append(frame)
        inputs.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'original_size': [width, height]})
    return frames, inputs


def model_inventory(model, source, *, synthetic):
    """Inspect the scene actually loaded, counting strip/fan triangles correctly."""
    from panda3d.core import GeomTriangles
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    triangles, geoms = 0, 0
    for path in model.find_all_matches('**/+GeomNode'):
        node = path.node()
        for index in range(node.get_num_geoms()):
            geom = node.get_geom(index)
            geoms += 1
            for primitive in geom.get_primitives():
                simple = primitive.decompose()
                if isinstance(simple, GeomTriangles):
                    triangles += simple.get_num_primitives()
    textures = []
    for texture in model.find_all_textures():
        textures.append({
            'name': texture.get_name(),
            'path': texture.get_fullpath().to_os_specific() or None,
            'loaded_size': [texture.get_x_size(), texture.get_y_size()],
            'source_size': [texture.get_orig_file_x_size(), texture.get_orig_file_y_size()],
        })
    return {'source': 'synthetic_fixture' if synthetic else 'local_model',
            'path': str(source), 'sha256': digest.hexdigest(),
            'triangles': triangles, 'geom_batches': geoms,
            'textures': sorted(textures, key=lambda item: (item['name'], item['path'] or ''))}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=60, help='Measured trial seconds, 1..1800 (default 60).')
    parser.add_argument('--source', choices=('camera', 'replay'), default='camera')
    parser.add_argument('--image', action='append', type=Path, default=[], help='Local replay image; repeat for a cycle.')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--model', type=Path, help='Existing local model; resources remain beside it. Default: synthetic fixture.')
    parser.add_argument('--motion', choices=('continuous', 'controls'), default='continuous',
                        help='Continuous imposed rotation, or real gesture controls (default continuous).')
    parser.add_argument('--windowed', action='store_true')
    parser.add_argument('--offscreen', action='store_true', help='Optional graphics smoke test; no display presentation.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing paths are never overwritten.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 1800:
        parser.error('--duration must be finite and between 1 and 1800 seconds')
    if not 0 <= args.camera_index <= 16:
        parser.error('--camera-index must be between 0 and 16')
    if args.source == 'replay' and (not args.image or not all(path.is_file() for path in args.image)):
        parser.error('--source replay requires one or more existing --image files')
    if args.source == 'camera' and args.image:
        parser.error('--image is only supported with --source replay')
    if args.model is not None:
        args.model = args.model.expanduser().resolve()
        if not args.model.is_file():
            parser.error('--model must be an existing local model file')
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() or args.output_dir.is_relative_to(Path('/var/lib/gestur').resolve()):
        parser.error('--output-dir must be new and outside /var/lib/gestur')
    return args


def run_trial(args):
    from controller import PoseController
    from scripts.benchmark_render import make_fixture
    from panda3d.core import loadPrcFileData
    frames, images = load_replay(args.image) if args.source == 'replay' else ([], [])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    config = trial_config(args.windowed or args.offscreen, args.camera_index)
    (output / 'trial-config.json').write_text(json.dumps(config, indent=2) + '\n')
    versions = {}
    for name in ('mediapipe', 'numpy', 'opencv-contrib-python', 'panda3d'):
        versions[name] = importlib.metadata.version(name)
    manifest = {
        'source': args.source, 'duration_requested_seconds': args.duration,
        'platform': platform.platform(), 'python': platform.python_version(), 'versions': versions,
        'images': images, 'capture_requested_size': [640, 480],
        'replay_fps': 24 if images else None, 'motion': args.motion,
        'temperature_abort_above_c': TEMPERATURE_LIMIT, 'persistent_error_seconds': ERROR_TIMEOUT,
        'startup_grace_seconds': STARTUP_GRACE, 'production_state_modified': False,
        'scope': [
            'Real installed PoseController, Pose Lite and Hand Lite inference; production scheduling/duty/idle retained.',
            ('Continuous imposed rotation after controls (task sort 11), overriding their transform.'
             if args.motion == 'continuous' else
             'Only real gesture controls move the model; render/inference idle policies may reduce workload.'),
            'Model triangle count and texture dimensions describe the loaded scene, not an assumed fixture.',
            'No recognition accuracy, physical movement tracking accuracy or optical camera-to-display latency measurement.',
            'Telemetry includes runtime startup/shutdown; fixture generation and window creation occur before trial timing.',
            'Replay repeats letterboxed images at 24 fps; camera mode uses the real camera and driver.',
            'Throttling since_boot bits may predate the trial. Missing sensors are null, never zero.',
            'Final FPS averages span the run; frame-time percentiles retain the last 18000 intervals only.',
        ],
    }
    controller = monitor = None
    started = None
    last_loop_elapsed = 0.0
    code, outcome = 1, 'failed'
    with tempfile.TemporaryDirectory(prefix='gestur-pi-trial-') as temporary, ExitStack() as stack:
        try:
            temp = Path(temporary)
            asset = args.model or temp / 'synthetic.bam'
            if args.model is None:
                make_fixture(asset)
            config_path = temp / 'config.json'
            config_path.write_text(json.dumps(config) + '\n')
            if frames:
                from pose_detector import PoseHandTracker
                stack.enter_context(patch.object(PoseHandTracker, '_open_camera',
                    lambda tracker: ReplayCapture(frames, tracker._stop_event)))
            if args.offscreen:
                loadPrcFileData('gestur-trial-offscreen', 'window-type offscreen')
            controller = PoseController(obj_path=asset, config_path=config_path, models_dir=temp,
                                        benchmark_seconds=args.duration, metrics_path=output / 'final-metrics.json')
            controller.status_path = output / 'runtime-status.json'
            window = controller.visualizer.win
            manifest['framebuffer'] = [window.get_x_size(), window.get_y_size()]
            manifest['msaa_actual'] = window.get_fb_properties().get_multisamples()
            manifest['renderer'] = window.get_gsg().get_driver_renderer()
            manifest['offscreen'] = args.offscreen
            manifest['model'] = model_inventory(controller.visualizer.model, asset, synthetic=args.model is None)
            manifest['triangles'] = manifest['model']['triangles']
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            started = time.monotonic()
            monitor = TelemetrySampler(output / 'telemetry.jsonl', controller.status_path, started)

            def move_and_guard(task):
                nonlocal last_loop_elapsed
                if controller._start is not None:
                    last_loop_elapsed = time.monotonic() - controller._start
                if monitor.abort.is_set():
                    controller.exit_code = 2
                    controller.request_stop()
                    return task.done
                if args.motion == 'continuous':
                    elapsed = time.monotonic() - started
                    controller.visualizer.update_model(position=[0, 0, 0], scale=1,
                        rotation=[elapsed * 30, 15 * math.sin(elapsed / 3), elapsed * 12])
                return task.cont

            controller.visualizer.taskMgr.add(move_and_guard, 'gestur-integrated-trial', sort=11)
            monitor.start()
            code = controller.run()
            if code:
                outcome = 'failed'
            elif last_loop_elapsed >= args.duration - .25:
                outcome = 'completed'
            else:
                code, outcome = 130, 'interrupted'
        finally:
            if monitor:
                monitor.finished.set()
                if monitor.ident is not None:
                    monitor.join(timeout=2)
            if controller:
                controller.cleanup()
            if monitor and monitor.abort.is_set():
                code, outcome = 2, 'aborted'
            final = read_json(output / 'final-metrics.json') or {}
            final_tracking = final.get('tracking', {}).get('metrics', {})
            if code == 0 and (not final_tracking.get('pose_frames') or not final_tracking.get('hand_frames')):
                code, outcome = 3, 'incomplete_inference'
            summary = {'outcome': outcome, 'exit_code': code,
                       'controller_loop_seconds': round(last_loop_elapsed, 3),
                       'duration_including_cleanup_seconds': round(time.monotonic() - started, 3) if started else None,
                       'abort_reason': monitor.reason if monitor else None,
                       'final_metrics': final}
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps({'outcome': outcome, 'exit_code': code, 'output_dir': str(output),
                              'abort_reason': summary['abort_reason']}), flush=True)
    return code


def main(argv=None):
    args = arguments(argv)
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
    try:
        return run_trial(args)
    except Exception as exc:
        print(f'Benchmark failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
