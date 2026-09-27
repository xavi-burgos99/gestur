"""Native tracker/controls smoke trial for torso and relative hand proximity.

Default: 60 seconds of the official still-image replay, without camera or renderer.
Use --source camera for a real-camera trial. Saves aggregate JSON only: no frames,
landmarks, coordinates, local asset paths, IP addresses, or credentials.
A finite result is a contract check, never a measurement of recognition accuracy.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

SIGNALS = tuple(
    f"torso_{field}" for field in ("x", "y", "scale", "pitch", "yaw", "roll")
) + ("left_hand_scale", "right_hand_scale")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/opt/gestur"))
    parser.add_argument("--source", choices=("replay", "camera"), default="replay")
    parser.add_argument(
        "--image", type=Path, default=Path("/home/gestur/gestur-bench/woman_hands.jpg")
    )
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 120:
        parser.error("--seconds must be between 1 and 120.")
    if not 0 <= args.camera <= 16:
        parser.error("--camera must be between 0 and 16.")
    if args.output.exists():
        parser.error("--output must be a new file.")
    return args


def configured(base, side):
    config = deepcopy(base)
    config["tracking"].update(
        use_pose=True,
        use_hands=True,
        width=640,
        height=480,
        inference_fps=24,
        hand_fps=15,
        smoothing_ms=60,
    )
    assignments = [
        ("torso_x", "position_x", 2),
        ("torso_y", "position_y", 2),
        ("torso_scale", "position_z", 2),
        ("torso_pitch", "rotation_pitch", 70),
        ("torso_yaw", "rotation_yaw", 70),
        ("torso_roll", "rotation_roll", 70),
        (f"{side}_hand_scale", "scale_uniform", 2),
    ]
    config["controls"]["mappings"] = [
        dict(
            id=source,
            input=source,
            output=target,
            mode="absolute",
            enabled=True,
            scale=sensitivity,
            invert=False,
            center=0.5,
        )
        for source, target, sensitivity in assignments
    ]
    return config


class ReplayCapture:
    """A single prepared image at 24 Hz, with bounded, interruptible waiting."""

    def __init__(self, image, stop_event):
        self.image, self.stopped = image, stop_event
        self.closed = False
        self.next_frame = time.monotonic()

    def isOpened(self):
        return not self.closed

    def read(self):
        if self.closed or self.stopped.wait(max(0, self.next_frame - time.monotonic())):
            return False, None
        self.next_frame = time.monotonic() + 1 / 24
        return True, self.image.copy()

    def release(self):
        self.closed = True


def main():
    args = arguments()
    report = {
        "trial": "torso-and-hand-proximity-contract",
        "source": args.source,
        "requested_seconds": args.seconds,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "fixture": None,
        "outcome": "failed",
        "scope": {
            "camera_requested": args.source == "camera",
            "renderer_started": False,
            "production_state_modified": False,
            "inference_models": "Installed MediaPipe Pose Lite and Hand Lite, CPU VIDEO",
            "input_size": [640, 480],
            "replay_fps": 24 if args.source == "replay" else None,
            "signal_samples": "Fresh tracker publications, which can include expiry; not independent video frames.",
            "control_samples": "Two independent control mappers for left/right proximity, consuming the same tracker.",
        },
    }
    report["limitations"] = [
        "Single still-image replay cannot validate movement accuracy, occlusion recovery or true 3D distance.",
        "Scale is a monocular relative proximity proxy, not absolute world.z or metres.",
        "Missing torso or hand detections may make coverage incomplete; not automatically a runtime defect.",
        "Body-only cadence with an occluded head is observed only if such frames occur.",
        "Replay teardown can increment the final capture-failure counter; pre-cleanup metrics are separate.",
        "No renderer, optical latency or comparison with historical system benchmarks.",
    ]
    report["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tracker = None
    elapsed = cpu_seconds = 0.0
    measured_started = cpu_started = None
    measuring = threading.Event()
    lock = threading.Lock()
    errors = set()
    counts = {
        name: dict(finite=0, missing=0, invalid=0, range_violations=0)
        for name in SIGNALS
    }
    observed = dict(
        publications=0,
        head_detected=0,
        torso_detected=0,
        left_hand_detected=0,
        right_hand_detected=0,
        torso_without_head=0,
        torso_without_head_idle=0,
    )
    control_samples = {"left": 0, "right": 0}
    try:
        if not (args.project / "tracking_geometry.py").is_file():
            raise FileNotFoundError("Project files missing")
        sys.path.insert(0, str(args.project))
        import cv2
        import numpy as np

        from control_system import create_control_system, create_extractors
        from pose_detector import PoseHandTracker
        from runtime_config import default_config, validate_config
        from tracking_session import tracking_request

        cv2.setNumThreads(1)
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("mediapipe", "numpy", "opencv-contrib-python")
        }
        report["code_sha256"] = {
            name: hashlib.sha256((args.project / name).read_bytes()).hexdigest()
            for name in (
                "tracking_geometry.py",
                "tracking_session.py",
                "pose_detector.py",
                "control_system.py",
                "config/schema.json",
            )
        }
        configs = {
            side: validate_config(configured(default_config(), side))
            for side in ("left", "right")
        }
        for config in configs.values():
            config["tracking"]["camera_index"] = args.camera
        requests = {
            side: tracking_request(config, has_model=True)
            for side, config in configs.items()
        }
        request = requests["left"]
        if (
            requests["right"] != request
            or not request
            or not request["use_pose"]
            or not request["use_hands"]
            or tuple(request.get("pose_parts", ())) != ("torso",)
        ):
            raise ValueError("Unexpected detector demand")
        report["requested_tracking"] = request
        disabled = deepcopy(configs["left"])
        disabled["tracking"]["use_pose"] = False
        hands_only = tracking_request(disabled, has_model=True)
        report["gating_checks"] = {
            "left_and_right_profiles_request_identical_detectors": requests["left"]
            == requests["right"],
            "torso_only_pose_presence": tuple(request["pose_parts"]) == ("torso",),
            "global_pose_switch_leaves_only_hands": bool(
                hands_only and not hands_only["use_pose"] and hands_only["use_hands"]
            ),
            "no_model_disables_tracking": tracking_request(
                configs["left"], has_model=False
            )
            is None,
            "no_camera_disables_tracking": tracking_request(
                configs["left"], no_camera=True
            )
            is None,
        }
        if not all(report["gating_checks"].values()):
            raise ValueError("Detector gating failed")
        systems = {
            side: create_control_system(config) for side, config in configs.items()
        }
        extractors = create_extractors()
        tracker = PoseHandTracker(**request)
        if args.source == "replay":
            image = cv2.imread(str(args.image))
            if image is None:
                raise ValueError("Replay image unreadable")
            h, w = image.shape[:2]
            ratio = min(640 / w, 480 / h)
            rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            canvas[
                (480 - rh) // 2 : (480 - rh) // 2 + rh,
                (640 - rw) // 2 : (640 - rw) // 2 + rw,
            ] = cv2.resize(image, (rw, rh))
            report["fixture"] = {
                "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
                "original_size": [w, h],
                "preparation": "Aspect-preserving letterbox to 640x480.",
            }
            tracker._open_camera = lambda: ReplayCapture(canvas, tracker._stop_event)

        def receive(data):
            if not measuring.is_set():
                return
            with lock:
                if not measuring.is_set():
                    return
                try:
                    observed["publications"] += 1
                    for part in ("head", "torso", "left_hand", "right_hand"):
                        observed[f"{part}_detected"] += bool(
                            data.get(part, {}).get("detected")
                        )
                    if data.get("torso", {}).get("detected") and not data.get(
                        "head", {}
                    ).get("detected"):
                        observed["torso_without_head"] += 1
                        idle = tracker.get_metrics().get("pose_idle", False)
                        observed["torso_without_head_idle"] += bool(idle)
                        if idle:
                            errors.add("requested_torso_detected_but_pose_idle")
                    for name in SIGNALS:
                        part, field = name.rsplit("_", 1)
                        sample = data.get(part, {})
                        if field not in sample:
                            errors.add(f"missing_schema_field:{name}")
                        raw = sample.get(field)
                        if raw is not None and (
                            type(raw) not in (float, int) or not math.isfinite(raw)
                        ):
                            counts[name]["invalid"] += 1
                            errors.add(f"nonfinite_or_non_numeric:{name}")
                        if not sample.get("detected") and raw is not None:
                            errors.add(f"uncleared_missing_part:{name}")
                        value = extractors[name](data)
                        if value is None:
                            counts[name]["missing"] += 1
                        elif math.isfinite(value):
                            counts[name]["finite"] += 1
                            if field == "scale" and not 0 <= value <= 1:
                                counts[name]["range_violations"] += 1
                                errors.add(f"proximity_out_of_range:{name}")
                        else:
                            counts[name]["invalid"] += 1
                            errors.add(f"nonfinite_extractor:{name}")
                except Exception as exc:
                    errors.add(f"callback_exception:{type(exc).__name__}")

        tracker.subscribe(receive)
        startup = time.monotonic()
        tracker.run()
        report["startup_seconds"] = time.monotonic() - startup
        measured_started, cpu_started = time.monotonic(), time.process_time()
        measuring.set()
        while time.monotonic() - measured_started < args.seconds:
            if tracker.last_error or not tracker.running:
                errors.add("tracker_stopped_or_failed")
                if tracker.last_error:
                    report["tracker_error_type"] = type(tracker.last_error).__name__
                break
            data = tracker.get_current_data()
            for side, system in systems.items():
                output = system.process_input(data)
                values = output["position"] + output["rotation"] + [output["scale"]]
                control_samples[side] += 1
                if not all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in values
                ):
                    errors.add(f"nonfinite_control_output:{side}")
            time.sleep(
                min(0.05, max(0, args.seconds - (time.monotonic() - measured_started)))
            )
        measuring.clear()
        elapsed, cpu_seconds = (
            time.monotonic() - measured_started,
            time.process_time() - cpu_started,
        )
        report["metrics_before_cleanup"] = tracker.get_metrics()
        with lock:
            report["signals"] = deepcopy(counts)
            report["observations"] = dict(observed)
        report["control_samples"] = control_samples
        if not report["metrics_before_cleanup"].get("pose_frames") or not report[
            "metrics_before_cleanup"
        ].get("hand_frames"):
            errors.add("missing_native_model_inference")
        missing_coverage = [
            name for name, record in counts.items() if not record["finite"]
        ]
        report["unobserved_signals"] = missing_coverage
        report["outcome"] = (
            "failed"
            if errors
            else "incomplete_coverage"
            if missing_coverage
            else "passed"
        )
    except Exception as exc:
        if measured_started is not None:
            elapsed, cpu_seconds = (
                time.monotonic() - measured_started,
                time.process_time() - cpu_started,
            )
        report["exception_type"] = type(exc).__name__
        errors.add("setup_or_execution_exception")
    finally:
        measuring.clear()
        if tracker is not None:
            try:
                tracker.stop()
                report["metrics_after_cleanup"] = tracker.get_metrics()
            except Exception as exc:
                errors.add("cleanup_exception")
                report["cleanup_exception_type"] = type(exc).__name__
        report.update(
            seconds=elapsed,
            process_cpu_seconds=cpu_seconds,
            process_cpu_percent_one_core=100 * cpu_seconds / elapsed
            if elapsed
            else None,
            peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
            / 1e6,
            errors=sorted(errors),
        )
        if errors:
            report["outcome"] = "failed"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "outcome": report["outcome"],
                    "seconds": elapsed,
                    "unobserved_signals": report.get("unobserved_signals", []),
                    "errors": sorted(errors),
                }
            )
        )
    return 1 if errors else 2 if report["outcome"] == "incomplete_coverage" else 0


if __name__ == "__main__":
    raise SystemExit(main())
