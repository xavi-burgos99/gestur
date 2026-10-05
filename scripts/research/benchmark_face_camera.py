"""Compare facial detectors and Pose on the same live frames; save statistics only.

No images, video, landmarks or identity data are written. Detection agreement
is NOT accuracy. Timing excludes shared RGB conversion, camera and rendering.
Each model processes the same frame; order rotates to reduce ordering bias.
"""

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path


def stats(values):
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median": statistics.median(values) if values else None,
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1] if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/opt/gestur"))
    parser.add_argument("--pose-model", type=Path)
    parser.add_argument(
        "--face-model",
        type=Path,
        help="Optional local FaceLandmarker .task; no downloads.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument(
        "--fps",
        type=float,
        default=3,
        help="Maximum paired frame rate, not per-model throughput.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (
        math.isfinite(args.seconds)
        and 0 < args.seconds <= 3600
        and math.isfinite(args.fps)
        and 0 < args.fps <= 30
        and args.warmup >= 0
        and args.width >= 128
        and args.height >= 128
        and 0 < args.confidence < 1
    ):
        parser.error("Invalid duration, cadence, warmup, image size or confidence.")
    pose_path = (
        args.pose_model or args.project / "tracking_models/pose_landmarker_lite.task"
    )
    if not pose_path.is_file() or (args.face_model and not args.face_model.is_file()):
        parser.error("Missing local model; this script never downloads assets.")
    sys.path.insert(0, str(args.project))
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    from tracking_geometry import pose_features

    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        parser.error("Camera could not be opened")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 24)
    condition, stopped = threading.Condition(), threading.Event()
    capture = {"frame": None, "sequence": 0, "failures": 0, "timestamp": None}

    def grab():
        while not stopped.is_set():
            ok, frame = cap.read()
            with condition:
                if not ok:
                    capture["failures"] += 1
                    stopped.set()
                else:
                    capture.update(
                        frame=frame,
                        sequence=capture["sequence"] + 1,
                        timestamp=time.monotonic(),
                    )
                condition.notify_all()

    thread = threading.Thread(target=grab, daemon=True)
    thread.start()
    data, paired = {}, {}
    actual_size = None
    started = cpu_started = None
    elapsed = cpu_elapsed = 0
    error = None
    try:
        with ExitStack() as resources:

            def base(path):
                return BaseOptions(
                    model_asset_path=str(path), delegate=BaseOptions.Delegate.CPU
                )

            pose = resources.enter_context(
                PoseLandmarker.create_from_options(
                    PoseLandmarkerOptions(
                        base_options=base(pose_path),
                        running_mode=RunningMode.VIDEO,
                        num_poses=1,
                        min_pose_detection_confidence=args.confidence,
                        min_pose_presence_confidence=args.confidence,
                        min_tracking_confidence=args.confidence,
                        output_segmentation_masks=False,
                    )
                )
            )
            short = resources.enter_context(
                mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=args.confidence
                )
            )
            full = resources.enter_context(
                mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=args.confidence
                )
            )
            runners = {
                "pose_lite": lambda rgb, image, stamp: pose.detect_for_video(
                    image, stamp
                ),
                "face_short_range": lambda rgb, image, stamp: short.process(rgb),
                "face_full_sparse": lambda rgb, image, stamp: full.process(rgb),
            }
            if args.face_model:
                face = resources.enter_context(
                    FaceLandmarker.create_from_options(
                        FaceLandmarkerOptions(
                            base_options=base(args.face_model),
                            running_mode=RunningMode.VIDEO,
                            num_faces=1,
                            min_face_detection_confidence=args.confidence,
                            min_face_presence_confidence=args.confidence,
                            min_tracking_confidence=args.confidence,
                            output_face_blendshapes=False,
                            output_facial_transformation_matrixes=True,
                        )
                    )
                )
                runners["face_landmarker"] = lambda rgb, image, stamp: (
                    face.detect_for_video(image, stamp)
                )
            for name in runners:
                data[name] = {
                    "wall_ms": [],
                    "cpu_ms": [],
                    "valid_frames": 0,
                    "frames": 0,
                    "finite_rotation_matrix_frames": 0,
                }
                paired[name] = {
                    "both_detected": 0,
                    "pose_only": 0,
                    "candidate_only": 0,
                    "neither": 0,
                }
            index, previous_sequence, last_stamp = 0, -1, -1
            next_due = time.monotonic()
            if args.warmup == 0:
                started, cpu_started = time.monotonic(), time.process_time()
            while not stopped.is_set():
                if started is not None and time.monotonic() - started >= args.seconds:
                    break
                with condition:
                    condition.wait_for(
                        lambda: (
                            stopped.is_set()
                            or (
                                capture["frame"] is not None
                                and capture["sequence"] != previous_sequence
                            )
                        ),
                        timeout=1,
                    )
                    if stopped.is_set():
                        break
                    if (
                        capture["frame"] is None
                        or capture["sequence"] == previous_sequence
                    ):
                        continue
                # Camera notifications do not shorten this cadence wait.
                if stopped.wait(max(0, next_due - time.monotonic())):
                    break
                with condition:
                    frame, captured = capture["frame"], capture["timestamp"]
                    previous_sequence = capture["sequence"]
                actual_size = [frame.shape[1], frame.shape[0]]
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                stamp = max(last_stamp + 1, int(captured * 1000))
                last_stamp = stamp
                names = list(runners)
                names = names[index % len(names) :] + names[: index % len(names)]
                valid = {}
                for name in names:
                    wall, cpu = time.perf_counter(), time.process_time()
                    result = runners[name](rgb, image, stamp)
                    wall, cpu = (
                        (time.perf_counter() - wall) * 1000,
                        (time.process_time() - cpu) * 1000,
                    )
                    matrix_ok = False
                    if name == "pose_lite":
                        head = (
                            pose_features(
                                result.pose_landmarks[0],
                                result.pose_world_landmarks[0],
                                visibility_threshold=args.confidence,
                                aspect=actual_size[0] / actual_size[1],
                            )["head"]
                            if result.pose_landmarks and result.pose_world_landmarks
                            else {}
                        )
                        valid[name] = bool(head.get("detected"))
                    elif name == "face_landmarker":
                        valid[name] = bool(result.face_landmarks)
                        matrix_ok = any(
                            np.asarray(m).shape == (4, 4) and np.isfinite(m).all()
                            for m in result.facial_transformation_matrixes
                        )
                    else:
                        detections = result.detections or []
                        valid[name] = any(
                            len(d.location_data.relative_keypoints) >= 6
                            and all(
                                math.isfinite(p.x) and math.isfinite(p.y)
                                for p in d.location_data.relative_keypoints
                            )
                            for d in detections
                        )
                    if index >= args.warmup:
                        record = data[name]
                        record["wall_ms"].append(wall)
                        record["cpu_ms"].append(cpu)
                        record["valid_frames"] += valid[name]
                        record["frames"] += 1
                        record["finite_rotation_matrix_frames"] += matrix_ok
                if index >= args.warmup:
                    for name in names:
                        key = (
                            "both_detected"
                            if valid["pose_lite"] and valid[name]
                            else (
                                "pose_only"
                                if valid["pose_lite"]
                                else "candidate_only"
                                if valid[name]
                                else "neither"
                            )
                        )
                        paired[name][key] += 1
                index += 1
                if args.warmup > 0 and index == args.warmup:
                    started, cpu_started = time.monotonic(), time.process_time()
                next_due = max(next_due + 1 / args.fps, time.monotonic())
            if started is not None:
                elapsed, cpu_elapsed = (
                    time.monotonic() - started,
                    time.process_time() - cpu_started,
                )
    except Exception as exc:
        error = str(exc)
    finally:
        stopped.set()
        thread.join(timeout=2)
        cap.release()
    for record in data.values():
        record["wall_ms"] = stats(record["wall_ms"])
        record["cpu_ms"] = stats(record["cpu_ms"])
        record["valid_fraction"] = (
            record["valid_frames"] / record["frames"] if record["frames"] else None
        )

    def model_info(path):
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    report = {
        "platform": platform.platform(),
        "mediapipe": mp.__version__,
        "models": {
            "pose": model_info(pose_path),
            "face_landmarker": model_info(args.face_model) if args.face_model else None,
        },
        "seconds": elapsed,
        "paired_fps_limit": args.fps,
        "warmup": args.warmup,
        "mirror": not args.no_mirror,
        "camera_requested_size": [args.width, args.height],
        "camera_actual_size": actual_size,
        "capture_frames": capture["sequence"],
        "capture_failures": capture["failures"],
        "process_cpu_percent_one_core": 100 * cpu_elapsed / elapsed
        if elapsed
        else None,
        "peak_rss_MB_combined_models": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024)
        / 1e6,
        "results": data,
        "agreement_with_pose_NOT_accuracy": paired,
        "error": error,
        "limitations": [
            "Only aggregate statistics saved; no photos/video/landmarks.",
            "All candidate models resident: total CPU/RSS is not a production backend measurement.",
            "Same image at capped cadence; no ground truth or 3D orientation error measurement.",
            "FaceDetection provides 2D points only; it cannot replace three head rotation axes.",
            "Camera acquisition and shared preprocessing excluded from individual model timings.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if error or capture["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
