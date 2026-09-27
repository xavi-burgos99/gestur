#!/usr/bin/env python3
"""Measure hand-runtime CPU cost and peak RSS in a fresh Linux/macOS process.

LiteRT runs the two native Lite models on fixed synthetic tensors. MediaPipe
runs its VIDEO Tasks pipeline on one prepared, static image. These are different
workloads: their timings are not an end-to-end performance comparison. Neither
mode opens a camera, downloads assets, or measures recognition accuracy.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import sys
import time


def summarize(values):
    ordered = sorted(values)
    return {
        "median": round(statistics.median(values), 4),
        "p95": round(ordered[math.ceil(.95 * len(ordered)) - 1], 4),
        "mean": round(statistics.mean(values), 4),
    }


def measure(invoke, frames, warmup):
    wall, cpu = [], []
    last = None
    for index in range(frames + warmup):
        cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
        last = invoke(index)
        elapsed = (time.perf_counter_ns() - wall_start) / 1e6
        used = (time.process_time_ns() - cpu_start) / 1e6
        if index >= warmup:
            wall.append(elapsed)
            cpu.append(used)
    return {"wall_ms": summarize(wall), "cpu_ms": summarize(cpu)}, last


def fingerprint(path):
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def benchmark(args):
    report = {
        "mode": args.mode,
        "host": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "threads_requested": args.threads,
        "frames": args.frames,
        "warmup": args.warmup,
        "memory_peak_MB": {},
        "memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; decimal MB",
        "opencv_threads": 1,
    }

    def memory(stage):
        # Darwin reports bytes; Linux reports KiB. This is peak process RSS,
        # not current RSS, system memory, or the model's file size.
        multiplier = 1 if sys.platform == "darwin" else 1024
        report["memory_peak_MB"][stage] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier / 1e6, 3)

    memory("startup")
    import_start = time.perf_counter()
    import numpy as np
    memory("numpy_import")
    import cv2
    cv2.setNumThreads(1)
    memory("opencv_import")
    report["numpy_version"] = np.__version__
    report["opencv_version"] = cv2.__version__

    if args.mode == "lite":
        from ai_edge_litert.interpreter import Interpreter
        report["runtime_version"] = importlib.metadata.version("ai-edge-litert")
        report["scope"] = "synthetic_tensor_model_only"
        memory("runtime_import")
        report["import_ms"] = round((time.perf_counter() - import_start) * 1000, 3)
        nets, details = [], []
        for stem in ("palm_detection_lite", "hand_landmark_lite"):
            options = {} if args.threads == "default" else {"num_threads": int(args.threads)}
            path = args.models_dir / (stem + ".tflite")
            net = Interpreter(model_path=str(path), **options)
            net.allocate_tensors()
            input_info = net.get_input_details()[0]
            tensor = np.random.default_rng(42).random(tuple(input_info["shape"]), dtype=np.float32)
            net.set_tensor(input_info["index"], tensor)
            nets.append(net)  # Keep BOTH models resident throughout the benchmark.
            details.append({
                "model": stem,
                **fingerprint(path),
                "input_shape": input_info["shape"].tolist(),
                "outputs": [
                    {"name": item["name"], "shape": item["shape"].tolist()}
                    for item in net.get_output_details()
                ],
            })
        memory("both_models_allocated")
        report["models"], report["benchmarks"] = details, {}
        for net, detail in zip(nets, details):
            timing, _ = measure(lambda index: net.invoke(), args.frames, args.warmup)
            report["benchmarks"][detail["model"]] = timing
            checks = []
            for output in net.get_output_details():
                value = net.get_tensor(output["index"])
                checks.append({
                    "name": output["name"],
                    "finite": bool(np.isfinite(value).all()),
                    "first_values": value.flatten()[:3].tolist(),
                    "sum": float(value.sum()),
                })
            detail["output_checks"] = checks
        memory("after_inference")
        report["pipeline_warning"] = (
            "invoke() only on fixed synthetic tensors. No image decode/preprocessing, "
            "detector NMS, oriented crops, association, temporal tracking or gestures. "
            "This is not an accuracy test or a full-pipeline benchmark."
        )
    else:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        report["runtime_version"] = mp.__version__
        report["scope"] = "mediapipe_tasks_prepared_image_video"
        memory("runtime_import")
        report["import_ms"] = round((time.perf_counter() - import_start) * 1000, 3)
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError("The image could not be decoded.")
        image = cv2.resize(image, (640, 480))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        model_path = args.models_dir / "hand_landmarker_lite.task"
        report["models"] = [fingerprint(model_path)]
        report["image"] = {**fingerprint(args.image), "prepared_size": [640, 480]}
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
        with HandLandmarker.create_from_options(options) as model:
            memory("model_created")
            counts = []

            def invoke(index):
                result = model.detect_for_video(image, (index + 1) * 67)
                counts.append(len(result.hand_landmarks))
                return result

            timing, _ = measure(invoke, args.frames, args.warmup)
            report["benchmarks"] = {"pipeline": timing}
            report["hands_count_set"] = sorted(set(counts[args.warmup:]))
            report["frames_with_two_hands"] = sum(count == 2 for count in counts[args.warmup:])
            memory("after_inference")
        report["pipeline_warning"] = (
            "Static prepared 640x480 image; decode, resizing, RGB conversion and mp.Image "
            "construction happen outside timing. Tasks includes ROI tracking, detector "
            "scheduling and landmark inference. Not comparable directly to LiteRT tensors; "
            "no ground truth, real 3D motion, camera/render overhead or accuracy measurement."
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lite", "mediapipe"), required=True)
    parser.add_argument("--models-dir", type=Path, required=True, help="Directory containing the local Lite models.")
    parser.add_argument("--image", type=Path, help="Static image; required for mediapipe mode only.")
    parser.add_argument("--threads", choices=("default", "1", "2", "4"), default="default",
                        help="LiteRT CPU threads. MediaPipe Tasks does not expose this option.")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Save JSON in addition to printing it.")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup must be non-negative")
    if args.mode == "mediapipe":
        if not args.image or not args.image.is_file():
            parser.error("mediapipe requires --image pointing to a local image")
        if args.threads != "default":
            parser.error("--threads is only available in lite mode")
    names = ("palm_detection_lite.tflite", "hand_landmark_lite.tflite") if args.mode == "lite" else ("hand_landmarker_lite.task",)
    if not all((args.models_dir / name).is_file() for name in names):
        parser.error("models-dir is missing one or more required files: " + ", ".join(names))
    report = benchmark(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
