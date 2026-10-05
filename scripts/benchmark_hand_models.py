#!/usr/bin/env python3
"""Compare local MediaPipe hand bundles on a fixed image and synthetic motion.

No camera access or downloads. This measures CPU cost, detection continuity and
repeatability, not recognition accuracy: synthetic 2D tilts are not real 3D wrist
rotations and the image has no ground-truth hand landmarks.
"""

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Direct execution from scripts/ still needs the repository's geometry module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking_geometry import hand_features, wrap_angle  # noqa: E402


def summary(values):
    if not values:
        return {"samples": 0, "median": None, "p95": None}
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
    }


def fingerprint(path):
    payload = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def transformed_image(frame, index, scenario, cv2):
    height, width = frame.shape[:2]
    if scenario == "static":
        angle, scale, dx, dy = 0, 1, 0, 0
    else:
        phase = index * 2 * math.pi / 60
        angle = 20 * math.sin(phase)
        scale = 0.96 + 0.03 * math.cos(phase)
        dx, dy = 8 * math.sin(phase / 2), 6 * math.cos(phase / 2)
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    transform[:, 2] += (dx, dy)
    image = cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return image, cv2.invertAffineTransform(transform), angle


def observations(result, inverse, angle, shape, np):
    height, width = shape[:2]
    samples = []
    for normalized, world, categories in zip(
        result.hand_landmarks, result.hand_world_landmarks, result.handedness
    ):
        if len(normalized) != 21 or len(world) != 21 or not categories:
            continue
        label = max(categories, key=lambda item: item.score).category_name
        image_points = (
            np.array([[p.x * width, p.y * height, 1] for p in normalized]) @ inverse.T
        )
        image_points /= (width, height)
        coords = np.array([[p.x, p.y, p.z] for p in world])
        coords -= coords[0]
        palm_width = np.linalg.norm(coords[5] - coords[17])
        if not np.isfinite(coords).all() or palm_width <= 1e-5:
            continue
        coords /= palm_width
        # OpenCV's positive image angle maps to a negative camera-space Z
        # rotation (image Y points down). Undo it before comparing 3D outputs.
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        undo = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        coords = coords @ undo.T
        feature = hand_features(
            [SimpleNamespace(x=x, y=y, z=0) for x, y in image_points],
            [SimpleNamespace(x=x, y=y, z=z) for x, y, z in coords],
            label,
            width / height,
        )
        if feature["detected"]:
            samples.append((image_points[0, 0], coords, feature, label))
    # Shared affine motion keeps left-to-right ordering of the two image hands.
    # No association is inferred on frames where a hand disappears.
    return sorted(samples, key=lambda sample: sample[0])


def benchmark(model, frame, scenario, frames, warmup, expected_hands):
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=expected_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    wall, cpu, detected, valid = [], [], [], []
    point_steps, angle_steps = [], {name: [] for name in ("pitch", "yaw", "roll")}
    previous = None
    tracks = [[] for _ in range(expected_hands)]
    with vision.HandLandmarker.create_from_options(options) as detector:
        for index in range(frames + warmup):
            image, inverse, angle = transformed_image(frame, index, scenario, cv2)
            cpu_started, started = time.process_time_ns(), time.perf_counter_ns()
            media_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
            result = detector.detect_for_video(media_image, (index + 1) * 67)
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            cpu_ms = (time.process_time_ns() - cpu_started) / 1e6
            if index < warmup:
                continue
            wall.append(wall_ms)
            cpu.append(cpu_ms)
            detected.append(len(result.hand_landmarks))
            samples = observations(result, inverse, angle, frame.shape, np)
            valid.append(len(samples))
            if len(samples) != expected_hands:
                previous = None
                continue
            for slot, sample in enumerate(samples):
                tracks[slot].append(sample[1])
            if previous is not None:
                for old, current in zip(previous, samples):
                    if old[3] != current[3]:
                        continue
                    point_steps.append(
                        float(
                            np.sqrt(np.mean(np.sum((current[1] - old[1]) ** 2, axis=1)))
                        )
                    )
                    for field in angle_steps:
                        angle_steps[field].append(
                            abs(wrap_angle(current[2][field] - old[2][field]))
                        )
            previous = samples
    variation = []
    for track in tracks:
        if track:
            values = np.array(track)
            reference = np.median(values, axis=0)
            variation.extend(
                np.sqrt(
                    np.mean(np.sum((values - reference) ** 2, axis=2), axis=1)
                ).tolist()
            )
    return {
        "scenario": scenario,
        "measured_frames": frames,
        "warmup_frames": warmup,
        "expected_hands": expected_hands,
        "frames_with_any_detection": sum(count > 0 for count in detected),
        "frames_with_all_detections": sum(
            count == expected_hands for count in detected
        ),
        "frames_with_all_valid_geometry": sum(
            count == expected_hands for count in valid
        ),
        "wall_ms": summary(wall),
        "cpu_ms": summary(cpu),
        "landmark_step_palm_widths": summary(point_steps),
        "landmark_deviation_from_median_palm_widths": summary(variation),
        "orientation_step_degrees": {
            name: summary(values) for name, values in angle_steps.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hands", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 2 or args.warmup < 0 or min(args.width, args.height) < 128:
        parser.error("Use at least 2 frames, non-negative warmup, dimensions >=128.")
    models = []
    for item in args.model:
        name, separator, source = item.partition("=")
        if not name or not separator or not Path(source).is_file():
            parser.error("--model must be NAME=PATH pointing to a local bundle.")
        models.append((name, Path(source)))
    import cv2
    import mediapipe as mp
    import numpy as np

    cv2.setNumThreads(1)
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error("The source image could not be opened.")
    height, width = frame.shape[:2]
    scale = min(args.width / width, args.height / height)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
    frame = np.full((args.height, args.width, 3), 255, dtype=np.uint8)
    top, left = (
        (args.height - resized.shape[0]) // 2,
        (args.width - resized.shape[1]) // 2,
    )
    frame[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    report = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mediapipe": mp.__version__,
        },
        "delegate": "CPU",
        "image": fingerprint(args.image),
        "image_shape": list(frame.shape),
        "timing_scope": "BGR-to-RGB conversion, mp.Image creation, inference; excludes synthetic transform and geometry analysis",
        "limitations": [
            "Repeated image and deterministic 2D motion, not a real motion video.",
            "No ground-truth pose: continuity and repeatability do not measure accuracy.",
            "Results describe this machine and do not predict Raspberry Pi performance.",
            "Orientation and world landmarks compensate only known image rotation; no true 3D tilt is simulated.",
        ],
        "models": [],
    }
    for name, model in models:
        report["models"].append(
            {
                "name": name,
                **fingerprint(model),
                "results": [
                    benchmark(
                        model, frame, scenario, args.frames, args.warmup, args.hands
                    )
                    for scenario in ("static", "affine_motion")
                ],
            }
        )
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
