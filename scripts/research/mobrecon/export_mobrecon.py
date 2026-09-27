"""Minimal CPU MobRecon inference/export, with public checkpoint and geometry.

The image must already be a single-hand crop. No detector, tracking, mirroring,
absolute depth recovery, pose confidence or camera/render pipeline is supplied.
Output joints use the public regressor's native order, not an assumed Gestur order.
"""

import argparse
import hashlib
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def peak_mb():
    return (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024)
        / 1e6
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "pytorch-results.json")
    args = parser.parse_args()
    if args.threads < 1 or args.frames < 1 or args.warmup < 0:
        parser.error("Invalid measurement configuration")
    import cv2
    import numpy as np
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT / "source"))
    from cmr.models.mobrecon_densestack import MobRecon

    geometry = np.load(ROOT / "geometry.npz", allow_pickle=False)
    spirals = [torch.from_numpy(geometry[f"spiral_{i}"]) for i in range(4)]
    up = [
        tuple(
            torch.from_numpy(geometry[f"up_{field}_{i}"])
            for field in ("row", "col", "val")
        )
        for i in range(4)
    ]
    model = MobRecon(
        SimpleNamespace(out_channels=[32, 64, 128, 256], dsconv=True), spirals, up
    )
    checkpoint_path = ROOT / "mobrecon_densestack_dsconv.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    if args.image:
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            parser.error("Cannot read image")
        image = cv2.cvtColor(cv2.resize(bgr, (128, 128)), cv2.COLOR_BGR2RGB)
        array = ((image.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    else:
        array = (
            np.random.default_rng(42)
            .uniform(-1, 1, (1, 3, 128, 128))
            .astype(np.float32)
        )
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    walls, cpus = [], []
    with torch.inference_mode():
        for index in range(args.frames + args.warmup):
            cpu, wall = time.process_time(), time.perf_counter()
            prediction = model(tensor)
            elapsed, used = (
                (time.perf_counter() - wall) * 1000,
                (time.process_time() - cpu) * 1000,
            )
            if index >= args.warmup:
                walls.append(elapsed)
                cpus.append(used)
    mesh = prediction["mesh_pred"].numpy() * 0.2
    uv = prediction["uv_pred"].numpy()
    joints = geometry["joint_regressor"].astype(np.float32) @ mesh
    np.savez(
        ROOT / "pytorch-reference.npz",
        input=array,
        mesh_meters=mesh,
        uv_normalized=uv,
        joints_public_order=joints,
    )
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_commit": json.loads((ROOT / "manifest.json").read_text())["commit"],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "threads": args.threads,
        "frames": args.frames,
        "warmup": args.warmup,
        "image": str(args.image) if args.image else None,
        "wall_median_ms": statistics.median(walls),
        "wall_p95_ms": sorted(walls)[int((len(walls) - 1) * 0.95)],
        "cpu_median_ms": statistics.median(cpus),
        "peak_rss_MB_before_export": peak_mb(),
        "outputs": {
            k: {"shape": list(v.shape), "finite": bool(np.isfinite(v).all())}
            for k, v in [
                ("mesh_meters", mesh),
                ("uv_normalized", uv),
                ("joints_public_order", joints),
            ]
        },
        "scope": "Prepared single-hand tensor; model forward only. No detection, confidence, camera or rendering.",
        "limitations": [
            "No ground-truth 3D or accuracy comparison.",
            "No MANO_RIGHT.pkl was loaded. Public geometry remains subject to its applicable license.",
            "The public joint regressor order has not been adapted to Gestur.",
            "Relative mesh scale follows official demo factor 0.2, without camera-space registration.",
        ],
    }
    if args.export:

        class Export(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                output = self.inner(x)
                return output["mesh_pred"] * 0.2, output["uv_pred"]

        torch.onnx.export(
            Export(model).eval(),
            tensor,
            str(args.export),
            opset_version=17,
            input_names=["rgb_crop"],
            output_names=["mesh_meters", "uv_normalized"],
            dynamo=False,
            external_data=False,
        )
        report["onnx"] = {
            "path": str(args.export),
            "bytes": args.export.stat().st_size,
            "sha256": hashlib.sha256(args.export.read_bytes()).hexdigest(),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
