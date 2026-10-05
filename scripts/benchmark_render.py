#!/usr/bin/env python3
"""Compare draw scheduling/clock cost with the same synthetic GPU workload.

This does not open a desktop window. It measures a real offscreen framebuffer,
not the camera, inference, thermal throttling, GPU power or display latency.
"""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda3d.core import (  # noqa: E402
    ConfigVariableDouble,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    PandaSystem,
    Texture,
)

from gestur.visualizer import ControlledObjViewer  # noqa: E402


def make_fixture(destination):
    segments = sides = 512
    data = GeomVertexData(
        "benchmark-ring", GeomVertexFormat.get_v3n3t2(), Geom.UH_static
    )
    data.set_num_rows((segments + 1) * (sides + 1))
    vertex, normal, uv = (
        GeomVertexWriter(data, name) for name in ("vertex", "normal", "texcoord")
    )
    triangles = GeomTriangles(Geom.UH_static)
    triangles.set_index_type(Geom.NT_uint32)
    for i in range(segments + 1):
        angle = i * math.tau / segments
        for j in range(sides + 1):
            tube = j * math.tau / sides
            radius = 1.75 + 0.42 * math.cos(tube)
            vertex.add_data3(
                radius * math.cos(angle),
                radius * math.sin(angle),
                0.42 * math.sin(tube),
            )
            normal.add_data3(
                math.cos(tube) * math.cos(angle),
                math.cos(tube) * math.sin(angle),
                math.sin(tube),
            )
            uv.add_data2(i / segments, j / sides)
            if i < segments and j < sides:
                a = i * (sides + 1) + j
                b = a + sides + 1
                triangles.add_vertices(a, b, a + 1)
                triangles.add_vertices(a + 1, b, b + 1)
    geom = Geom(data)
    geom.add_primitive(triangles)
    node = GeomNode("benchmark-ring")
    node.add_geom(geom)
    root = NodePath("benchmark")
    root.attach_new_node(node)
    texture = Texture("benchmark-texture")
    texture.setup_2d_texture(2048, 2048, Texture.T_unsigned_byte, Texture.F_rgb)
    texture.set_ram_image(bytes((85, 180, 150)) * (2048 * 2048))
    root.set_texture(texture)
    root.write_bam_file(str(destination))
    root.remove_node()


def measure(
    name,
    asset,
    seconds,
    *,
    continuous=False,
    moving=False,
    welcome=False,
    precision=0.004,
):
    viewer = ControlledObjViewer(
        None if welcome else asset, window_type="offscreen", fullscreen=False
    )
    sleep_precision = ConfigVariableDouble("sleep-precision")
    sleep_precision.set_value(precision)
    samples = []
    try:
        if continuous:
            viewer.taskMgr.remove("gestur-render-cadence")
            viewer.win.set_active(True)

        def control(task):
            samples.append(time.monotonic())
            if moving:
                viewer.update_model(rotation=[task.time * 30, 10, 0])
            return task.cont

        viewer.taskMgr.add(control, "benchmark-control", sort=10)
        warmup = time.monotonic()
        while time.monotonic() - warmup < 0.5:
            viewer.taskMgr.step()
        samples.clear()
        started = time.monotonic()
        cpu = time.process_time()
        frames = viewer.render_metrics.frames
        while time.monotonic() - started < seconds:
            viewer.taskMgr.step()
        elapsed = time.monotonic() - started
        cpu = time.process_time() - cpu
        draws = viewer.render_metrics.frames - frames
        intervals = sorted((b - a) * 1000 for a, b in zip(samples, samples[1:]))
        p95 = intervals[int((len(intervals) - 1) * 0.95)] if intervals else None
        return {
            "case": name,
            "seconds": round(elapsed, 4),
            "process_cpu_seconds": round(cpu, 4),
            "cpu_percent_one_core": round(cpu / elapsed * 100, 2),
            "draws": draws,
            "control_ticks": len(samples),
            "draw_fps": round(draws / elapsed, 3),
            "control_fps": round(len(samples) / elapsed, 3),
            "control_ms_p95": round(p95, 3) if p95 is not None else None,
            "sleep_precision_seconds": precision,
            "renderer": viewer.win.get_gsg().get_driver_renderer(),
            "framebuffer": [viewer.win.get_x_size(), viewer.win.get_y_size()],
            "msaa": viewer.win.get_fb_properties().get_multisamples(),
            "triangles": 1536 if welcome else 524288,
            "texture": None if welcome else [2048, 2048],
        }
    finally:
        viewer.taskMgr.remove("benchmark-control")
        viewer.destroy()
        sleep_precision.clear_local_value()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds", type=float, default=5, help="Seconds per scenario, default 5"
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result file")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    results = []
    with TemporaryDirectory(prefix="gestur-render-") as directory:
        asset = Path(directory) / "synthetic.bam"
        make_fixture(asset)
        scenarios = (
            ("previous-static", {"continuous": True, "precision": 0.01}),
            ("clock-only-static", {"continuous": True}),
            ("optimized-static", {}),
            ("optimized-moving", {"moving": True}),
            ("optimized-welcome", {"welcome": True}),
        )
        for name, options in scenarios:
            result = measure(name, asset, args.seconds, **options)
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "platform": platform.platform(),
        "panda3d": PandaSystem.get_version_string(),
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
