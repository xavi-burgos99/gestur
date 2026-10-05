#!/usr/bin/env python3
"""Validate an imported model in a disposable process, without a GPU or camera."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def count_triangles(model):
    from panda3d.core import Geom

    return sum(
        node.node()
        .get_geom(index)
        .get_primitive(primitive)
        .decompose()
        .get_num_primitives()
        for node in model.find_all_matches("**/+GeomNode")
        for index in range(node.node().get_num_geoms())
        for primitive in range(node.node().get_geom(index).get_num_primitives())
        if node.node().get_geom(index).get_primitive(primitive).get_primitive_type()
        == Geom.PT_polygons
    )


def inspect_model(path):
    from gestur.visualizer import ControlledObjViewer

    viewer = ControlledObjViewer(path, window_type="none", fullscreen=False)
    try:
        triangles = count_triangles(viewer.model)
        if not triangles:
            raise ValueError("El modelo no contiene caras visibles")
        return {
            "ok": True,
            "primitives": triangles,
            "triangles": triangles,
            "textures": len(viewer.model.find_all_textures()),
        }
    finally:
        viewer.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validar un modelo con el mismo cargador del expositor"
    )
    parser.add_argument("model")
    args = parser.parse_args(argv)
    if sys.platform.startswith("linux"):
        # An invalid/native loader can fail in this child without taking down the
        # portal or exhausting a Pi. The caller also imposes a wall-clock limit.
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    try:
        with redirect_stdout(sys.stderr):
            result = inspect_model(Path(args.model).resolve(strict=True))
    except Exception:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "El motor 3D no pudo cargar el modelo. Revisa la geometría y los archivos del ZIP.",
                }
            )
        )
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
