"""Count rendered triangles, excluding points/lines, with the real Panda loader."""

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_model", ROOT / "scripts/check_model.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def mesh_glb(mode, vertices):
    binary = struct.pack("<" + "f" * len(vertices), *vertices)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": mode}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteLength": len(binary)}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices) // 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            }
        ],
    }
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    return (
        struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(binary))
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


class ModelPreflightTests(unittest.TestCase):
    def test_triangle_strip_is_counted_as_two_triangles(self):
        from panda3d.core import (
            Geom,
            GeomNode,
            GeomTristrips,
            GeomVertexData,
            GeomVertexFormat,
            NodePath,
        )

        vertices = GeomVertexData("strip", GeomVertexFormat.get_v3(), Geom.UH_static)
        vertices.set_num_rows(4)
        strip = GeomTristrips(Geom.UH_static)
        strip.add_vertices(0, 1, 2, 3)
        strip.close_primitive()
        geom = Geom(vertices)
        geom.add_primitive(strip)
        node = GeomNode("strip")
        node.add_geom(geom)
        model = NodePath("root")
        model.attach_new_node(node)
        self.assertEqual(MODULE.count_triangles(model), 2)

    def test_line_only_model_has_no_visible_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "line.glb"
            model.write_bytes(mesh_glb(1, [0, 0, 0, 1, 0, 0]))
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/check_model.py"), str(model)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(result.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
