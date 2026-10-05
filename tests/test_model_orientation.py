"""Fixed model orientation is persisted separately from gesture transforms."""

import json
import os

import pytest

from runtime_config import (
    ConfigurationError,
    default_config,
    load_model_orientation,
    save_config,
    validate_model_orientation,
)


def box_model(root):
    """An off-center non-cubic box makes axis swaps and centering observable."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "model.obj"
    path.write_text(
        "\n".join(
            [
                "v 1 2 3",
                "v 3 2 3",
                "v 3 6 3",
                "v 1 6 3",
                "v 1 2 9",
                "v 3 2 9",
                "v 3 6 9",
                "v 1 6 9",
                "f 1 3 2",
                "f 1 4 3",
                "f 5 6 7",
                "f 5 7 8",
                "f 1 2 6",
                "f 1 6 5",
                "f 2 3 7",
                "f 2 7 6",
                "f 3 4 8",
                "f 3 8 7",
                "f 4 1 5",
                "f 4 5 8",
            ]
        )
        + "\n"
    )
    return path


def model_package(root, orientation=None):
    asset = box_model(root / "11111111-1111-1111-1111-111111111111")
    metadata = {"entrypoint": "model.obj", "name": "Caja"}
    if orientation is not None:
        metadata["orientation"] = orientation
    info = asset.parent / ".gestur-model.json"
    info.write_text(json.dumps(metadata))
    return f"{asset.parent.name}/model.obj", asset, info


def test_legacy_models_have_no_fixed_rotation_and_orientation_is_bounded(tmp_path):
    model_id, _, info = model_package(tmp_path)
    assert load_model_orientation(model_id, tmp_path) == {"x": 0, "y": 0, "z": 0}
    assert load_model_orientation(None, tmp_path) == {"x": 0, "y": 0, "z": 0}
    orientation = {"x": 90, "y": 180, "z": 270}
    metadata = json.loads(info.read_text())
    metadata["orientation"] = orientation
    info.write_text(json.dumps(metadata))
    assert load_model_orientation(model_id, tmp_path) == orientation
    for invalid in (
        {"x": 0},
        {"x": -90, "y": 0, "z": 0},
        {"x": 90.0, "y": 0, "z": 0},
        {"x": True, "y": 0, "z": 0},
        {"x": 0, "y": 0, "z": 0, "w": 0},
    ):
        with pytest.raises(ConfigurationError):
            validate_model_orientation(invalid)


def test_orientation_metadata_cannot_escape_model_package(tmp_path):
    model_id, _, info = model_package(tmp_path / "models")
    external = tmp_path / "external.json"
    external.write_text('{"orientation":{"x":90,"y":0,"z":0}}')
    info.unlink()
    info.symlink_to(external)
    with pytest.raises(ConfigurationError):
        load_model_orientation(model_id, tmp_path / "models")


def test_real_panda_fixed_axes_refit_bounds_without_changing_gestures_or_geometry(
    tmp_path, monkeypatch
):
    pytest.importorskip("panda3d.core")
    from visualizer import ControlledObjViewer

    viewer = ControlledObjViewer(
        box_model(tmp_path), window_type="none", fullscreen=False
    )
    try:
        wrapper = viewer.model
        geoms = [
            (
                node.node().get_geom(i).get_vertex_data().get_num_rows(),
                node.node().get_geom(i).get_primitive(0).get_num_primitives(),
            )
            for node in wrapper.find_all_matches("**/+GeomNode")
            for i in range(node.node().get_num_geoms())
        ]
        assert sum(count for _, count in geoms) == 12
        monkeypatch.setattr(
            viewer.loader,
            "loadModel",
            lambda *a, **kw: pytest.fail("orientation reparsed model"),
        )
        viewer.update_model(rotation=[17, 23, 41], position=[2, 3, 4], scale=1.5)
        for orientation, expected in [
            ({"x": 90, "y": 0, "z": 0}, [4, 12, 8]),
            ({"x": 0, "y": 90, "z": 0}, [12, 8, 4]),
            ({"x": 0, "y": 0, "z": 90}, [8, 4, 12]),
            ({"x": 0, "y": 0, "z": 0}, [4, 8, 12]),
        ]:
            assert viewer.set_model_orientation(orientation)
            assert viewer.model == wrapper
            low, high = wrapper.get_tight_bounds(wrapper)
            assert list(high - low) == pytest.approx(expected, abs=1e-5)
            assert list((low + high) * 0.5) == pytest.approx([0, 0, 0], abs=1e-5)
            assert list(wrapper.get_hpr()) == pytest.approx([17, 23, 41])
            assert list(wrapper.get_pos()) == pytest.approx([2, 3, 4])
            assert list(wrapper.get_scale()) == pytest.approx([1.5] * 3)
            assert list(viewer.model_basis.get_hpr()) == pytest.approx(
                [orientation["z"], orientation["x"], orientation["y"]]
            )
        assert not viewer.set_model_orientation({"x": 0, "y": 0, "z": 0})
        assert viewer.current_state["rotation"] == [17, 23, 41]
    finally:
        viewer.destroy()


def test_live_orientation_and_reboot_restore_without_reloading_on_rename(
    tmp_path, monkeypatch
):
    pytest.importorskip("panda3d.core")
    import controller
    import visualizer

    models = tmp_path / "models"
    model_id, asset, info = model_package(models, {"x": 90, "y": 0, "z": 0})
    path = tmp_path / "config.json"
    config = default_config()
    config["active_model"] = model_id
    save_config(config, path)
    real_viewer = visualizer.ControlledObjViewer

    def headless(model_path, **kwargs):
        kwargs["fullscreen"] = False
        return real_viewer(model_path, window_type="none", **kwargs)

    monkeypatch.setattr(visualizer, "ControlledObjViewer", headless)
    app = controller.PoseController(config_path=path, models_dir=models, no_camera=True)
    orientation = {"x": 0, "y": 90, "z": 270}
    try:
        assert app.rendered_orientation == {"x": 90, "y": 0, "z": 0}
        assert list(app.visualizer.model_basis.get_hpr()) == [0, 90, 0]
        model = app.visualizer.model
        original = app.visualizer.set_model_orientation
        applied = []

        def apply(value):
            applied.append(value)
            return original(value)

        monkeypatch.setattr(app.visualizer, "set_model_orientation", apply)
        monkeypatch.setattr(
            app.visualizer,
            "load_model",
            lambda *a, **kw: pytest.fail("same model reloaded"),
        )

        def edit_metadata(**values):
            metadata = json.loads(info.read_text())
            metadata.update(values)
            info.write_text(json.dumps(metadata))
            stamp = models.stat().st_mtime_ns + 1_000_000
            os.utime(models, ns=(stamp, stamp))
            app._reload_config()

        edit_metadata(name="Caja renombrada")
        assert applied == []
        edit_metadata(orientation=orientation)
        assert applied == [orientation]
        assert app.visualizer.model == model
        assert app.rendered_orientation == orientation
        edit_metadata(name="Caja final")
        assert applied == [orientation]
        assert app.requested_model == app.rendered_model == model_id
        assert app.model_error is None
        app.status_path = tmp_path / "runtime.json"
        app._write_status()
        status = json.loads(app.status_path.read_text())
        assert status["rendered_orientation"] == orientation
        assert status["rendered_model"] == model_id
    finally:
        app.cleanup()
    restarted = controller.PoseController(
        config_path=path, models_dir=models, no_camera=True
    )
    try:
        assert restarted.rendered_orientation == orientation
        assert list(restarted.visualizer.model_basis.get_hpr()) == [270, 0, 90]
        assert restarted.rendered_model == model_id
    finally:
        restarted.cleanup()
