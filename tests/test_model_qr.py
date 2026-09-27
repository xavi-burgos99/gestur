"""Per-model links are static white overlays, independent of lighting and poses."""

import json
import os

import pytest

from runtime_config import default_config, save_config


@pytest.fixture
def viewer(tmp_path):
    pytest.importorskip("panda3d.core")
    from visualizer import ControlledObjViewer

    asset = tmp_path / "model.obj"
    asset.write_text("v -1 0 0\nv 1 0 0\nv 0 0 2\nf 1 2 3\n")
    app = ControlledObjViewer(asset, window_type="none", fullscreen=False)
    try:
        yield app
    finally:
        app.destroy()


def decode(card):
    import cv2
    import numpy as np

    texture = card.get_texture()
    pixels = np.frombuffer(texture.get_ram_image(), dtype=np.uint8).reshape(
        texture.get_y_size(), texture.get_x_size(), 4
    )[::-1]
    assert np.all(pixels[:, :, :3] == 255)
    assert set(np.unique(pixels[:, :, 3])) == {0, 255}
    assert not pixels[:4, :, 3].any() and not pixels[:, :4, 3].any()
    assert not pixels[-4:, :, 3].any() and not pixels[:, -4:, 3].any()
    image = cv2.resize(
        255 - pixels[:, :, 3], None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST
    )
    return cv2.QRCodeDetector().detectAndDecode(image)[0]


def test_optional_link_is_white_transparent_small_and_anchored_with_gap(viewer):
    from panda3d.core import TransparencyAttrib

    assert viewer.model_qr is None
    model = viewer.model
    assert viewer.set_model_url("https://example.org/pieza/1")
    card = viewer.model_qr
    assert card.get_parent() == viewer.a2dBottomRight
    assert card.get_x() == pytest.approx(-0.06)
    assert card.get_z() == pytest.approx(0.06)
    assert 0.32 - 1e-6 <= card.get_sx() < 0.72
    assert card.get_transparency() == TransparencyAttrib.M_alpha
    assert not card.get_depth_test() and not card.get_depth_write()
    assert decode(card) == "https://example.org/pieza/1"
    assert viewer.model == model
    assert not viewer.set_model_url("https://example.org/pieza/1")
    assert viewer.model_qr == card


def test_url_edit_remove_and_lighting_never_reload_the_model_or_tint_qr(viewer):
    viewer.set_model_url("https://example.org/first")
    model = viewer.model
    viewer.apply_settings(ambient_light="sunset", exposure=0)
    assert decode(viewer.model_qr) == "https://example.org/first"
    viewer.set_model_url("https://example.org/second")
    assert decode(viewer.model_qr) == "https://example.org/second"
    assert viewer.model == model
    viewer.set_model_url("")
    assert viewer.model_qr is None and viewer.model_url is None
    assert viewer.model == model


def test_failed_replacement_keeps_old_qr_and_welcome_removes_it(viewer, tmp_path):
    viewer.set_model_url("https://example.org/current")
    with pytest.raises((ValueError, OSError)):
        viewer.load_model(
            tmp_path / "missing.obj", model_url="https://example.org/other"
        )
    assert decode(viewer.model_qr) == "https://example.org/current"
    viewer.show_welcome()
    assert viewer.model_qr is None and viewer.model_url is None
    assert viewer.welcome_overlay is not None


def test_controller_hot_reload_and_restart_restore_link_without_reloading_mesh(
    tmp_path, monkeypatch
):
    pytest.importorskip("panda3d.core")
    import controller
    import visualizer

    models = tmp_path / "models"
    package = models / "11111111-1111-1111-1111-111111111111"
    package.mkdir(parents=True)
    asset = package / "model.obj"
    asset.write_text("v -1 0 0\nv 1 0 0\nv 0 0 2\nf 1 2 3\n")
    info = package / ".gestur-model.json"
    info.write_text(
        json.dumps(
            {
                "entrypoint": "model.obj",
                "name": "Test",
                "url": "https://example.org/one",
            }
        )
    )
    config = default_config()
    config["active_model"] = f"{package.name}/model.obj"
    config_path = tmp_path / "config.json"
    save_config(config, config_path)
    real = visualizer.ControlledObjViewer

    def headless(model_path, **kwargs):
        kwargs["fullscreen"] = False
        return real(model_path, window_type="none", **kwargs)

    monkeypatch.setattr(visualizer, "ControlledObjViewer", headless)
    app = controller.PoseController(
        config_path=config_path, models_dir=models, no_camera=True
    )
    try:
        assert app.rendered_url == "https://example.org/one"
        model = app.visualizer.model
        monkeypatch.setattr(
            app.visualizer,
            "load_model",
            lambda *a, **kw: pytest.fail("geometry reloaded"),
        )
        for url in ["https://example.org/two", None, "https://example.org/three"]:
            metadata = json.loads(info.read_text())
            metadata["url"] = url
            info.write_text(json.dumps(metadata))
            stamp = models.stat().st_mtime_ns + 1_000_000
            os.utime(models, ns=(stamp, stamp))
            app._reload_config()
            assert app.rendered_url == url
            assert app.visualizer.model == model
            assert app.visualizer.model_url == url
    finally:
        app.cleanup()
    restarted = controller.PoseController(
        config_path=config_path, models_dir=models, no_camera=True
    )
    try:
        assert restarted.rendered_url == "https://example.org/three"
        assert decode(restarted.visualizer.model_qr) == restarted.rendered_url
    finally:
        restarted.cleanup()
