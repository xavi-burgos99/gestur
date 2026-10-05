import pytest

from runtime_config import ConfigurationError, validate_model_content


def test_content_validation_rejects_control_sequences_and_unsupported_positions():
    for value in (
        {"title": "x"},
        {"title": "x", "description": "bad\x01", "placement": "top"},
        {"title": "", "description": "", "placement": "side"},
    ):
        with pytest.raises(ConfigurationError):
            validate_model_content(value)


def test_content_overlay_is_optional_and_survives_resize_without_reloading_model(
    tmp_path,
):
    pytest.importorskip("panda3d.core")
    from visualizer import ControlledObjViewer

    asset = tmp_path / "triangle.obj"
    asset.write_text("v -1 0 0\nv 1 0 0\nv 0 0 2\nf 1 2 3\n")
    viewer = ControlledObjViewer(asset, window_type="offscreen", fullscreen=False)
    try:
        model = viewer.model
        assert viewer.content_overlay is None
        for placement in ("top", "bottom"):
            viewer.set_model_content(
                {
                    "title": "Mi pieza",
                    "description": "Descripción " * 60,
                    "placement": placement,
                }
            )
            for ratio in (16 / 9, 9 / 16):
                viewer.win.set_size(960, round(960 / ratio))
                viewer.adjustWindowAspectRatio(ratio)
                viewer.taskMgr.step()
                viewer._layout_model_content()
                viewer.graphicsEngine.render_frame()
                assert viewer.model == model
                low, high = viewer.content_overlay.get_tight_bounds(viewer.render2d)
                assert -1.01 <= low.x <= high.x <= 1.01
                assert -1.01 <= low.z <= high.z <= 1.01
        viewer.set_model_content(
            {"title": "Mi pieza", "description": "Descripción", "placement": "bottom"}
        )
        viewer.set_model_url("https://example.org")
        viewer._layout_model_content()
        _, qr_high = viewer.model_qr.get_tight_bounds(viewer.render2d)
        text_low, _ = viewer.content_description.get_tight_bounds(viewer.render2d)
        assert (text_low.z - qr_high.z) * viewer.win.get_y_size() / 2 >= 19.9
        raised = viewer.content_description.get_z()
        viewer.set_model_url(None)
        assert viewer.content_description.get_z() < raised
        viewer.set_model_content(None)
        assert viewer.content_overlay is None
        viewer.show_welcome()
        assert viewer.content_overlay is None
    finally:
        viewer.destroy()
