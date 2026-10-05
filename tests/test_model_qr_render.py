"""Check the QR as actually drawn, including resize and scene transitions."""

import pytest


@pytest.fixture
def rendered_viewer(tmp_path, monkeypatch):
    p = pytest.importorskip("panda3d.core")
    import visualizer

    monkeypatch.setattr(visualizer, "portal_url", lambda: "http://192.168.1.93")
    asset = tmp_path / "triangle.obj"
    asset.write_text("v -1 0 0\nv 1 0 0\nv 0 0 2\nf 1 2 3\n")
    viewer = visualizer.ControlledObjViewer(
        asset,
        window_type="offscreen",
        fullscreen=False,
        model_orientation={"x": 90, "y": 0, "z": 0},
    )
    clock = p.ClockObject.get_global_clock()
    clock.set_mode(p.ClockObject.M_non_real_time)
    clock.set_dt(1 / 60)
    try:
        yield viewer, asset
    finally:
        viewer.destroy()
        clock.set_mode(p.ClockObject.M_limited)
        clock.set_frame_rate(60)


def pixels(viewer):
    import numpy as np

    for _ in range(3):
        viewer.invalidate()
        viewer.taskMgr.step()
    texture = viewer.win.get_screenshot()
    return np.frombuffer(texture.get_ram_image_as("RGB"), dtype=np.uint8).reshape(
        texture.get_y_size(), texture.get_x_size(), 3
    )[::-1]


def without_author_credit(viewer, frame):
    """Exclude the persistent credit when checking model and QR pixels."""
    import math

    import numpy as np

    height, width = frame.shape[:2]
    low, high = viewer.author_credit.get_tight_bounds(viewer.render2d)
    assert -1 < low.x < high.x < 0 and -1 < low.z < high.z < 0
    left = max(0, math.floor((low.x + 1) * width / 2) - 2)
    right = min(width, math.ceil((high.x + 1) * width / 2) + 2)
    top = max(0, math.floor((1 - high.z) * height / 2) - 2)
    bottom = min(height, math.ceil((1 - low.z) * height / 2) + 2)
    assert np.any(frame[top:bottom, left:right]), "author credit remains visible"
    result = frame.copy()
    result[top:bottom, left:right] = 0
    return result


def test_zero_exposure_keeps_qr_fully_white_and_actual_frame_decodes(rendered_viewer):
    import cv2
    import numpy as np

    viewer, _ = rendered_viewer
    viewer.set_model_url("https://example.org/pieza/1")
    viewer.apply_settings(ambient_light="sunset", exposure=0)
    frame = pixels(viewer)
    assert np.array_equal(
        np.unique(without_author_credit(viewer, frame).reshape(-1, 3), axis=0),
        [[0, 0, 0], [255, 255, 255]],
    )
    # OpenCV expects dark modules; invert the framebuffer rather than decoding
    # the original texture, so geometry size/filtering/placement are exercised.
    assert (
        cv2.QRCodeDetector().detectAndDecode(255 - frame[..., 0])[0]
        == "https://example.org/pieza/1"
    )


def test_qr_holes_preserve_background_and_resize_stays_inside_viewport(rendered_viewer):
    import numpy as np

    viewer, _ = rendered_viewer
    viewer.set_model_url("https://example.org/pieza/1")
    viewer.apply_settings(exposure=0)
    viewer.setBackgroundColor(0.1, 0.2, 0.3, 1)
    for width, height in ((1920, 1080), (540, 960), (640, 480)):
        viewer.win.set_size(width, height)
        # GraphicsBuffer has no native window event; simulate ShowBase's normal
        # aspect update while its render task detects and lays out the new size.
        viewer.adjustWindowAspectRatio(width / height)
        frame = pixels(viewer)
        low, high = viewer.model_qr.get_tight_bounds(viewer.render2d)
        assert 0 < low.x < high.x < 1
        assert -1 < low.z < high.z < 0
        assert frame.shape == (height, width, 3)
        left, right = int((low.x + 1) * width / 2), int((high.x + 1) * width / 2)
        top, bottom = int((1 - high.z) * height / 2), int((1 - low.z) * height / 2)
        drawn = frame[top:bottom, left:right]
        colors = np.unique(drawn.reshape(-1, 3), axis=0)
        assert any(np.array_equal(color, [255, 255, 255]) for color in colors)
        assert any(
            np.all(np.abs(color.astype(int) - [26, 51, 77]) <= 1) for color in colors
        )
        assert not any(np.array_equal(color, [0, 0, 0]) for color in colors), (
            "no opaque black backing"
        )


def test_no_url_after_welcome_leaves_neither_qr_nor_welcome_geometry(rendered_viewer):
    import numpy as np

    viewer, asset = rendered_viewer
    viewer.set_model_url("https://example.org/previous")
    viewer.show_welcome()
    assert viewer.model_qr is None
    viewer.load_model(asset, orientation={"x": 90, "y": 0, "z": 0})
    viewer.apply_settings(exposure=0)
    assert viewer.model_url is None and viewer.model_qr is None
    assert viewer.welcome is None and viewer.welcome_overlay is None
    assert not np.any(without_author_credit(viewer, pixels(viewer)))
