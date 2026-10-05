"""The welcome screen has a readable IP QR without an opaque background."""

import pytest


@pytest.fixture
def welcome(monkeypatch):
    pytest.importorskip("panda3d.core")
    from gestur import visualizer

    monkeypatch.setattr(visualizer, "portal_url", lambda: "http://192.168.1.93")
    viewer = visualizer.ControlledObjViewer(None, window_type="none", fullscreen=False)
    try:
        yield viewer
    finally:
        viewer.destroy()


def test_white_qr_alpha_is_decodable_and_has_a_transparent_quiet_zone(welcome):
    import cv2
    import numpy as np
    from panda3d.core import TransparencyAttrib

    card = welcome.welcome_overlay.find("**/welcome-qr")
    texture = card.get_texture()
    pixels = np.frombuffer(texture.get_ram_image(), dtype=np.uint8).reshape(
        texture.get_y_size(), texture.get_x_size(), 4
    )[::-1]
    assert np.all(pixels[..., :3] == 255)
    assert set(np.unique(pixels[..., 3])) == {0, 255}
    assert np.all(pixels[:4, :, 3] == 0)
    assert np.all(pixels[-4:, :, 3] == 0)
    assert np.all(pixels[:, :4, 3] == 0)
    assert np.all(pixels[:, -4:, 3] == 0)
    assert card.get_transparency() == TransparencyAttrib.M_alpha
    # OpenCV expects dark modules: the stored alpha is the exact module mask.
    image = cv2.resize(
        255 - pixels[..., 3], None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST
    )
    assert cv2.QRCodeDetector().detectAndDecode(image)[0] == "http://192.168.1.93"


def test_sculpture_is_neutral_white_with_fifteen_percent_alpha(welcome):
    from panda3d.core import GeomVertexReader, TransparencyAttrib

    ring = welcome.welcome_ring
    rgba = ring.get_color_scale()
    assert list(rgba)[:3] == [1, 1, 1]
    # Panda quantizes render-state color components to 1/1024.
    assert rgba[3] == pytest.approx(0.15, abs=1 / 1024)
    assert ring.get_transparency() == TransparencyAttrib.M_alpha
    data = ring.node().get_geom(0).get_vertex_data()
    reader = GeomVertexReader(data, "color")
    while not reader.is_at_end():
        assert list(reader.get_data4()) == [1, 1, 1, 1]


def test_overlay_order_and_requested_copy(welcome):
    overlay = welcome.welcome_overlay
    brand = overlay.find("**/welcome-brand")
    qr = overlay.find("**/welcome-qr")
    prompt = overlay.find("**/welcome-prompt")
    url = overlay.find("**/welcome-url")
    assert brand.node().get_text() == "GESTUR"
    assert prompt.node().get_text() == "Escanea el QR para comenzar"
    assert url.node().get_text() == "o accede a http://192.168.1.93"
    assert brand.get_z() > qr.get_z() > prompt.get_z() > url.get_z()
    assert brand.get_sx() > prompt.get_sx() > url.get_sx()
    assert list(brand.node().get_text_color()) == [1, 1, 1, 1]


def test_no_network_waits_without_false_qr_and_recovers(welcome, monkeypatch):
    from gestur import visualizer

    monkeypatch.setattr(visualizer, "portal_url", lambda: None)
    welcome._refresh_welcome_overlay()
    assert welcome.welcome_url is None
    overlay = welcome.welcome_overlay
    assert overlay.find("**/welcome-qr").is_empty()
    assert (
        overlay.find("**/welcome-network").node().get_text()
        == "Esperando conexión de red"
    )
    welcome._refresh_welcome_overlay()
    assert welcome.welcome_overlay == overlay
    monkeypatch.setattr(visualizer, "portal_url", lambda: "http://10.42.0.1")
    welcome._refresh_welcome_overlay()
    assert not welcome.welcome_overlay.find("**/welcome-qr").is_empty()
    assert welcome.welcome_url == "http://10.42.0.1"


def test_failed_existing_model_never_shows_onboarding(welcome):
    welcome.show_model_error("El archivo no está disponible")
    assert welcome.welcome is None
    assert welcome.welcome_overlay is None
    assert welcome.welcome_url is None
    assert welcome.model_error_overlay is not None
    assert welcome.aspect2d.find("**/welcome-qr").is_empty()
    welcome.show_welcome()
    assert welcome.model_error_overlay is None
    assert welcome.welcome is not None
