from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest

from runtime_state import FrameMetrics, LatestPose


def test_mailbox_discards_backlog_and_never_exposes_mutable_worker_data():
    clock = [10.0]
    slot = LatestPose(clock=lambda: clock[0])
    incoming = {"head": {"detected": True, "x": 0.8}}
    slot.publish(incoming)
    incoming["head"]["x"] = 0.1
    assert slot.read()["head"]["x"] == 0.8
    read = slot.read()
    read["head"]["x"] = 0.2
    for index in range(100):
        slot.publish({"head": {"detected": True, "x": index / 100}})
    assert slot.read()["head"]["x"] == .99
    clock[0] += .36
    assert slot.read() == {}


def test_frame_metrics_reports_measured_intervals_with_bounded_memory():
    metrics = FrameMetrics(max_samples=3)
    for now in (0, .02, .04, .06, .08):
        metrics.tick(now)
    result = metrics.summary()
    assert result["render_fps"] == 50
    assert result["frame_ms_p95"] == 20
    assert result["samples"] == 3


def test_callback_cannot_touch_renderer_and_stale_input_reaches_controls(monkeypatch):
    import controller
    app = controller.PoseController.__new__(controller.PoseController)
    app.mailbox = LatestPose()
    app._on_pose_update({"head": {"detected": True, "x": .3}})
    assert app.mailbox.read()["head"]["x"] == .3
    # There is deliberately no visualizer on this instance.


def test_model_resolution_rejects_escape_and_symlinks(tmp_path):
    from controller import resolve_model
    assert resolve_model(None, tmp_path) is None
    models = tmp_path / "models"
    models.mkdir()
    external = tmp_path / "private.obj"
    external.write_text("private")
    (models / "link.obj").symlink_to(external)
    for candidate in ("../private.obj", "link.obj", str(external)):
        with pytest.raises(ValueError):
            resolve_model(candidate, models)
    (models / "package").mkdir()
    asset = models / "package" / "model.obj"
    asset.write_text("v 0 0 0")
    assert resolve_model("package/model.obj", models) == asset


def test_config_reload_preserves_scene_after_invalid_update(tmp_path, monkeypatch):
    import controller
    from runtime_config import default_config, save_config
    config = default_config()
    config_path = tmp_path / "config.json"
    save_config(config, config_path)
    app = controller.PoseController.__new__(controller.PoseController)
    app.config_path = config_path
    app.overrides = {}
    app.config = config
    app._config_stamp = app._stamp()
    app.obj_override = None
    app.models_dir = tmp_path / "models"
    app.control_system = object()
    app.visualizer = SimpleNamespace(
        load_model=lambda path: pytest.fail("invalid config must not change model"),
        apply_settings=lambda **kwargs: pytest.fail("invalid config must not change settings"),
    )
    config_path.write_text('{"schema_version":100}')
    app._reload_config()
    assert app.config == config


def write_obj_fixture(root):
    # A small textured triangle proves the real OBJ/MTL path, independent of
    # any shipped artwork or binary asset.
    from PIL import Image
    Image.new("RGB", (2, 2), (100, 200, 150)).save(root / "surface.png")
    (root / "model.mtl").write_text("newmtl material\nKd 1 1 1\nmap_Kd surface.png\n")
    path = root / "model.obj"
    path.write_text("mtllib model.mtl\nv -1 0 0\nv 1 0 0\nv 0 0 1\n"
                    "vt 0 0\nvt 1 0\nvt 0.5 1\nusemtl material\nf 1/1 2/2 3/3\n")
    return path


def test_real_viewer_empty_model_empty_transitions_preserve_imported_texture(tmp_path, monkeypatch):
    pytest.importorskip("panda3d.core")
    from visualizer import ControlledObjViewer
    monkeypatch.setenv("GESTUR_PORTAL_URL", "http://10.42.0.1:3000")
    path = write_obj_fixture(tmp_path)
    viewer = ControlledObjViewer(None, window_type="none", fullscreen=False)
    try:
        assert viewer.model is None
        assert viewer.model_path is None
        assert viewer.welcome is not None
        assert viewer.welcome_url == "http://10.42.0.1:3000"
        welcome = viewer.welcome
        overlay = viewer.welcome_overlay
        viewer.load_model(path)
        assert welcome.is_empty() and overlay.is_empty()
        assert viewer.welcome is None and viewer.welcome_overlay is None
        assert viewer.model_path == str(path)
        textures = viewer.model.find_all_textures()
        assert [(t.get_x_size(), t.get_y_size()) for t in textures] == [(2, 2)]
        old = viewer.model
        with pytest.raises(FileNotFoundError):
            viewer.load_model(tmp_path / "missing.obj")
        assert viewer.model == old
        viewer.update_model(rotation=[25, -5, 7], scale=1.75)
        assert list(viewer.model.get_hpr()) == [25, -5, 7]
        assert list(viewer.model.get_scale()) == [1.75] * 3
        viewer.load_model(None)
        assert old.is_empty()
        assert viewer.model is None and viewer.model_path is None
        assert viewer.welcome is not None and viewer.welcome_overlay is not None
        nodes = viewer.welcome.find_all_matches("**/+GeomNode")
        triangles = sum(n.node().get_geom(i).get_primitive(j).get_num_primitives()
                        for n in nodes for i in range(n.node().get_num_geoms())
                        for j in range(n.node().get_geom(i).get_num_primitives()))
        assert triangles == 1536
        # Idle frames do not move the QR or rebuild the scene.
        before = viewer.welcome_overlay
        viewer.taskMgr.step()
        viewer._refresh_welcome_overlay()
        assert viewer.welcome_overlay == before
    finally:
        viewer.destroy()


def tracking_controller():
    from controller import PoseController
    app = PoseController.__new__(PoseController)
    app.no_camera = False
    app.rendered_model = None
    app.pose_tracker = None
    app.mailbox = LatestPose()
    app.model_error = None
    app.tracking_error = None
    app.config_error = None
    app._next_camera_attempt = 0
    return app


def test_no_model_never_initializes_camera_or_tracking(monkeypatch):
    app = tracking_controller()
    app._init_pose_tracker = lambda: pytest.fail("empty viewer must not use camera")
    app._sync_tracking(100)
    assert app.pose_tracker is None
    assert app.last_error is None


def test_camera_failure_keeps_viewer_available_and_retries_with_backoff():
    app = tracking_controller()
    app.rendered_model = "package/model.glb"
    attempts = []
    def fail_camera():
        attempts.append(True)
        raise RuntimeError("camera disconnected")
    app._init_pose_tracker = fail_camera
    app._sync_tracking(100)
    assert app.last_error == "No se pudo iniciar el seguimiento: camera disconnected"
    app._sync_tracking(101)
    assert len(attempts) == 1
    app._sync_tracking(115)
    assert len(attempts) == 2
    app.rendered_model = None
    app._sync_tracking(116)
    assert app.last_error is None


def test_unavailable_selected_model_shows_welcome_and_honest_status(tmp_path, monkeypatch):
    import controller
    from runtime_config import default_config, save_config
    config = default_config()
    config['active_model'] = 'package/missing.obj'
    path = tmp_path / 'config.json'
    save_config(config, path)
    loaded = []
    class Viewer:
        def __init__(self, path, **kwargs):
            loaded.append(path)
            self.taskMgr = SimpleNamespace(add=lambda *args, **kwargs: None)
        def accept(self, *args):
            pass
    monkeypatch.setitem(sys.modules, 'visualizer', SimpleNamespace(ControlledObjViewer=Viewer))
    app = controller.PoseController(config_path=path, models_dir=tmp_path / 'models', no_camera=True)
    assert loaded == [None]
    assert app.rendered_model is None
    assert app.requested_model == 'package/missing.obj'
    assert app.last_error


def test_live_selection_missing_model_and_clear_return_to_welcome(tmp_path, monkeypatch):
    import controller
    from runtime_config import default_config, save_config
    config_path = tmp_path / "config.json"
    config = default_config()
    save_config(config, config_path)
    models = tmp_path / "models"
    (models / "package").mkdir(parents=True)
    asset = models / "package" / "model.obj"
    asset.write_text("v 0 0 0")
    loaded = []
    class Viewer:
        def __init__(self, path, **kwargs):
            loaded.append(path)
            self.taskMgr = SimpleNamespace(add=lambda *args, **kwargs: None)
        def accept(self, *args):
            pass
        def load_model(self, path):
            loaded.append(path)
        def apply_settings(self, **kwargs):
            pass
    monkeypatch.setitem(sys.modules, "visualizer", SimpleNamespace(ControlledObjViewer=Viewer))
    app = controller.PoseController(config_path=config_path, models_dir=models, no_camera=True)
    config["active_model"] = "package/model.obj"
    save_config(config, config_path)
    app._reload_config()
    assert loaded == [None, asset]
    assert app.rendered_model == "package/model.obj"
    assert app.last_error is None
    config["active_model"] = "package/missing.glb"
    save_config(config, config_path)
    app._reload_config()
    assert loaded[-1] is None
    assert app.rendered_model is None
    assert app.requested_model == "package/missing.glb"
    assert "el visor está vacío" in app.last_error
    config["active_model"] = None
    save_config(config, config_path)
    app._reload_config()
    assert app.rendered_model is None and app.requested_model is None
    assert app.last_error is None


def test_empty_scene_stops_camera_and_preserves_invalid_config_error():
    app = tracking_controller()
    stopped = []
    app.pose_tracker = SimpleNamespace(stop=lambda: stopped.append(True))
    app.mailbox.publish({"head": {"detected": True}})
    app.config_error = "Invalid JSON"
    app._sync_tracking(100)
    assert stopped == [True]
    assert app.pose_tracker is None and app.mailbox.read() == {}
    assert app.last_error == "Invalid JSON"


def test_config_restart_exits_without_starting_camera_during_shutdown():
    from controller import PoseController, RESTART_REQUESTED
    from control_system import create_default_control_system
    app = PoseController.__new__(PoseController)
    app.metrics = FrameMetrics()
    app.mailbox = LatestPose()
    app.control_system = create_default_control_system()
    app.visualizer = SimpleNamespace(update_model=lambda **kwargs: None)
    app._last_config_check = 0
    app._running = True
    app.verbose = False
    app.benchmark_seconds = 0
    app._reload_config = lambda: setattr(app, "exit_code", RESTART_REQUESTED)
    app._sync_tracking = lambda now: pytest.fail("a pending restart must not initialize camera")
    app._write_status = lambda: None
    assert app._render_tick(SimpleNamespace(cont="cont")) == "cont"
    assert app.exit_code == 42


def test_portal_url_prefers_access_point_over_lan(monkeypatch):
    import visualizer
    monkeypatch.delenv("GESTUR_PORTAL_URL", raising=False)
    monkeypatch.setattr(visualizer.socket, "if_nameindex", lambda: [(1, "eth0"), (2, "wlan0")])
    monkeypatch.setattr(visualizer, "_interface_ipv4", lambda name: {"eth0": "192.168.1.8", "wlan0": "10.42.0.1"}.get(name))
    assert visualizer.portal_url() == "http://10.42.0.1:3000"
    monkeypatch.setattr(visualizer, "_interface_ipv4", lambda name: {"eth0": "192.168.1.8"}.get(name))
    assert visualizer.portal_url() == "http://192.168.1.8:3000"


def test_portal_url_honors_valid_override_and_ignores_loopback_or_tokens(monkeypatch):
    import visualizer
    monkeypatch.setattr(visualizer.socket, "if_nameindex", lambda: [])
    monkeypatch.setattr(visualizer, "_interface_ipv4", lambda name: "10.42.0.1")
    monkeypatch.setenv("GESTUR_PORTAL_URL", "https://exhibition.local/gestur/")
    assert visualizer.portal_url() == "https://exhibition.local/gestur"
    for value in ("http://localhost:3000", "http://127.0.0.1:3000", "http://[::1]:3000", "file:///etc/passwd",
                  "http://10.42.0.1:3000/?token=secret", "http://admin:secret@10.42.0.1:3000", "http://0.0.0.0:3000"):
        monkeypatch.setenv("GESTUR_PORTAL_URL", value)
        assert visualizer.portal_url() == "http://10.42.0.1:3000"


def test_qr_contains_reachable_url_without_admin_token(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from visualizer import ControlledObjViewer
    monkeypatch.setenv("GESTUR_PORTAL_URL", "http://10.42.0.1:3000")
    viewer = ControlledObjViewer(None, window_type="none", fullscreen=False)
    try:
        texture = viewer.welcome_overlay.find("**/welcome-qr").get_texture()
        raw = np.frombuffer(texture.get_ram_image(), dtype=np.uint8)
        image = raw.reshape(texture.get_y_size(), texture.get_x_size())[::-1]
        image = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
        assert decoded == "http://10.42.0.1:3000"
    finally:
        viewer.destroy()
