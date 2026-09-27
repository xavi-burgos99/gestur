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
    from runtime_config import default_config
    from tracking_session import TrackingSession
    app = PoseController.__new__(PoseController)
    app.config = default_config()
    app.no_camera = False
    app.rendered_model = None
    app.mailbox = LatestPose()
    app.tracking_session = TrackingSession(app.mailbox.publish,
        factory=lambda settings: pytest.fail("empty viewer must not use camera"))
    app.model_error = None
    app.tracking_error = None
    app.config_error = None
    return app


def test_no_model_never_initializes_camera_or_tracking(monkeypatch):
    app = tracking_controller()
    app._sync_tracking(100)
    assert app.tracking_session.snapshot()['state'] == 'stopped'
    assert app.last_error is None


def test_controller_reports_async_camera_error_and_clears_it_when_model_removed():
    app = tracking_controller()
    app.rendered_model = "package/model.glb"
    errors = ['No se pudo iniciar el seguimiento: camera disconnected']
    app.tracking_session = SimpleNamespace(
        request=lambda settings: errors.__setitem__(0, errors[0] if settings else None),
        snapshot=lambda: {'error': errors[0]})
    app._sync_tracking(100)
    assert app.last_error == "No se pudo iniciar el seguimiento: camera disconnected"
    app.rendered_model = None
    app._sync_tracking(116)
    assert app.last_error is None


def package_model(root, name="11111111-1111-1111-1111-111111111111"):
    import json
    package = root / name
    package.mkdir(parents=True)
    model = package / "model.obj"
    model.write_text("v 0 0 0")
    (package / ".gestur-model.json").write_text(json.dumps({"entrypoint": "model.obj"}))
    return f"{name}/model.obj", model


def viewer_stub(monkeypatch, fail=None):
    loaded, errors = [], []
    class Viewer:
        def __init__(self, path, **kwargs):
            self.taskMgr = SimpleNamespace(add=lambda *args, **kwargs: None)
            self.load_model(path)
        def accept(self, *args):
            pass
        def load_model(self, path, **kwargs):
            if path and fail and fail(path):
                raise ValueError("Geometría no válida")
            loaded.append(path)
        def show_model_error(self, error):
            errors.append(error)
        def apply_settings(self, **kwargs):
            pass
    monkeypatch.setitem(sys.modules, "visualizer", SimpleNamespace(ControlledObjViewer=Viewer))
    return loaded, errors


def test_missing_selection_with_empty_library_shows_welcome(tmp_path, monkeypatch):
    import controller
    from runtime_config import default_config, save_config
    config = default_config()
    config["active_model"] = "package/missing.obj"
    path = tmp_path / "config.json"
    save_config(config, path)
    loaded, errors = viewer_stub(monkeypatch)
    app = controller.PoseController(config_path=path, models_dir=tmp_path / "models", no_camera=True)
    assert loaded == [None]
    assert app.rendered_model is None and app.requested_model is None
    assert app.last_error is None and errors == []


def test_bad_geometry_at_start_reports_error_instead_of_qr(tmp_path, monkeypatch):
    import controller
    from runtime_config import default_config, save_config
    config_path = tmp_path / "config.json"
    save_config(default_config(), config_path)
    models = tmp_path / "models"
    model_id, asset = package_model(models)
    loaded, errors = viewer_stub(monkeypatch, fail=lambda path: True)
    app = controller.PoseController(config_path=config_path, models_dir=models, no_camera=True)
    assert loaded == [None]
    assert app.requested_model == model_id and app.rendered_model is None
    assert errors == [app.last_error]
    assert "Geometría no válida" in app.last_error



def test_real_panda_failed_initial_model_releases_showbase_and_boots_error_screen(tmp_path, monkeypatch):
    pytest.importorskip("panda3d.core")
    import builtins
    import controller
    import visualizer
    from runtime_config import default_config, save_config
    config_path = tmp_path / "config.json"
    models = tmp_path / "models"
    model_id, asset = package_model(models)  # OBJ exists, but has no faces.
    config = default_config()
    config["active_model"] = model_id
    save_config(config, config_path)
    monkeypatch.setenv("GESTUR_PORTAL_URL", "http://192.168.1.8")
    real_viewer = visualizer.ControlledObjViewer
    attempts = []

    def headless_viewer(model_path, **kwargs):
        attempts.append(model_path)
        kwargs["fullscreen"] = False
        return real_viewer(model_path, window_type="none", **kwargs)

    monkeypatch.setattr(visualizer, "ControlledObjViewer", headless_viewer)
    app = controller.PoseController(config_path=config_path, models_dir=models, no_camera=True)
    try:
        assert attempts == [asset, None]
        # This is the real ShowBase singleton: the rejected first viewer must
        # release it before the fallback window can be created.
        assert builtins.base is app.visualizer
        assert app.requested_model == model_id and app.rendered_model is None
        assert app.model_error and app.last_error == app.model_error
        assert app.visualizer.model_error_overlay is not None
        assert app.visualizer.welcome is None
        assert app.visualizer.welcome_overlay is None
        app.visualizer.taskMgr.step()
    finally:
        app.cleanup()

def test_live_selection_failure_keeps_model_and_empty_library_alone_shows_qr(tmp_path, monkeypatch):
    import controller
    from runtime_config import default_config, save_config
    config_path = tmp_path / "config.json"
    config = default_config()
    save_config(config, config_path)
    models = tmp_path / "models"
    loaded, errors = viewer_stub(monkeypatch, fail=lambda path: path.parent.name.startswith("2"))
    app = controller.PoseController(config_path=config_path, models_dir=models, no_camera=True)
    first, asset = package_model(models)
    # Catalog publication is enough to activate the first model, even if the
    # viewer boots before the portal reconciles its persisted configuration.
    app._reload_config()
    assert loaded == [None, asset] and app.rendered_model == first
    second, broken = package_model(models, "22222222-2222-2222-2222-222222222222")
    config["active_model"] = second
    save_config(config, config_path)
    app._reload_config()
    assert loaded == [None, asset]
    assert app.rendered_model == first and app.requested_model == second
    assert "Geometría no válida" in app.last_error
    assert errors == []
    # A stale/manual null cannot blank the last successfully loaded exhibition.
    config["active_model"] = None
    save_config(config, config_path)
    app._reload_config()
    assert app.rendered_model == first
    assert app.last_error is None
    import shutil
    shutil.rmtree(asset.parent)
    shutil.rmtree(broken.parent)
    app._reload_config()
    assert loaded[-1] is None
    assert app.rendered_model is None and app.requested_model is None
    assert app.last_error is None


def test_restart_loads_last_selection_and_never_overwrites_shared_config(tmp_path, monkeypatch):
    import controller
    from runtime_config import default_config, save_config
    config_path = tmp_path / "config.json"
    models = tmp_path / "models"
    first, _ = package_model(models)
    last, asset = package_model(models, "22222222-2222-2222-2222-222222222222")
    config = default_config()
    config["active_model"] = last
    save_config(config, config_path)
    before = config_path.read_bytes()
    loaded, _ = viewer_stub(monkeypatch)
    for _ in range(2):
        app = controller.PoseController(config_path=config_path, models_dir=models, no_camera=True)
        assert app.rendered_model == last
    assert loaded == [asset, asset]
    assert config_path.read_bytes() == before
    # Unchanged config/catalog checks must never rescan model metadata.
    with monkeypatch.context() as unchanged:
        unchanged.setattr(controller, "reconcile_model_selection", lambda *args, **kwargs:
                          pytest.fail("unchanged catalog must not be scanned"))
        for _ in range(5):
            app._reload_config()
    # Missing saved model deterministically resolves the other complete import.
    asset.unlink()
    app = controller.PoseController(config_path=config_path, models_dir=models, no_camera=True)
    assert app.rendered_model == first
    assert config_path.read_bytes() == before


def test_empty_scene_stops_camera_and_preserves_invalid_config_error():
    app = tracking_controller()
    requests = []
    def request(settings):
        requests.append(settings)
        app.mailbox.publish({})
    app.tracking_session = SimpleNamespace(request=request, snapshot=lambda: {'error': None})
    app.mailbox.publish({"head": {"detected": True}})
    app.config_error = "Invalid JSON"
    app._sync_tracking(100)
    assert requests == [None]
    assert app.mailbox.read() == {}
    assert app.last_error == "Invalid JSON"


def test_config_restart_exits_without_starting_camera_during_shutdown():
    from controller import PoseController, RESTART_REQUESTED
    from control_system import create_default_control_system
    app = PoseController.__new__(PoseController)
    app.metrics = FrameMetrics()
    app.device_metrics = SimpleNamespace(sample=lambda: {})
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


def test_cleanup_does_not_recreate_window_in_process_with_unreleased_camera():
    from controller import PoseController, RESTART_REQUESTED
    from runtime_config import default_config
    app = PoseController.__new__(PoseController)
    app._cleaned = False
    app.exit_code = RESTART_REQUESTED
    app.tracking_session = SimpleNamespace(close=lambda: False, snapshot=lambda: {'state': 'stopping'})
    app.visualizer = SimpleNamespace(render_metrics=FrameMetrics(), model_path=None,
        get_render_status=lambda: {}, destroy=lambda: None)
    app.metrics = FrameMetrics()
    app._hardware = {}
    app.config = default_config()
    app.no_camera = False
    app.metrics_path = None
    app.benchmark_seconds = 0
    app.cleanup()
    assert app.exit_code == 1  # Kiosk must restart the process, releasing driver state.


def test_portal_url_uses_access_point_then_lan_when_route_is_unavailable(monkeypatch):
    import visualizer
    from unittest.mock import MagicMock
    route = MagicMock()
    route.__enter__.return_value = route
    route.connect.side_effect = OSError("No route")
    monkeypatch.setattr(visualizer.socket, "socket", lambda *args: route)
    monkeypatch.delenv("GESTUR_PORTAL_URL", raising=False)
    monkeypatch.setattr(visualizer.socket, "if_nameindex", lambda: [(1, "eth0"), (2, "wlan0")])
    monkeypatch.setattr(visualizer, "_interface_ipv4", lambda name: {"eth0": "192.168.1.8", "wlan0": "10.42.0.1"}.get(name))
    assert visualizer.portal_url() == "http://10.42.0.1"
    monkeypatch.setattr(visualizer, "_interface_ipv4", lambda name: {"eth0": "192.168.1.8"}.get(name))
    assert visualizer.portal_url() == "http://192.168.1.8"


def test_portal_url_honors_valid_override_and_ignores_loopback_or_tokens(monkeypatch):
    import visualizer
    from unittest.mock import MagicMock
    route = MagicMock()
    route.__enter__.return_value = route
    route.connect.side_effect = OSError("No route")
    monkeypatch.setattr(visualizer.socket, "socket", lambda *args: route)
    monkeypatch.setattr(visualizer.socket, "if_nameindex", lambda: [])
    monkeypatch.setattr(visualizer, "_interface_ipv4", lambda name: "10.42.0.1")
    monkeypatch.setenv("GESTUR_PORTAL_URL", "https://192.168.1.8/gestur/")
    assert visualizer.portal_url() == "https://192.168.1.8/gestur"
    monkeypatch.setenv("GESTUR_PORTAL_URL", "http://192.168.1.8:8080/gestur/")
    assert visualizer.portal_url() == "http://192.168.1.8:8080/gestur"
    for value in ("http://exhibition.local", "http://localhost:3000", "http://127.0.0.1:3000", "http://[::1]:3000", "file:///etc/passwd",
                  "http://10.42.0.1:3000/?token=secret", "http://admin:secret@10.42.0.1:3000", "http://0.0.0.0:3000"):
        monkeypatch.setenv("GESTUR_PORTAL_URL", value)
        assert visualizer.portal_url() == "http://10.42.0.1"


def test_portal_url_uses_route_and_waits_for_network_without_fabricating_url(monkeypatch):
    import visualizer
    from unittest.mock import MagicMock
    monkeypatch.delenv('GESTUR_PORTAL_URL', raising=False)
    monkeypatch.setattr(visualizer.socket, 'if_nameindex', lambda: [])
    monkeypatch.setattr(visualizer, '_interface_ipv4', lambda name: None)
    route = MagicMock()
    route.__enter__.return_value = route
    route.getsockname.return_value = ('192.168.1.22', 0)
    monkeypatch.setattr(visualizer.socket, 'socket', lambda *args: route)
    assert visualizer.portal_url() == 'http://192.168.1.22'
    route.connect.side_effect = OSError('No route')
    monkeypatch.setattr(visualizer.socket, 'gethostname', lambda: 'gestur')
    assert visualizer.portal_url() is None
    monkeypatch.setattr(visualizer.socket, 'gethostname', lambda: 'exhibidor.local')
    assert visualizer.portal_url() is None


def test_qr_contains_reachable_url_without_admin_token(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import visualizer
    from visualizer import ControlledObjViewer
    monkeypatch.setenv("GESTUR_PORTAL_URL", "http://10.42.0.1")
    monkeypatch.setattr(visualizer.socket, 'if_nameindex', lambda: [(1, 'wlan0')])
    monkeypatch.setattr(visualizer, '_interface_ipv4', lambda name: '10.42.0.1')
    viewer = ControlledObjViewer(None, window_type="none", fullscreen=False)
    try:
        texture = viewer.welcome_overlay.find("**/welcome-qr").get_texture()
        raw = np.frombuffer(texture.get_ram_image(), dtype=np.uint8)
        rgba = raw.reshape(texture.get_y_size(), texture.get_x_size(), 4)[::-1]
        assert np.all(rgba[:, :, :3] == 255)
        image = 255 - rgba[:, :, 3]
        image = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
        assert decoded == "http://10.42.0.1"
    finally:
        viewer.destroy()
