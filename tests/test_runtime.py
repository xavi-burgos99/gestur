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
    from controller import ROOT, resolve_model
    assert resolve_model("capitell.obj", tmp_path) == ROOT / "capitell.obj"
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


def test_real_capitell_geometry_and_texture_are_preserved():
    pytest.importorskip("panda3d.core")
    from controller import ROOT
    from visualizer import ControlledObjViewer
    viewer = ControlledObjViewer(ROOT / "capitell.obj", window_type="none", fullscreen=False)
    try:
        nodes = viewer.model.find_all_matches("**/+GeomNode")
        triangles = sum(n.node().get_geom(i).get_primitive(j).get_num_primitives()
                        for n in nodes for i in range(n.node().get_num_geoms())
                        for j in range(n.node().get_geom(i).get_num_primitives()))
        assert triangles == 491038
        textures = viewer.model.find_all_textures()
        assert [(t.get_x_size(), t.get_y_size()) for t in textures] == [(2048, 2048)]
        old = viewer.model
        with pytest.raises(FileNotFoundError):
            viewer.load_model(ROOT / "missing.obj")
        assert viewer.model == old
        viewer.update_model(rotation=[25, -5, 7], scale=1.75)
        assert list(viewer.model.get_hpr()) == [25, -5, 7]
        assert list(viewer.model.get_scale()) == [1.75] * 3
    finally:
        viewer.destroy()


def test_camera_failure_exits_for_kiosk_recovery():
    from controller import PoseController
    from control_system import create_default_control_system
    app = PoseController.__new__(PoseController)
    app.metrics = FrameMetrics()
    app.mailbox = LatestPose()
    app.control_system = create_default_control_system()
    app._last_config_check = 0
    app.pose_tracker = SimpleNamespace(last_error=RuntimeError("camera disconnected"))
    app.status_path = None
    stopped = []
    app.visualizer = SimpleNamespace(update_model=lambda **kwargs: None,
                                     taskMgr=SimpleNamespace(stop=lambda: stopped.append(True)))
    result = app._render_tick(SimpleNamespace(done="done", cont="cont"))
    assert result == "done"
    assert app.exit_code == 1
    assert stopped == [True]


def test_unavailable_selected_model_falls_back_to_capitell(tmp_path, monkeypatch):
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
    assert loaded == [controller.ROOT / 'capitell.obj']
    assert app.rendered_model == 'capitell.obj'
    assert app.requested_model == 'package/missing.obj'
    assert app.last_error
