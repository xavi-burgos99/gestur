"""Lighting presets affect only the model, with a bounded light count."""

import pytest


def write_model(root):
    from PIL import Image

    Image.new("RGB", (2, 2), (120, 180, 90)).save(root / "texture.png")
    (root / "model.mtl").write_text("newmtl surface\nKd 1 1 1\nmap_Kd texture.png\n")
    asset = root / "model.obj"
    asset.write_text(
        "mtllib model.mtl\nv -1 0 0\nv 1 0 0\nv 0 0 1\n"
        "vt 0 0\nvt 1 0\nvt 0.5 1\nusemtl surface\nf 1/1 2/2 3/3\n"
    )
    return asset


def test_presets_preserve_assets_and_transforms_without_leaking_lights(
    tmp_path, monkeypatch
):
    core = pytest.importorskip("panda3d.core")
    from gestur.visualizer import ControlledObjViewer

    asset = write_model(tmp_path)
    viewer = ControlledObjViewer(asset, window_type="none", fullscreen=False)
    try:
        model = viewer.model
        original_light_state = model.get_attrib(core.LightAttrib)
        original_textures = [
            (str(t), bytes(t.get_ram_image())) for t in model.find_all_textures()
        ]
        original_geoms = list(model.find_all_matches("**/+GeomNode"))
        viewer.update_model(rotation=[10, 20, 30], position=[1, 2, 3], scale=0.8)
        state = viewer.get_current_state()
        monkeypatch.setattr(
            viewer.loader,
            "loadModel",
            lambda *a, **kw: pytest.fail("lights reloaded model"),
        )
        for _ in range(4):
            for preset in ("studio", "gallery", "sunset", "rim", "none"):
                viewer.apply_settings(ambient_light=preset)
                assert viewer.model == model
                assert viewer.get_current_state() == state
                assert list(model.find_all_matches("**/+GeomNode")) == original_geoms
                assert [
                    (str(t), bytes(t.get_ram_image()))
                    for t in model.find_all_textures()
                ] == original_textures
                assert viewer.render.get_attrib(core.LightAttrib) is None
                assert viewer.get_render_status()["ambient_light"] == preset
                if preset == "none":
                    assert model.get_attrib(core.LightAttrib) == original_light_state
                    assert viewer._model_light_root is None
                    assert viewer.render.find("**/gestur-model-lighting").is_empty()
                else:
                    assert model.get_attrib(core.LightAttrib).get_num_on_lights() == 3
                    assert (
                        len(viewer.render.find_all_matches("**/gestur-model-lighting"))
                        == 1
                    )
                    lamps = [
                        node
                        for node in model.get_attrib(core.LightAttrib).get_on_lights()
                        if not isinstance(node.node(), core.AmbientLight)
                    ]
                    assert len(lamps) == 2
                    assert any(
                        isinstance(node.node(), core.Spotlight) for node in lamps
                    ) == (preset == "gallery")
                    assert all(not node.node().is_shadow_caster() for node in lamps)
                    assert all(
                        list(node.node().get_specular_color()) == [0, 0, 0, 1]
                        for node in lamps
                    )
                    transforms = [node.get_transform(viewer.render) for node in lamps]
                    viewer.update_model(rotation=[35, 60, 90])
                    assert [
                        node.get_transform(viewer.render) for node in lamps
                    ] == transforms
                    viewer.update_model(**state)
    finally:
        viewer.destroy()


def test_welcome_is_unaffected_and_next_model_inherits_selected_lighting(
    tmp_path, monkeypatch
):
    core = pytest.importorskip("panda3d.core")
    from gestur import visualizer

    monkeypatch.setattr(visualizer, "portal_url", lambda: "http://10.42.0.1")
    viewer = visualizer.ControlledObjViewer(None, window_type="none", fullscreen=False)
    try:
        welcome_state = viewer.welcome.get_state()
        welcome_lights = [
            list(light.node().get_color())
            for light in welcome_state.get_attrib(core.LightAttrib).get_on_lights()
        ]
        overlay = viewer.welcome_overlay
        for preset in ("gallery", "sunset", "rim", "studio"):
            viewer.apply_settings(ambient_light=preset, exposure=75)
            assert viewer.welcome.get_state() == welcome_state
            assert viewer.welcome_overlay == overlay
            assert viewer._model_light_root is None
            assert viewer._exposure_stage is None
            assert viewer.render.get_attrib(core.ColorScaleAttrib) is None
        viewer.load_model(write_model(tmp_path))
        assert viewer.model.get_attrib(core.LightAttrib).get_num_on_lights() == 3
        assert viewer.get_render_status()["ambient_light"] == "studio"
        assert viewer.get_render_status()["exposure"] == 75
        assert viewer.model.has_texture(viewer._exposure_stage)
        viewer.apply_settings(target_fps=30)  # Older callers may omit lighting.
        assert viewer.get_render_status()["ambient_light"] == "studio"
        viewer.load_model(None)
        assert viewer._model_light_root is None
        assert [
            list(light.node().get_color())
            for light in viewer.welcome.get_attrib(core.LightAttrib).get_on_lights()
        ] == welcome_lights
    finally:
        viewer.destroy()


@pytest.mark.parametrize("custom_controls", [False, True])
def test_controller_updates_lighting_without_reload_and_preserves_pose_on_control_changes(
    tmp_path, monkeypatch, custom_controls
):
    import json
    from types import SimpleNamespace

    from gestur import controller, visualizer
    from gestur.runtime_config import default_config, save_config

    models = tmp_path / "models"
    package = models / "11111111-1111-1111-1111-111111111111"
    package.mkdir(parents=True)
    write_model(package)
    (package / ".gestur-model.json").write_text('{"entrypoint":"model.obj"}')
    config = default_config()
    config["active_model"] = f"{package.name}/model.obj"
    config["controls"]["mappings"] = [
        {
            "id": "x",
            "input": "head_x",
            "output": "position_x",
            "enabled": True,
            "mode": "absolute",
            "scale": 2,
            "center": 0.5,
            "invert": False,
        }
    ]
    path = tmp_path / "config.json"
    save_config(config, path)
    original_viewer = visualizer.ControlledObjViewer

    def headless(model_path, **kwargs):
        kwargs["fullscreen"] = False
        return original_viewer(model_path, window_type="none", **kwargs)

    monkeypatch.setattr(visualizer, "ControlledObjViewer", headless)
    custom = (
        SimpleNamespace(
            process_input=lambda data: {
                "position": [1, 2, 3],
                "rotation": [10, 20, 30],
                "scale": 1.2,
            }
        )
        if custom_controls
        else None
    )
    app = controller.PoseController(
        config_path=path, models_dir=models, no_camera=True, control_system=custom
    )
    try:
        controls = app.control_system
        visible = controls.process_input({"head": {"detected": True, "x": 0.9}})
        app.visualizer.update_model(**visible)
        visual_state = app.visualizer.get_current_state()
        model = app.visualizer.model
        monkeypatch.setattr(
            app.visualizer,
            "load_model",
            lambda *a, **kw: pytest.fail("settings reloaded model"),
        )
        config["render"]["ambient_light"] = "sunset"
        config["render"]["exposure"] = 75
        save_config(config, path)
        app._reload_config()
        assert app.control_system is controls
        assert app.visualizer.model == model
        assert app.visualizer.get_render_status()["ambient_light"] == "sunset"
        assert app.visualizer.get_render_status()["exposure"] == 75
        light_root = app.visualizer._model_light_root
        config["render"]["exposure"] = 50
        save_config(config, path)
        app._reload_config()
        assert app.visualizer._model_light_root == light_root
        assert not app.visualizer.model.has_texture(app.visualizer._exposure_stage)
        assert app.visualizer.get_render_status()["exposure"] == 50
        assert app.visualizer.get_current_state() == visual_state
        assert app.exit_code == 0
        config["controls"]["idle_mode"] = "hold"
        save_config(config, path)
        app._reload_config()
        assert app.control_system is not controls
        assert app.control_system.process_input({}) == visible
        assert app.visualizer.get_current_state() == visual_state
        app.status_path = tmp_path / "runtime.json"
        app._write_status()
        status = json.loads(app.status_path.read_text())
        assert status["idle"] == app.control_system.get_idle_status()
        assert status["render_scheduler"]["ambient_light"] == "sunset"
        assert status["render_scheduler"]["exposure"] == 50
    finally:
        app.cleanup()
