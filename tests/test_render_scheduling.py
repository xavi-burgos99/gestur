"""Scheduling invariants plus framebuffer checks against real Panda3D."""
import pytest

from render_scheduler import RenderCadence


def test_static_refresh_does_not_lower_control_or_changed_scene_cadence():
    cadence = RenderCadence()
    draws = [cadence.due(frame / 60) for frame in range(60)]
    assert cadence.ticks == 60
    assert 10 <= sum(draws) <= 12  # Two initial draws, then 10 Hz refresh.
    assert cadence.skipped == 60 - sum(draws)
    assert cadence.mode == "idle"
    for frame in range(60, 120):
        cadence.invalidate()
        assert cadence.due(frame / 60), "Changed input must draw on the same tick"


def test_welcome_has_its_own_cadence_and_respects_lower_target():
    cadence = RenderCadence()
    assert 30 <= sum(cadence.due(frame / 60, welcome=True) for frame in range(60)) <= 32
    cadence = RenderCadence(target_fps=15)
    assert all(cadence.due(frame / 15, welcome=True) for frame in range(30))


def test_clock_jitter_does_not_drop_welcome_to_twenty_fps():
    cadence = RenderCadence()
    draws = [frame for frame in range(180)
             if cadence.due(frame / 60 + (-.00005 if frame % 2 else .00005), welcome=True)]
    assert 90 <= len(draws) <= 92
    assert all(right - left == 2 for left, right in zip(draws[2:], draws[3:]))


@pytest.fixture
def offscreen_viewer(tmp_path, monkeypatch):
    core = pytest.importorskip("panda3d.core")
    from visualizer import ControlledObjViewer
    monkeypatch.setenv("GESTUR_PORTAL_URL", "http://10.42.0.1:3000")
    # A textured card facing the exhibition camera, with no external assets.
    root = core.NodePath("fixture")
    card = core.CardMaker("textured-card")
    card.set_frame(-1, 1, -1, 1)
    node = root.attach_new_node(card.generate())
    node.set_p(90)
    texture = core.Texture("fixture-texture")
    texture.setup_2d_texture(2, 2, core.Texture.T_unsigned_byte, core.Texture.F_rgb)
    texture.set_ram_image(bytes((60, 180, 120, 120, 210, 80, 200, 100, 40, 30, 130, 220)))
    node.set_texture(texture)
    path = tmp_path / "fixture.bam"
    root.write_bam_file(str(path))
    root.remove_node()
    viewer = ControlledObjViewer(path, window_type="offscreen", fullscreen=False)
    clock = core.ClockObject.get_global_clock()
    clock.set_mode(core.ClockObject.M_non_real_time)
    clock.set_dt(1 / 60)
    now = [0.0]
    viewer._render_clock = lambda: now[0]
    frame = [0]

    def step():
        now[0] = frame[0] / 60
        viewer.taskMgr.step()
        frame[0] += 1

    try:
        assert viewer.win is not None, "Panda3D must create an actual offscreen framebuffer"
        yield viewer, step, path
    finally:
        viewer.taskMgr.remove("test-control")
        viewer.destroy()
        clock.set_mode(core.ClockObject.M_limited)
        clock.set_frame_rate(60)


def framebuffer(viewer):
    texture = viewer.win.get_screenshot()
    assert texture is not None
    return bytes(texture.get_ram_image())


def test_real_framebuffer_survives_idle_and_wakes_in_same_control_tick(offscreen_viewer):
    viewer, step, _ = offscreen_viewer
    control_ticks = []

    def control(task):
        control_ticks.append(True)
        if len(control_ticks) == 4:
            assert viewer.update_model(rotation=[25, 0, 0])
        return task.cont

    viewer.taskMgr.add(control, "test-control", sort=10)
    step()
    step()
    original = framebuffer(viewer)
    drawn = viewer.render_metrics.frames
    assert drawn == 2
    assert len(set(original)) > 10, "Fixture must draw visible colored geometry"
    step()
    assert not viewer.win.is_active()
    assert viewer.render_metrics.frames == drawn
    assert framebuffer(viewer) == original
    step()  # Control at sort 10 invalidates before renderer's sort 49.
    assert viewer.win.is_active()
    assert viewer.render_metrics.frames == drawn + 1
    assert framebuffer(viewer) != original
    assert not viewer.update_model(rotation=[25, 0, 0])
    assert [(t.get_x_size(), t.get_y_size()) for t in viewer.model.find_all_textures()] == [(2, 2)]
    assert (viewer.win.get_x_size(), viewer.win.get_y_size()) == (1920, 1080)
    assert viewer.win.get_fb_properties().get_multisamples() == 2
    for _ in range(56):
        step()
    report = viewer.get_render_status()
    assert len(control_ticks) == report["control_ticks"] == 60
    assert 10 <= report["frames"] <= 14
    assert report["skipped_draws"] == 60 - report["frames"]


def test_real_framebuffer_model_welcome_and_url_changes_are_drawn(offscreen_viewer, monkeypatch):
    viewer, step, path = offscreen_viewer
    step()
    step()
    original = framebuffer(viewer)
    previous = viewer.model
    viewer.load_model(None)
    step()
    welcome = framebuffer(viewer)
    assert previous.is_empty()
    assert welcome != original
    assert viewer.welcome_url == "http://10.42.0.1:3000"
    step()  # Finish initial double-buffer refresh.
    before = viewer.render_metrics.frames
    overlay = viewer.welcome_overlay
    for _ in range(60):
        step()
    assert viewer.render_metrics.frames - before == 30
    assert viewer.welcome_overlay == overlay
    assert framebuffer(viewer) != welcome  # The sculpture continues to animate.
    monkeypatch.setenv("GESTUR_PORTAL_URL", "http://10.42.0.2:3000")
    viewer._refresh_welcome_overlay()
    assert overlay.is_empty()
    step()
    assert viewer.welcome_url == "http://10.42.0.2:3000"
    assert viewer.render_metrics.frames == before + 31
    welcome_node = viewer.welcome
    viewer.load_model(path)
    step()
    assert welcome_node.is_empty()
    assert viewer.welcome_overlay is None
    assert framebuffer(viewer) == original


def test_explicit_refresh_draws_on_next_tick_while_idle(offscreen_viewer):
    viewer, step, _ = offscreen_viewer
    step()
    step()
    step()
    assert not viewer.win.is_active()
    before = viewer.render_metrics.frames
    viewer.invalidate(frames=2)
    step()
    step()
    assert viewer.render_metrics.frames == before + 2


def test_real_buffer_resize_wakes_an_idle_renderer(offscreen_viewer):
    viewer, step, _ = offscreen_viewer
    step()
    step()
    step()
    assert not viewer.win.is_active()
    before = viewer.render_metrics.frames
    viewer.win.set_size(640, 360)
    step()
    assert viewer.render_metrics.frames == before + 1
    shot = viewer.win.get_screenshot()
    assert (shot.get_x_size(), shot.get_y_size()) == (640, 360)
    assert len(set(bytes(shot.get_ram_image()))) > 10


def test_floating_draws_thirty_fps_without_slowing_controls_and_wakes_immediately(offscreen_viewer):
    viewer, step, _ = offscreen_viewer
    viewer.set_idle_animation(True)
    for frame in range(60):
        viewer.update_model(rotation=[frame * .2, 0, 0])
        step()
    report = viewer.get_render_status()
    assert report["control_ticks"] == 60
    assert 30 <= report["frames"] <= 32
    assert report["skipped_draws"] == 60 - report["frames"]
    assert report["mode"] == "floating"
    before = viewer.render_metrics.frames
    viewer.set_idle_animation(False)
    viewer.update_model(rotation=[40, 0, 0])
    step()
    assert viewer.win.is_active()
    assert viewer.render_metrics.frames == before + 1
    assert not viewer.get_render_status()["idle_animation"]


def test_lighting_can_return_to_identical_unlit_framebuffer(offscreen_viewer):
    viewer, step, _ = offscreen_viewer
    step()
    step()
    unlit = framebuffer(viewer)
    lit_frames = []
    for preset in ("soft", "warm", "cool", "contrast"):
        viewer.apply_settings(ambient_light=preset)
        step()
        step()
        lit_frames.append(framebuffer(viewer))
    assert any(image != unlit for image in lit_frames)
    viewer.apply_settings(ambient_light="none")
    step()
    step()
    assert framebuffer(viewer) == unlit
