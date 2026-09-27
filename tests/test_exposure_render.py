"""Exposure modifies final model RGB without a postprocess or alpha changes."""
import pytest


def model_asset(root, kind):
    import panda3d.core as p
    model = p.NodePath("fixture")
    card = p.CardMaker("surface")
    card.set_frame(-1, 1, -1, 1)
    node = model.attach_new_node(card.generate())
    node.set_p(90)
    if kind == "vertex":
        geom = node.node().modify_geom(0)
        data = p.GeomVertexData(geom.get_vertex_data().convert_to(p.GeomVertexFormat.get_v3n3c4()))
        colors = p.GeomVertexWriter(data, "color")
        for _ in range(data.get_num_rows()):
            colors.add_data4(.15, .25, .35, 1)
        geom.set_vertex_data(data)
    else:
        texture = p.Texture("original-texture")
        texture.setup_2d_texture(1, 1, p.Texture.T_unsigned_byte, p.Texture.F_rgba)
        texture.set_ram_image(bytes((40, 80, 120, 128 if kind == "alpha" else 255)))
        node.set_texture(texture)
        if kind == "alpha":
            node.set_transparency(p.TransparencyAttrib.M_alpha)
        if kind == "multistage":
            detail = p.TextureStage("imported-detail")
            detail.set_sort(17)
            extra = p.Texture("detail-texture")
            extra.setup_2d_texture(1, 1, p.Texture.T_unsigned_byte, p.Texture.F_rgba)
            extra.set_ram_image(bytes((180, 180, 180, 255)))
            node.set_texture(detail, extra)
    asset = root / f"{kind}.bam"
    model.write_bam_file(str(asset))
    model.remove_node()
    return asset


@pytest.mark.parametrize("scenario", ["none", "studio"])
@pytest.mark.parametrize("kind", ["texture", "vertex", "alpha", "multistage"])
def test_exposure_is_monotonic_reversible_and_preserves_alpha(tmp_path, kind, scenario):
    p = pytest.importorskip("panda3d.core")
    import numpy as np
    from visualizer import ControlledObjViewer
    viewer = ControlledObjViewer(model_asset(tmp_path, kind), window_type="offscreen",
                                 fullscreen=False, ambient_light=scenario)
    clock = p.ClockObject.get_global_clock()
    clock.set_mode(p.ClockObject.M_non_real_time)
    clock.set_dt(1 / 60)
    try:
        model = viewer.model
        original_stages = list(model.find_all_texture_stages())
        original_textures = [(texture, bytes(texture.get_ram_image())) for texture in model.find_all_textures()]
        original_geometry = list(model.find_all_matches("**/+GeomNode"))
        def frame(exposure):
            viewer.apply_settings(exposure=exposure)
            for _ in range(3):
                viewer.invalidate()
                viewer.taskMgr.step()
            return bytes(viewer.win.get_screenshot().get_ram_image())
        baseline = frame(50)
        baseline_pixels = np.frombuffer(baseline, dtype=np.uint8).reshape(-1, 4)
        measures = {}
        identity = None
        for value in (10, 50, 75, 100):
            pixels = np.frombuffer(frame(value), dtype=np.uint8).reshape(-1, 4)
            measures[value] = int(pixels[:, :3].sum())
            assert np.array_equal(pixels[:, 3], baseline_pixels[:, 3])
            assert viewer.model == model
            assert list(model.find_all_matches("**/+GeomNode")) == original_geometry
            assert all(bytes(texture.get_ram_image()) == raw for texture, raw in original_textures)
            assert viewer.get_render_status()["exposure"] == value
            if value != 50:
                stage = viewer._exposure_stage
                assert stage.get_sort() > max((item.get_sort() for item in original_stages), default=0)
                assert stage.get_combine_alpha_mode() == p.TextureStage.CM_replace
                if identity is None:
                    identity = viewer._exposure_texture
                assert identity == viewer._exposure_texture
                assert (identity.get_x_size(), identity.get_y_size()) == (1, 1)
        assert 0 < measures[10] < measures[50] < measures[75] < measures[100]
        assert frame(50) == baseline
        assert list(model.find_all_texture_stages()) == original_stages
        assert viewer._model_light_root is None if scenario == "none" else viewer._model_light_root is not None
    finally:
        viewer.destroy()


def test_bad_exposure_does_not_partially_apply_a_light_change():
    pytest.importorskip("panda3d.core")
    from visualizer import ControlledObjViewer
    viewer = ControlledObjViewer(None, window_type="none", fullscreen=False)
    try:
        for value in (True, 9, 101, 50.0, "50"):
            with pytest.raises(ValueError):
                viewer.apply_settings(ambient_light="gallery", exposure=value)
            assert viewer.get_render_status()["ambient_light"] == "none"
            assert viewer.get_render_status()["exposure"] == 50
    finally:
        viewer.destroy()
