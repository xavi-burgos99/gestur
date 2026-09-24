"""Panda3D viewer. All scene changes belong to the application's render thread."""
import math
from pathlib import Path
import sys

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AntialiasAttrib, ClockObject, Filename, WindowProperties, loadPrcFileData,
)


class ControlledObjViewer(ShowBase):
    def __init__(self, obj_path, *, target_fps=60, antialias_samples=2,
                 fullscreen=True, hide_cursor=True, show_fps=False,
                 window_type=None):
        # Configure before creating the context. Preserve geometry and textures.
        if antialias_samples not in (0, 2, 4):
            raise ValueError("antialias_samples debe ser 0, 2 o 4")
        loadPrcFileData("gestur", "\n".join((
            "load-file-type p3assimp",
            "win-size 1920 1080",
            f"fullscreen {'true' if fullscreen and sys.platform != 'darwin' else 'false'}",
            f"fullscreen-windowed {'true' if fullscreen and sys.platform == 'darwin' else 'false'}",
            f"framebuffer-multisample {'true' if antialias_samples else 'false'}",
            f"multisamples {antialias_samples}",
            "sync-video true",
            "audio-library-name null",
            "textures-power-2 none",
            "model-cache-models true",
        )))
        super().__init__(**({"windowType": window_type} if window_type else {}))
        self.setBackgroundColor(0, 0, 0, 1)
        self.disableMouse()
        if self.cam:
            # Preserve the original capitell framing and axes.
            self.cam.set_pos(0, 1, -25)
            self.cam.look_at(0, 0, 0)
        self.render.set_shader_auto()
        self.render.set_antialias(
            AntialiasAttrib.MMultisample if antialias_samples else AntialiasAttrib.MNone)
        self.current_state = {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
        self.model = None
        self.model_path = None
        self.apply_settings(target_fps=target_fps, hide_cursor=hide_cursor)
        self.setFrameRateMeter(show_fps)
        try:
            self.load_model(obj_path)
        except Exception:
            self.destroy()
            raise

    def apply_settings(self, *, target_fps=60, hide_cursor=True):
        clock = ClockObject.get_global_clock()
        clock.set_mode(ClockObject.MLimited)
        clock.set_frame_rate(target_fps)
        if self.win and hasattr(self.win, "request_properties"):
            properties = WindowProperties()
            properties.set_cursor_hidden(hide_cursor)
            self.win.request_properties(properties)

    def load_model(self, obj_path):
        """Load before replacing the current scene; failed loads leave it intact."""
        path = Path(obj_path).expanduser().resolve(strict=True)
        try:
            candidate = self.loader.loadModel(Filename.from_os_specific(str(path)), okMissing=True)
        except Exception as exc:
            raise ValueError(f"No se pudo interpretar el modelo: {path.name}") from exc
        if candidate is None or candidate.is_empty():
            raise ValueError(f"No se pudo cargar el modelo: {path.name}")
        try:
            if not any(node.node().get_num_geoms() for node in candidate.find_all_matches("**/+GeomNode")):
                raise ValueError(f"El modelo no contiene geometría visible: {path.name}")
            # Each exhibition object moves as one unit. Merge compatible draw
            # batches without decimating vertices, UVs, materials or textures.
            candidate.clear_model_nodes()
            candidate.flatten_strong()
            wrapper = self.render.attach_new_node("gestur-object")
            candidate.reparent_to(wrapper)
            if path != Path(__file__).with_name("capitell.obj").resolve():
                bounds = candidate.get_tight_bounds()
                if bounds:
                    low, high = bounds
                    extent = max(high - low)
                    if extent > 1e-8:
                        factor = 12.0 / extent
                        candidate.set_scale(factor)
                        candidate.set_pos(-(low + high) * (0.5 * factor))
            wrapper.set_pos(*self.current_state["position"])
            wrapper.set_hpr(*self.current_state["rotation"])
            wrapper.set_scale(*self.current_state["scale"])
        except Exception as exc:
            candidate.remove_node()
            if "wrapper" in locals():
                wrapper.remove_node()
            raise ValueError(f"No se pudo preparar el modelo: {path.name}") from exc
        previous = self.model
        self.model = wrapper
        self.model_path = str(path)
        if previous is not None:
            previous.remove_node()
        # The scene owns its assets; avoid retaining previously selected models.
        self.loader.unloadModel(Filename.from_os_specific(str(path)))

    def update_model(self, **kwargs):
        if self.model is None:
            return
        for key, setter in (("position", self.model.set_pos),
                            ("rotation", self.model.set_hpr),
                            ("scale", self.model.set_scale)):
            if key not in kwargs:
                continue
            value = kwargs[key]
            if key == "scale" and isinstance(value, (int, float)):
                value = [value] * 3
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"Transformación inválida: {key}")
            value = list(value)
            if value != self.current_state[key]:
                setter(*value)
                self.current_state[key] = value

    def get_current_state(self):
        return {key: list(value) for key, value in self.current_state.items()}

    def set_model_rotation_limited(self, pitch=0, yaw=0, roll=0):
        self.update_model(rotation=[yaw, pitch, roll])

    def set_model_position(self, x=0, y=0, z=0):
        self.update_model(position=[x, y, z])

    def set_model_scale(self, scale=1):
        self.update_model(scale=scale)
