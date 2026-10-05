"""Panda3D viewer. All scene changes belong to the application's render thread."""

import ipaddress
import math
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    ClockObject,
    DirectionalLight,
    DynamicTextFont,
    Filename,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    PerspectiveLens,
    PNMImage,
    PythonCallbackObject,
    Spotlight,
    TextNode,
    Texture,
    TextureStage,
    TransparencyAttrib,
    WindowProperties,
    loadPrcFileData,
)

from render_scheduler import RenderCadence
from runtime_config import validate_model_orientation, validate_model_url
from runtime_state import FrameMetrics
from screen_settings import DEFAULT_SCREEN, SIZE_FACTORS, rotate_display

# Positions are in the fixed exhibition camera's frame: X right, Y away
# from the viewer, Z up. These are distinct light rigs, not exposure filters.
# Each rig uses at most three lights, without shadow maps or extra render passes.
MODEL_LIGHT_PRESETS = {
    "studio": {
        "ambient": (0.4, 0.4, 0.4, 1),
        "lights": (
            {
                "role": "key",
                "type": "directional",
                "color": (1.6, 1.6, 1.6, 1),
                "position": (-8, -10, 10),
            },
            {
                "role": "fill",
                "type": "directional",
                "color": (0.8, 0.8, 0.8, 1),
                "position": (10, -7, 1),
            },
        ),
    },
    "gallery": {
        "ambient": (0.25, 0.25, 0.25, 1),
        "lights": (
            {
                "role": "key",
                "type": "spot",
                "color": (3.2, 3.05, 2.83, 1),
                "position": (-3, -8, 16),
                "fov": 40,
                "exponent": 6,
                "attenuation": (1, 0, 0.001),
            },
            {
                "role": "fill",
                "type": "directional",
                "color": (0.22, 0.25, 0.28, 1),
                "position": (8, -4, -2),
            },
        ),
    },
    "sunset": {
        "ambient": (0.2, 0.24, 0.32, 1),
        "lights": (
            {
                "role": "key",
                "type": "directional",
                "color": (2.5, 1.25, 0.55, 1),
                "position": (-14, -4, 2),
            },
            {
                "role": "fill",
                "type": "directional",
                "color": (0.35, 0.45, 0.65, 1),
                "position": (7, -10, 7),
            },
        ),
    },
    "rim": {
        "ambient": (0.2, 0.21, 0.23, 1),
        "lights": (
            {
                "role": "rim",
                "type": "directional",
                "color": (2.1, 2.2, 2.4, 1),
                "position": (12, 3, 5),
            },
            {
                "role": "fill",
                "type": "directional",
                "color": (0.5, 0.55, 0.65, 1),
                "position": (-2, -12, 0),
            },
        ),
    },
}


def _validate_exposure(value):
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("La exposición debe ser un entero entre 0 y 100")
    return value


def _usable_ipv4(address):
    try:
        ip = ipaddress.IPv4Address(address)
        return not (
            ip.is_loopback or ip.is_unspecified or ip.is_multicast or ip.is_link_local
        )
    except ipaddress.AddressValueError:
        return False


def _interface_ipv4(name):
    """Read Linux interface addresses without DNS or an external command."""
    if not sys.platform.startswith("linux"):
        return None
    import fcntl
    import struct

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
            result = fcntl.ioctl(
                handle.fileno(), 0x8915, struct.pack("256s", name.encode()[:15])
            )
        return socket.inet_ntoa(result[20:24])
    except OSError:
        return None


def portal_url():
    """Use an actual device IP: its routed LAN address, or an access point offline."""
    override = os.environ.get("GESTUR_PORTAL_URL", "").strip()
    if override:
        try:
            parsed = urlsplit(override)
            if (
                parsed.scheme in ("http", "https")
                and _usable_ipv4(parsed.hostname)
                and not parsed.username
                and not parsed.password
                and not parsed.query
                and not parsed.fragment
            ):
                _ = parsed.port  # Explicit development overrides may use another port.
                return override.rstrip("/")
        except (TypeError, ValueError):
            pass
    # Connecting a UDP socket selects an existing route without sending packets.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
            handle.connect(("1.1.1.1", 80))
            address = handle.getsockname()[0]
        if _usable_ipv4(address):
            return f"http://{address}"
    except OSError:
        pass
    try:
        interfaces = [name for _, name in socket.if_nameindex()]
    except OSError:
        interfaces = []
    ordered = ["wlan0"] + sorted(name for name in interfaces if name != "wlan0")
    for name in ordered:
        address = _interface_ipv4(name)
        if _usable_ipv4(address):
            return f"http://{address}"
    # DHCP may not have finished at boot. The welcome screen retries periodically.
    return None


def _welcome_geometry():
    """A small sculptural ring, generated in memory (1,536 triangles)."""
    data = GeomVertexData("welcome-ring", GeomVertexFormat.get_v3n3c4(), Geom.UH_static)
    vertices, normals, colors = (
        GeomVertexWriter(data, name) for name in ("vertex", "normal", "color")
    )
    triangles = GeomTriangles(Geom.UH_static)
    segments, sides = 64, 12
    for i in range(segments + 1):
        angle = i * math.tau / segments
        for j in range(sides + 1):
            tube = j * math.tau / sides
            radius = 1.75 + 0.42 * math.cos(tube)
            vertices.add_data3(
                radius * math.cos(angle),
                radius * math.sin(angle),
                0.42 * math.sin(tube),
            )
            normals.add_data3(
                math.cos(tube) * math.cos(angle),
                math.cos(tube) * math.sin(angle),
                math.sin(tube),
            )
            colors.add_data4(1, 1, 1, 1)
            if i < segments and j < sides:
                a = i * (sides + 1) + j
                b = a + sides + 1
                triangles.add_vertices(a, b, a + 1)
                triangles.add_vertices(a + 1, b, b + 1)
    geom = Geom(data)
    geom.add_primitive(triangles)
    node = GeomNode("welcome-ring")
    node.add_geom(geom)
    return node


def _qr_texture(url, name):
    import qrcode

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=1, border=4
    )
    qr.add_data(url)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    texture = Texture(name)
    texture.setup_2d_texture(
        len(matrix), len(matrix), Texture.T_unsigned_byte, Texture.F_rgba
    )
    texture.set_ram_image(
        bytes(
            channel
            for row in reversed(matrix)
            for value in row
            for channel in (255, 255, 255, 255 if value else 0)
        )
    )
    texture.set_minfilter(Texture.FT_nearest)
    texture.set_magfilter(Texture.FT_nearest)
    return texture


class ControlledObjViewer(ShowBase):
    def __init__(
        self,
        obj_path=None,
        *,
        target_fps=60,
        antialias_samples=2,
        fullscreen=True,
        hide_cursor=True,
        show_fps=False,
        window_type=None,
        model_orientation=None,
        ambient_light="none",
        exposure=50,
        model_url=None,
        model_content=None,
    ):
        # Configure before creating the context. Preserve geometry and textures.
        if antialias_samples not in (0, 2, 4):
            raise ValueError("antialias_samples debe ser 0, 2 o 4")
        if ambient_light != "none" and ambient_light not in MODEL_LIGHT_PRESETS:
            raise ValueError("Iluminación ambiental no válida")
        self._ambient_light = ambient_light
        self._exposure = _validate_exposure(exposure)
        self._exposure_stage = None
        self._exposure_texture = None
        self._model_light_root = None
        self._screen_settings = dict(DEFAULT_SCREEN)
        self._screen_applied = False
        self._idle_animation = False
        self._render_cadence = RenderCadence(target_fps)
        self.render_metrics = FrameMetrics()
        self._render_clock = time.monotonic
        self._draw_region = None
        self._draw_callback = None
        self._last_draw_size = None
        self._draw_meter = None
        self._meter_time = self._render_clock()
        self._meter_frames = 0
        self._meter_ticks = 0
        loadPrcFileData(
            "gestur",
            "\n".join(
                (
                    "load-file-type p3assimp",
                    "win-size 1920 1080",
                    f"fullscreen {'true' if fullscreen and sys.platform != 'darwin' else 'false'}",
                    f"fullscreen-windowed {'true' if fullscreen and sys.platform == 'darwin' else 'false'}",
                    f"framebuffer-multisample {'true' if antialias_samples else 'false'}",
                    f"multisamples {antialias_samples}",
                    "sync-video true",
                    # Panda otherwise busy-waits up to 10 ms on every limited tick.
                    # Keep a 4 ms margin: 1 ms overslept and reduced active FPS
                    # on macOS.
                    *(
                        ("sleep-precision 0.004",)
                        if sys.platform in ("linux", "darwin")
                        else ()
                    ),
                    "audio-library-name null",
                    "textures-power-2 none",
                    "model-cache-models true",
                )
            ),
        )
        super().__init__(**({"windowType": window_type} if window_type else {}))
        self.setBackgroundColor(0, 0, 0, 1)
        self.disableMouse()
        self._set_model_camera()
        self.render.set_shader_auto()
        self.render.set_antialias(
            AntialiasAttrib.MMultisample if antialias_samples else AntialiasAttrib.MNone
        )
        self.current_state = {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
        self.model = None
        self.model_path = None
        self.model_orientation = validate_model_orientation()
        self.model_fit = None
        self.model_basis = None
        self.model_url = None
        self.model_qr = None
        self.model_content = None
        self.content_overlay = None
        self.welcome = None
        self.welcome_overlay = None
        self.welcome_url = None
        self.model_error_overlay = None
        self._show_author_credit()
        self._last_url_check = 0.0
        # Leave ShowBase's input/event/igLoop tasks in place. GraphicsOutput's
        # active flag skips cull/draw only, keeping control and events responsive.
        self.taskMgr.add(self._prepare_draw, "gestur-render-cadence", sort=49)
        if self.cam and self.cam.node().get_num_display_regions():
            self._draw_region = self.cam.node().get_display_region(0)
            self._draw_callback = PythonCallbackObject(self._record_draw)
            self._draw_region.set_draw_callback(self._draw_callback)
        self.apply_settings(
            target_fps=target_fps,
            hide_cursor=hide_cursor,
            ambient_light=ambient_light,
            exposure=exposure,
        )
        # Panda's built-in FPS meter counts task ticks, including skipped draws.
        self.setFrameRateMeter(False)
        if show_fps:
            self._draw_meter = TextNode("gestur-draw-meter")
            self._draw_meter.set_text("Dibujo: -- fps | Control: -- fps")
            meter = self.a2dTopLeft.attach_new_node(self._draw_meter)
            meter.set_scale(0.035)
            meter.set_pos(0.03, 0, -0.06)
        try:
            self.load_model(
                obj_path, orientation=model_orientation, model_url=model_url
            )
            self.set_model_content(model_content)
        except Exception:
            self.destroy()
            raise

    def apply_settings(
        self, *, target_fps=60, hide_cursor=True, ambient_light=None, exposure=None
    ):
        if (
            ambient_light is not None
            and ambient_light != "none"
            and ambient_light not in MODEL_LIGHT_PRESETS
        ):
            raise ValueError("Iluminación ambiental no válida")
        if exposure is not None:
            _validate_exposure(exposure)
        if ambient_light is not None and ambient_light != self._ambient_light:
            self._ambient_light = ambient_light
            self._apply_model_lighting()
        if exposure is not None and exposure != self._exposure:
            self._exposure = exposure
            self._apply_model_exposure()
        clock = ClockObject.get_global_clock()
        clock.set_mode(ClockObject.MLimited)
        clock.set_frame_rate(target_fps)
        self._render_cadence.target_fps = float(target_fps)
        self.invalidate(frames=2)
        if self.win and hasattr(self.win, "request_properties"):
            properties = WindowProperties()
            properties.set_cursor_hidden(hide_cursor)
            self.win.request_properties(properties)

    def invalidate(self, *, frames=1):
        """Request a draw on the next control tick; call on the render thread."""
        self._render_cadence.invalidate(frames)

    def windowEvent(self, win):
        super().windowEvent(win)
        if win == self.win:
            # Preserve ShowBase resize, close and foreground handling, including
            # rebuilding both buffers after resize/restore on double buffering.
            self.invalidate(frames=2)

    def _record_draw(self, callback_data):
        callback_data.upcall()
        # Count the actual main-camera draw traversal, not a requested frame.
        self.render_metrics.tick(self._render_clock())

    def _prepare_draw(self, task):
        now = self._render_clock()
        if self._draw_meter is not None and now - self._meter_time >= 1.0:
            duration = now - self._meter_time
            draws = (self.render_metrics.frames - self._meter_frames) / duration
            ticks = (self._render_cadence.ticks - self._meter_ticks) / duration
            self._draw_meter.set_text(
                f"Dibujo: {draws:.1f} fps | Control: {ticks:.1f} fps"
            )
            self._meter_time = now
            self._meter_frames = self.render_metrics.frames
            self._meter_ticks = self._render_cadence.ticks
            self.invalidate()
        if self.win:
            size = self.win.get_x_size(), self.win.get_y_size()
            if size != self._last_draw_size:
                self._last_draw_size = size
                self._layout_welcome()
                self._layout_model_qr()
                self._layout_model_content()
                self.invalidate(frames=2)
            draw = self._render_cadence.due(
                now, welcome=self.welcome is not None or self._idle_animation
            )
            self.win.set_active(draw)
            if draw:
                self._animate_welcome(task)
        return task.cont

    def get_render_status(self):
        return {
            **self.render_metrics.summary(),
            "control_ticks": self._render_cadence.ticks,
            "skipped_draws": self._render_cadence.skipped,
            "mode": (
                "floating"
                if self._idle_animation and self._render_cadence.mode == "welcome"
                else self._render_cadence.mode
            ),
            "ambient_light": self._ambient_light,
            "exposure": self._exposure,
            "idle_animation": self._idle_animation,
            "floating_fps_limit": min(
                self._render_cadence.target_fps, self._render_cadence.welcome_fps
            ),
            "idle_refresh_fps": min(
                self._render_cadence.target_fps, self._render_cadence.idle_fps
            ),
            "welcome_fps_limit": min(
                self._render_cadence.target_fps, self._render_cadence.welcome_fps
            ),
        }

    def apply_screen_settings(self, settings):
        """Apply only changed display settings; no extra work in the draw loop."""
        if self._screen_applied and settings == self._screen_settings:
            return
        old = self._screen_settings
        if not self._screen_applied or settings["orientation"] != old["orientation"]:
            try:
                size = rotate_display(settings["orientation"])
                if size and self.win:
                    properties = WindowProperties()
                    properties.set_size(*size)
                    self.win.request_properties(properties)
            except (OSError, RuntimeError, TimeoutError) as error:
                raise RuntimeError(
                    f"No se pudo aplicar la orientación de pantalla: {error}"
                ) from error
        self._screen_settings = dict(settings)
        self._screen_applied = True
        self.author_credit.set_scale(SIZE_FACTORS[settings["content_size"]])
        if self.model is not None:
            self._orient_and_fit(
                self.model_basis, self.model_fit, self.model_orientation
            )
        self._refresh_welcome_overlay(force=True)
        self._layout_model_qr()
        self._layout_model_content()
        if self.model_error_overlay is not None:
            self.model_error_overlay.set_scale(SIZE_FACTORS[settings["content_size"]])
        self.invalidate(frames=2)

    def destroy(self):
        if getattr(self, "_draw_region", None) is not None:
            self._draw_region.clear_draw_callback()
            self._draw_region = None
            self._draw_callback = None
        if getattr(self, "taskMgr", None) is not None:
            self.taskMgr.remove("gestur-render-cadence")
        super().destroy()

    def load_model(self, obj_path, *, orientation=None, model_url=None):
        """Load before replacing the current scene; failed loads leave it intact."""
        if obj_path is None:
            self.show_welcome()
            return
        orientation = validate_model_orientation(orientation)
        model_url = validate_model_url(model_url)
        path = Path(obj_path).expanduser().resolve(strict=True)
        try:
            candidate = self.loader.loadModel(
                Filename.from_os_specific(str(path)), okMissing=True
            )
        except Exception as exc:
            raise ValueError(f"No se pudo interpretar el modelo: {path.name}") from exc
        if candidate is None or candidate.is_empty():
            raise ValueError(f"No se pudo cargar el modelo: {path.name}")
        try:
            if not any(
                node.node().get_num_geoms()
                for node in candidate.find_all_matches("**/+GeomNode")
            ):
                raise ValueError(
                    f"El modelo no contiene geometría visible: {path.name}"
                )
            # Each exhibition object moves as one unit. Merge compatible draw
            # batches without decimating vertices, UVs, materials or textures.
            candidate.clear_model_nodes()
            candidate.flatten_strong()
            wrapper = self.render.attach_new_node("gestur-object")
            fit = wrapper.attach_new_node("gestur-model-fit")
            basis = fit.attach_new_node("gestur-model-orientation")
            candidate.reparent_to(basis)
            self._orient_and_fit(basis, fit, orientation)
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
        self.model_fit = fit
        self.model_basis = basis
        self.model_orientation = orientation
        self.model_path = str(path)
        self._apply_model_lighting()
        self._apply_model_exposure()
        if previous is not None:
            previous.remove_node()
        self._remove_welcome()
        self._remove_model_error()
        self.author_credit.set_color_scale(1, 1, 1, 0.65)
        self.set_model_url(model_url)
        self.set_model_content(None)
        self._set_model_camera()
        self.setBackgroundColor(0, 0, 0, 1)
        # The scene owns its assets; avoid retaining previously selected models.
        self.loader.unloadModel(Filename.from_os_specific(str(path)))
        self.invalidate(frames=2)

    def set_model_content(self, content=None):
        from runtime_config import validate_model_content

        content = validate_model_content(content)
        if self.model is None:
            content = validate_model_content()
        if content == self.model_content:
            return
        self.model_content = content
        if self.content_overlay is not None:
            self.content_overlay.remove_node()
            self.content_overlay = None
        if content["title"] or content["description"]:
            anchor = (
                self.a2dTopCenter
                if content["placement"] == "top"
                else self.a2dBottomCenter
            )
            self.content_overlay = anchor.attach_new_node("gestur-model-content")
            self._configure_screen_overlay(self.content_overlay)
            image = PNMImage(1, 256, 4)
            for row in range(256):
                distance = row / 255
                if content["placement"] == "bottom":
                    distance = 1 - distance
                # Keep the first quarter dark, then fade with flat endpoints.
                progress = max(0, min(1, (distance - 0.25) / 0.75))
                alpha = 0.8 * (1 - progress * progress * (3 - 2 * progress))
                image.set_xel_a(0, row, 0, 0, 0, alpha)
            texture = Texture("content-gradient")
            texture.load(image)
            # Repeating the texture blends the dark edge into the clear edge.
            texture.set_wrap_u(Texture.WM_clamp)
            texture.set_wrap_v(Texture.WM_clamp)
            texture.set_minfilter(Texture.FT_linear)
            texture.set_magfilter(Texture.FT_linear)
            card = CardMaker("content-gradient")
            card.set_frame(-1, 1, -1, 0)
            self.content_gradient = self.content_overlay.attach_new_node(
                card.generate()
            )
            self.content_gradient.set_texture(texture)
            self.content_gradient.set_transparency(TransparencyAttrib.M_alpha)
            self.content_gradient.set_bin("fixed", 39)
            self.content_title = self._overlay_label(
                self.content_overlay, "content-title", content["title"], 0, 0.10
            )
            self.content_description = self._overlay_label(
                self.content_overlay,
                "content-description",
                content["description"],
                0,
                0.045,
            )
            self.content_description.set_color_scale(1, 1, 1, 0.8)
            self._layout_model_content()
        self.invalidate(frames=2)

    def _layout_model_content(self):
        if self.content_overlay is None:
            return
        width = abs(self.content_overlay.get_relative_point(self.render2d, (1, 0, 0)).x)
        content_factor = SIZE_FACTORS[self._screen_settings["content_size"]]
        self.content_title.set_scale(0.10 * content_factor)
        self.content_description.set_scale(0.045 * content_factor)
        for label in (self.content_title, self.content_description):
            label.node().set_wordwrap(width * 1.35 / label.get_sx())
        title_height = (
            self.content_title.node().get_height() * self.content_title.get_sz()
        )
        description_height = (
            self.content_description.node().get_height()
            * self.content_description.get_sz()
        )
        height = title_height + description_height + 0.33
        # Long text must stay within the viewport, including portrait displays.
        factor = min(1, 0.75 / max(height, 0.01))
        self.content_title.set_scale(0.10 * content_factor * factor)
        self.content_description.set_scale(0.045 * content_factor * factor)
        title_height *= factor
        description_height *= factor
        height = title_height + description_height + 0.33
        if self.model_content["placement"] == "top":
            self.content_title.set_z(-0.24)
            self.content_description.set_z(-0.24 - title_height - 0.04)
            self.content_gradient.set_pos(0, 0, 0)
        else:
            self.content_title.set_z(height - 0.06)
            self.content_description.set_z(height - 0.1 - title_height)
            self.content_gradient.set_pos(0, 0, height + 0.24)
        self.content_gradient.set_scale(width, 1, height + 0.24)

    def set_model_url(self, url):
        """Update a static screen overlay without reloading geometry or textures."""
        url = validate_model_url(url) if self.model is not None else None
        if url == self.model_url:
            return False
        # Build first so an invalid/unencodable value cannot erase a working QR.
        texture = _qr_texture(url, "model-url-qr") if url else None
        if self.model_qr is not None:
            self.model_qr.remove_node()
            self.model_qr = None
        self.model_url = url
        if texture is not None:
            card = CardMaker("model-qr")
            card.set_frame(-1, 0, 0, 1)
            self.model_qr = self.a2dBottomRight.attach_new_node(card.generate())
            self.model_qr.set_texture(texture)
            self.model_qr.set_transparency(TransparencyAttrib.M_alpha)
            self._configure_screen_overlay(self.model_qr)
            self._layout_model_qr()
        self.invalidate(frames=2)
        return True

    def _layout_model_qr(self):
        if self.model_qr is None:
            return
        short_side = (
            min(self.win.get_x_size(), self.win.get_y_size()) if self.win else 1080
        )
        modules = self.model_qr.get_texture().get_x_size()
        # Typical URLs use ~17% of the short edge; dense codes get more room,
        # still below the welcome QR (36%). Anchor follows every window resize.
        size = max(0.32, min(0.60, modules * 4 * 2 / max(1, short_side)))
        self.model_qr.set_scale(
            size * SIZE_FACTORS[self._screen_settings["content_size"]]
        )
        self.model_qr.set_pos(-0.06, 0, 0.06)

    def _clear_model_lighting(self):
        if self.model is not None:
            self.model.clear_light()
        if self._model_light_root is not None:
            self._model_light_root.remove_node()
            self._model_light_root = None

    def _apply_model_lighting(self):
        self._clear_model_lighting()
        if self.model is None or self._ambient_light == "none":
            return
        preset = MODEL_LIGHT_PRESETS[self._ambient_light]
        # The lamp rig uses the same camera frame as _set_model_camera, but
        # remains at the object's exhibition origin and never follows gestures.
        self._model_light_root = self.render.attach_new_node("gestur-model-lighting")
        self._model_light_root.set_pos(0, 1, -25)
        self._model_light_root.look_at(0, 0, 0)
        self._model_light_root.set_pos(0, 0, 0)
        ambient = AmbientLight("model-ambient")
        ambient.set_color(preset["ambient"])
        self.model.set_light(self._model_light_root.attach_new_node(ambient))
        for spec in preset["lights"]:
            if spec["type"] == "spot":
                light = Spotlight(f"model-{spec['role']}")
                lens = PerspectiveLens()
                lens.set_fov(spec["fov"])
                lens.set_near_far(1, 100)
                light.set_lens(lens)
                light.set_exponent(spec["exponent"])
                light.set_attenuation(spec["attenuation"])
            else:
                light = DirectionalLight(f"model-{spec['role']}")
            light.set_color(spec["color"])
            # Keep diffuse scan textures readable: some imports have zero
            # shininess and otherwise produce a broad artificial white glare.
            light.set_specular_color((0, 0, 0, 1))
            light.set_shadow_caster(False)
            node = self._model_light_root.attach_new_node(light)
            node.set_pos(*spec["position"])
            node.look_at(0, 0, 0)
            self.model.set_light(node)
        self.invalidate(frames=2)

    def _apply_model_exposure(self):
        if self.model is None:
            return
        if self._exposure_stage is not None:
            self.model.clear_texture(self._exposure_stage)
        if self._exposure != 50:
            if self._exposure_stage is None:
                self._exposure_stage = TextureStage("gestur-exposure")
                self._exposure_stage.set_combine_rgb(
                    TextureStage.CM_modulate,
                    TextureStage.CS_previous,
                    TextureStage.CO_src_color,
                    TextureStage.CS_constant,
                    TextureStage.CO_src_color,
                )
                self._exposure_stage.set_combine_alpha(
                    TextureStage.CM_replace,
                    TextureStage.CS_previous,
                    TextureStage.CO_src_alpha,
                )
                self._exposure_texture = Texture("gestur-exposure-identity")
                self._exposure_texture.setup_2d_texture(
                    1, 1, Texture.T_unsigned_byte, Texture.F_rgba
                )
                self._exposure_texture.set_ram_image(bytes((255, 255, 255, 255)))
            # Preserve the saved 10–100 curve exactly. Extend its lower end
            # continuously to black without changing alpha or the 50 baseline.
            gain = (
                (self._exposure / 10) * 2 ** (-40 / 25)
                if self._exposure < 10
                else 2 ** ((self._exposure - 50) / 25)
            )
            rgb_scale = 4 if gain > 2 else 2 if gain > 1 else 1
            self._exposure_stage.set_rgb_scale(rgb_scale)
            self._exposure_stage.set_color((gain / rgb_scale,) * 3 + (1,))
            stages = self.model.find_all_texture_stages()
            self._exposure_stage.set_sort(
                max((stage.get_sort() for stage in stages), default=0) + 1
            )
            # A last texture-combine stage scales the already textured RGB,
            # avoiding fixed-function clamping of vertex colors above one.
            # It preserves alpha and uses the existing draw, not a postprocess.
            self.model.set_texture(self._exposure_stage, self._exposure_texture)
        # At 50 the stage is absent, exactly restoring the original render state.
        self.invalidate(frames=2)

    def set_idle_animation(self, active):
        """Limit decorative motion only; recognition and control ticks remain live."""
        active = bool(active) and self.model is not None
        if active != self._idle_animation:
            self._idle_animation = active
            # Also immediately wake both buffers when a person starts controlling.
            self.invalidate(frames=2)

    def _orient_and_fit(self, basis, fit, orientation):
        # Panda's heading/pitch/roll rotate Z/X/Y. The inner basis is fixed;
        # the outer exhibition wrapper remains controlled only by gestures.
        basis.set_hpr(orientation["z"], orientation["x"], orientation["y"])
        bounds = basis.get_tight_bounds(fit)
        if bounds:
            low, high = bounds
            extent = max(high - low)
            if extent > 1e-8:
                factor = (
                    12.0 * SIZE_FACTORS[self._screen_settings["model_size"]] / extent
                )
                fit.set_scale(factor)
                fit.set_pos(-(low + high) * (0.5 * factor))

    def set_model_orientation(self, orientation):
        """Rotate an existing model without reparsing assets or changing controls."""
        orientation = validate_model_orientation(orientation)
        if self.model is None or orientation == self.model_orientation:
            return False
        self._orient_and_fit(self.model_basis, self.model_fit, orientation)
        self.model_orientation = orientation
        self.invalidate(frames=2)
        return True

    def _set_model_camera(self):
        if self.cam:
            self.cam.set_pos(0, 1, -25)
            self.cam.look_at(0, 0, 0)

    def _remove_welcome(self):
        for name in ("welcome", "welcome_overlay"):
            node = getattr(self, name, None)
            if node is not None:
                node.remove_node()
                setattr(self, name, None)
        self.welcome_url = None

    def _remove_model_error(self):
        if self.model_error_overlay is not None:
            self.model_error_overlay.remove_node()
            self.model_error_overlay = None

    def show_model_error(self, message):
        """An unavailable library is not an empty library; do not show onboarding."""
        if self.model is not None:
            return
        self._remove_welcome()
        self._remove_model_error()
        self.setBackgroundColor(0, 0, 0, 1)
        self.model_error_overlay = self.aspect2d.attach_new_node("gestur-model-error")
        self.model_error_overlay.set_scale(
            SIZE_FACTORS[self._screen_settings["content_size"]]
        )
        self._overlay_label(
            self.model_error_overlay, "model-error-brand", "GESTUR", 0.24, 0.13
        )
        self._overlay_label(
            self.model_error_overlay,
            "model-error-title",
            "No se pudo cargar el modelo",
            -0.02,
            0.055,
        )
        detail = "Revisa el modelo desde el portal del dispositivo."
        self._overlay_label(
            self.model_error_overlay, "model-error-detail", detail, -0.17, 0.035
        )
        self.invalidate(frames=2)

    @staticmethod
    def _configure_screen_overlay(node):
        """Keep screen overlays independent of the model's lighting and depth."""
        node.set_light_off()
        node.set_shader_off()
        node.set_depth_test(False)
        node.set_depth_write(False)
        node.set_bin("fixed", 40)

    def _overlay_label(self, parent, name, text, z, size):
        node = TextNode(name)
        if not hasattr(self, "_welcome_font"):
            font = TextNode.get_default_font()
            if isinstance(font, DynamicTextFont):
                font = DynamicTextFont(font)
                font.set_pixels_per_unit(96)
                font.set_minfilter(Texture.FT_linear_mipmap_linear)
                font.set_magfilter(Texture.FT_linear)
            self._welcome_font = font
        node.set_font(self._welcome_font)
        node.set_text(text)
        node.set_align(TextNode.A_center)
        node.set_text_color(1, 1, 1, 1)
        item = parent.attach_new_node(node)
        item.set_scale(size)
        item.set_pos(0, 0, z)
        return item

    def _show_author_credit(self):
        """Keep attribution anchored to the viewport across all scene states."""
        self.author_credit = self.a2dBottomLeft.attach_new_node("gestur-author-credit")
        self.author_credit.set_pos(0.06, 0, 0.045)
        self.author_credit.set_transparency(TransparencyAttrib.M_alpha)
        self._configure_screen_overlay(self.author_credit)
        # A separate glyph atlas keeps static credits unchanged when welcome
        # labels add characters and regenerate their font texture.
        font = TextNode.get_default_font()
        if isinstance(font, DynamicTextFont):
            font = DynamicTextFont(font)
            font.set_pixels_per_unit(96)
            font.set_minfilter(Texture.FT_linear_mipmap_linear)
            font.set_magfilter(Texture.FT_linear)
        for name, text, height, size in (
            ("author-brand", "GESTUR", 0.034, 0.04),
            ("author-name", "Desarrollado por Xavier Burgos", 0, 0.027),
        ):
            label = self._overlay_label(self.author_credit, name, text, height, size)
            label.node().set_font(font)
            label.node().set_align(TextNode.A_left)

    def show_welcome(self):
        self.set_model_content(None)
        self.author_credit.set_color_scale(1, 1, 1, 1)
        self.invalidate(frames=2)
        self._idle_animation = False
        self._clear_model_lighting()
        self._remove_model_error()
        if self.model is not None:
            self.model.remove_node()
        self.model = None
        self.set_model_url(None)
        self.model_path = None
        self.model_fit = None
        self.model_basis = None
        self.model_orientation = validate_model_orientation()
        if self.welcome is not None:
            return
        self.setBackgroundColor(0, 0, 0, 1)
        if self.cam:
            self.cam.set_pos(0, -21, 0)
            self.cam.look_at(0, 0, 0)
        self.welcome = self.render.attach_new_node("gestur-welcome")
        self.welcome_ring = self.welcome.attach_new_node(_welcome_geometry())
        self.welcome_ring.set_hpr(25, 60, -12)
        self.welcome_ring.set_color_scale(1, 1, 1, 0.15)
        self.welcome_ring.set_transparency(TransparencyAttrib.M_alpha)
        ambient = AmbientLight("welcome-ambient")
        ambient.set_color((0.4, 0.4, 0.4, 1))
        self.welcome.set_light(self.welcome.attach_new_node(ambient))
        key = DirectionalLight("welcome-key")
        key.set_color((0.75, 0.75, 0.75, 1))
        key_node = self.welcome.attach_new_node(key)
        key_node.set_hpr(-40, -35, 0)
        self.welcome.set_light(key_node)
        self._refresh_welcome_overlay()
        self._layout_welcome()

    def _layout_welcome(self):
        """Cover the viewport with the sculpture; keep the overlay inside narrow screens."""
        if self.welcome is None:
            return
        if self.cam:
            fov = self.cam.node().get_lens().get_fov()
            span = 2 * 21 * math.tan(math.radians(max(fov)) / 2)
            self.welcome_ring.set_scale(span / 4.34 * 1.1)
        # ShowBase keeps aspect2d's shortest viewport dimension at two units,
        # so this compact centered stack fits both portrait and landscape.

    def _refresh_welcome_overlay(self, force=False):
        if self.welcome is None:
            return
        self._last_url_check = time.monotonic()
        url = portal_url()
        if not force and url == self.welcome_url and self.welcome_overlay is not None:
            return
        if self.welcome_overlay is not None:
            self.welcome_overlay.remove_node()
        self.welcome_url = url
        self.welcome_overlay = self.aspect2d.attach_new_node("gestur-onboarding")
        self.welcome_overlay.set_scale(
            SIZE_FACTORS[self._screen_settings["content_size"]]
        )
        self._configure_screen_overlay(self.welcome_overlay)

        def label(name, text, z, size):
            return self._overlay_label(self.welcome_overlay, name, text, z, size)

        label("welcome-brand", "GESTUR", 0.62, 0.17)
        if url is None:
            label("welcome-network", "Esperando conexión de red", -0.03, 0.05)
            self._layout_welcome()
            self.invalidate(frames=2)
            return
        texture = _qr_texture(url, "local-portal-qr")
        card = CardMaker("welcome-qr")
        card.set_frame(-0.36, 0.36, -0.36, 0.36)
        qr_node = self.welcome_overlay.attach_new_node(card.generate())
        qr_node.set_texture(texture)
        qr_node.set_transparency(TransparencyAttrib.M_alpha)
        qr_node.set_pos(0, 0, 0.06)
        label("welcome-prompt", "Escanea el QR para comenzar", -0.48, 0.057)
        address = f"o accede a {url}"
        label("welcome-url", address, -0.60, min(0.038, 2.1 / max(1, len(address))))
        self._layout_welcome()
        self.invalidate(frames=2)

    def _animate_welcome(self, task):
        if self.welcome is not None:
            self.welcome_ring.set_hpr(
                25 + task.time * 7, 60 + math.sin(task.time * 0.35) * 7, -12
            )
            self.welcome_ring.set_z(math.sin(task.time * 0.7) * 0.09)
            if time.monotonic() - self._last_url_check >= 15:
                self._refresh_welcome_overlay()
        return task.cont

    def update_model(self, **kwargs):
        if self.model is None:
            return False
        changed = False
        for key, setter in (
            ("position", self.model.set_pos),
            ("rotation", self.model.set_hpr),
            ("scale", self.model.set_scale),
        ):
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
                changed = True
        if changed and not self._idle_animation:
            self.invalidate()
        return changed

    def get_current_state(self):
        return {key: list(value) for key, value in self.current_state.items()}

    def set_model_rotation_limited(self, pitch=0, yaw=0, roll=0):
        self.update_model(rotation=[yaw, pitch, roll])

    def set_model_position(self, x=0, y=0, z=0):
        self.update_model(position=[x, y, z])

    def set_model_scale(self, scale=1):
        self.update_model(scale=scale)
