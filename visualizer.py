"""Panda3D viewer. All scene changes belong to the application's render thread."""
import math
import ipaddress
import os
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlsplit

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight, AntialiasAttrib, CardMaker, ClockObject, DirectionalLight,
    Filename, Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, PythonCallbackObject, TextNode, Texture, WindowProperties, loadPrcFileData,
)

from render_scheduler import RenderCadence
from runtime_state import FrameMetrics


def _usable_ipv4(address):
    try:
        ip = ipaddress.IPv4Address(address)
        return not (ip.is_loopback or ip.is_unspecified or ip.is_multicast or ip.is_link_local)
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
            result = fcntl.ioctl(handle.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))
        return socket.inet_ntoa(result[20:24])
    except OSError:
        return None


def portal_url():
    """Prefer the Wi-Fi access point, then the LAN; never encode loopback."""
    override = os.environ.get("GESTUR_PORTAL_URL", "").strip()
    if override:
        try:
            parsed = urlsplit(override)
            host = parsed.hostname
            valid_host = host and host.lower() not in ("localhost", "localhost.localdomain")
            try:
                ip = ipaddress.ip_address(host or "")
                valid_host = valid_host and not (ip.is_loopback or ip.is_unspecified or ip.is_multicast)
            except ValueError:
                pass
            if (parsed.scheme in ("http", "https") and valid_host and not parsed.username
                    and not parsed.password and not parsed.query and not parsed.fragment):
                _ = parsed.port  # Reject invalid ports before producing an unusable QR.
                return override.rstrip("/")
        except ValueError:
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
    # Connecting a UDP socket selects an existing route without sending packets.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
            handle.connect(("1.1.1.1", 80))
            address = handle.getsockname()[0]
        if _usable_ipv4(address):
            return f"http://{address}"
    except OSError:
        pass
    hostname = socket.gethostname().split(".", 1)[0]
    if not hostname or hostname.lower() == "localhost":
        hostname = "gestur"
    return f"http://{hostname}.local"


def _welcome_geometry():
    """A small sculptural ring, generated in memory (1,536 triangles)."""
    data = GeomVertexData("welcome-ring", GeomVertexFormat.get_v3n3c4(), Geom.UH_static)
    vertices, normals, colors = (GeomVertexWriter(data, name) for name in ("vertex", "normal", "color"))
    triangles = GeomTriangles(Geom.UH_static)
    segments, sides = 64, 12
    for i in range(segments + 1):
        angle = i * math.tau / segments
        for j in range(sides + 1):
            tube = j * math.tau / sides
            radius = 1.75 + .42 * math.cos(tube)
            vertices.add_data3(radius * math.cos(angle), radius * math.sin(angle), .42 * math.sin(tube))
            normals.add_data3(math.cos(tube) * math.cos(angle), math.cos(tube) * math.sin(angle), math.sin(tube))
            shade = .5 + .5 * math.cos(angle - .8)
            colors.add_data4(.18 + .13 * shade, .57 + .21 * shade, .57 + .18 * shade, 1)
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


class ControlledObjViewer(ShowBase):
    def __init__(self, obj_path=None, *, target_fps=60, antialias_samples=2,
                 fullscreen=True, hide_cursor=True, show_fps=False,
                 window_type=None):
        # Configure before creating the context. Preserve geometry and textures.
        if antialias_samples not in (0, 2, 4):
            raise ValueError("antialias_samples debe ser 0, 2 o 4")
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
        loadPrcFileData("gestur", "\n".join((
            "load-file-type p3assimp",
            "win-size 1920 1080",
            f"fullscreen {'true' if fullscreen and sys.platform != 'darwin' else 'false'}",
            f"fullscreen-windowed {'true' if fullscreen and sys.platform == 'darwin' else 'false'}",
            f"framebuffer-multisample {'true' if antialias_samples else 'false'}",
            f"multisamples {antialias_samples}",
            "sync-video true",
            # Panda otherwise busy-waits up to 10 ms on every limited tick.
            # Keep a 4 ms margin: 1 ms overslept and reduced active FPS on macOS.
            *(("sleep-precision 0.004",) if sys.platform in ("linux", "darwin") else ()),
            "audio-library-name null",
            "textures-power-2 none",
            "model-cache-models true",
        )))
        super().__init__(**({"windowType": window_type} if window_type else {}))
        self.setBackgroundColor(0, 0, 0, 1)
        self.disableMouse()
        self._set_model_camera()
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
        self.welcome = None
        self.welcome_overlay = None
        self.welcome_url = None
        self._last_url_check = 0.0
        # Leave ShowBase's input/event/igLoop tasks in place. GraphicsOutput's
        # active flag skips cull/draw only, keeping control and events responsive.
        self.taskMgr.add(self._prepare_draw, "gestur-render-cadence", sort=49)
        if self.cam and self.cam.node().get_num_display_regions():
            self._draw_region = self.cam.node().get_display_region(0)
            self._draw_callback = PythonCallbackObject(self._record_draw)
            self._draw_region.set_draw_callback(self._draw_callback)
        self.apply_settings(target_fps=target_fps, hide_cursor=hide_cursor)
        # Panda's built-in FPS meter counts task ticks, including skipped draws.
        self.setFrameRateMeter(False)
        if show_fps:
            self._draw_meter = TextNode("gestur-draw-meter")
            self._draw_meter.set_text("Dibujo: -- fps | Control: -- fps")
            meter = self.a2dTopLeft.attach_new_node(self._draw_meter)
            meter.set_scale(.035)
            meter.set_pos(.03, 0, -.06)
        try:
            self.load_model(obj_path)
        except Exception:
            self.destroy()
            raise

    def apply_settings(self, *, target_fps=60, hide_cursor=True):
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
            self._draw_meter.set_text(f"Dibujo: {draws:.1f} fps | Control: {ticks:.1f} fps")
            self._meter_time = now
            self._meter_frames = self.render_metrics.frames
            self._meter_ticks = self._render_cadence.ticks
            self.invalidate()
        if self.win:
            size = self.win.get_x_size(), self.win.get_y_size()
            if size != self._last_draw_size:
                self._last_draw_size = size
                self.invalidate(frames=2)
            draw = self._render_cadence.due(now, welcome=self.welcome is not None)
            self.win.set_active(draw)
            if draw:
                self._animate_welcome(task)
        return task.cont

    def get_render_status(self):
        return {**self.render_metrics.summary(),
                "control_ticks": self._render_cadence.ticks,
                "skipped_draws": self._render_cadence.skipped,
                "mode": self._render_cadence.mode,
                "idle_refresh_fps": min(self._render_cadence.target_fps, self._render_cadence.idle_fps),
                "welcome_fps_limit": min(self._render_cadence.target_fps, self._render_cadence.welcome_fps)}

    def destroy(self):
        if getattr(self, "_draw_region", None) is not None:
            self._draw_region.clear_draw_callback()
            self._draw_region = None
            self._draw_callback = None
        if getattr(self, "taskMgr", None) is not None:
            self.taskMgr.remove("gestur-render-cadence")
        super().destroy()

    def load_model(self, obj_path):
        """Load before replacing the current scene; failed loads leave it intact."""
        if obj_path is None:
            self.show_welcome()
            return
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
        self._remove_welcome()
        self._set_model_camera()
        self.setBackgroundColor(0, 0, 0, 1)
        # The scene owns its assets; avoid retaining previously selected models.
        self.loader.unloadModel(Filename.from_os_specific(str(path)))
        self.invalidate(frames=2)

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

    def show_welcome(self):
        self.invalidate(frames=2)
        if self.model is not None:
            self.model.remove_node()
        self.model = None
        self.model_path = None
        if self.welcome is not None:
            return
        self.setBackgroundColor(.027, .062, .071, 1)
        if self.cam:
            self.cam.set_pos(0, -19, 4)
            self.cam.look_at(0, 0, 1.5)
        self.welcome = self.render.attach_new_node("gestur-welcome")
        self.welcome_ring = self.welcome.attach_new_node(_welcome_geometry())
        self.welcome_ring.set_scale(.85)
        self.welcome_ring.set_pos(0, 0, -1.3)
        self.welcome_ring.set_hpr(25, 60, -12)
        ambient = AmbientLight("welcome-ambient")
        ambient.set_color((.5, .57, .6, 1))
        self.welcome.set_light(self.welcome.attach_new_node(ambient))
        key = DirectionalLight("welcome-key")
        key.set_color((.9, .96, 1, 1))
        key_node = self.welcome.attach_new_node(key)
        key_node.set_hpr(-40, -35, 0)
        self.welcome.set_light(key_node)
        self._refresh_welcome_overlay()

    def _refresh_welcome_overlay(self):
        import qrcode
        self._last_url_check = time.monotonic()
        url = portal_url()
        if url == self.welcome_url:
            return
        if self.welcome_overlay is not None:
            self.welcome_overlay.remove_node()
        self.welcome_url = url
        self.welcome_overlay = self.aspect2d.attach_new_node("gestur-onboarding")
        self.welcome_overlay.set_light_off()
        self.welcome_overlay.set_shader_off()
        self.welcome_overlay.set_depth_test(False)
        self.welcome_overlay.set_depth_write(False)
        self.welcome_overlay.set_bin("fixed", 40)

        def label(name, text, z, size, color):
            node = TextNode(name)
            node.set_text(text)
            node.set_align(TextNode.A_center)
            node.set_text_color(*color)
            item = self.welcome_overlay.attach_new_node(node)
            item.set_scale(size)
            item.set_pos(0, 0, z)

        label("welcome-brand", "G E S T U R", .85, .032, (.49, .83, .79, 1))
        label("welcome-prompt", "Escanea el QR para comenzar.", .72, .062, (.96, .97, .96, 1))
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=1, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        matrix = qr.get_matrix()
        size = len(matrix)
        texture = Texture("local-portal-qr")
        texture.setup_2d_texture(size, size, Texture.T_unsigned_byte, Texture.F_luminance)
        texture.set_ram_image(bytes(0 if value else 255 for row in reversed(matrix) for value in row))
        texture.set_minfilter(Texture.FT_nearest)
        texture.set_magfilter(Texture.FT_nearest)
        card = CardMaker("welcome-qr")
        card.set_frame(-.245, .245, -.245, .245)
        qr_node = self.welcome_overlay.attach_new_node(card.generate())
        qr_node.set_texture(texture)
        qr_node.set_pos(0, 0, .35)
        label("welcome-url", url, .027, min(.033, 1.5 / max(1, len(url))), (.64, .72, .73, 1))
        self.invalidate(frames=2)

    def _animate_welcome(self, task):
        if self.welcome is not None:
            self.welcome_ring.set_hpr(25 + task.time * 7, 60 + math.sin(task.time * .35) * 7, -12)
            self.welcome_ring.set_z(-1.3 + math.sin(task.time * .7) * .09)
            if time.monotonic() - self._last_url_check >= 15:
                self._refresh_welcome_overlay()
        return task.cont

    def update_model(self, **kwargs):
        if self.model is None:
            return False
        changed = False
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
                changed = True
        if changed:
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
