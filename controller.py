"""Independent camera inference and frame-paced control for Gestur."""
import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

from control_system import create_control_system
from runtime_config import default_config, load_config, validate_config
from runtime_state import FrameMetrics, LatestPose

LOG = logging.getLogger("gestur")
ROOT = Path(__file__).resolve().parent
RESTART_REQUESTED = 42


def resolve_model(model_id, models_dir):
    if model_id == "capitell.obj":
        return ROOT / "capitell.obj"
    root = Path(models_dir).resolve()
    candidate = (root / model_id).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError("El modelo debe estar dentro de la biblioteca de Gestur")
    return candidate


class PoseController:
    def __init__(self, obj_path=None, control_system=None, smoothing_config=None,
                 verbose=False, *, config_path=None, models_dir=None,
                 overrides=None, no_camera=False, show_fps=False,
                 benchmark_seconds=0, metrics_path=None):
        self.verbose = verbose
        self.config_path = Path(config_path) if config_path else None
        self.models_dir = Path(models_dir or os.environ.get("GESTUR_MODELS_DIR", ROOT / "data" / "models"))
        self.obj_override = Path(obj_path).expanduser().resolve() if obj_path else None
        self.overrides = overrides or {}
        self.config = self._read_config()
        self._config_stamp = self._stamp()
        self.control_system = control_system or create_control_system(self.config)
        self.mailbox = LatestPose()
        self.metrics = FrameMetrics()
        self.benchmark_seconds = benchmark_seconds
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.no_camera = no_camera
        self.status_path = Path(os.environ["GESTUR_STATUS_PATH"]) if os.environ.get("GESTUR_STATUS_PATH") else None
        self.last_error = None
        self.requested_model = self.config["active_model"]
        self.rendered_model = None
        self.exit_code = 0
        self.pose_tracker = None
        self.visualizer = None
        self._cleaned = False
        self._last_config_check = 0.0
        self._last_log = 0.0
        self._start = None
        from visualizer import ControlledObjViewer
        try:
            model_path = self.obj_override or resolve_model(self.config["active_model"], self.models_dir)
            self.visualizer = ControlledObjViewer(model_path, **self.config["render"], show_fps=show_fps)
            self.rendered_model = str(self.obj_override) if self.obj_override else self.requested_model
        except (ValueError, OSError, RuntimeError) as exc:
            if self.obj_override or self.requested_model == "capitell.obj":
                raise
            self.last_error = f"No se pudo cargar el modelo seleccionado; se muestra el capitell: {exc}"
            LOG.error(self.last_error)
            self.visualizer = ControlledObjViewer(ROOT / "capitell.obj", **self.config["render"], show_fps=show_fps)
            self.rendered_model = "capitell.obj"
        self.visualizer.accept("escape", self.request_stop)
        self.visualizer.taskMgr.add(self._render_tick, "gestur-control", sort=10)

    def _read_config(self):
        config = load_config(self.config_path) if self.config_path else default_config()
        for section, values in self.overrides.items():
            config[section].update(values)
        return validate_config(config)

    def _stamp(self):
        if not self.config_path:
            return None
        try:
            stat = self.config_path.stat()
            return stat.st_ino, stat.st_mtime_ns, stat.st_size
        except FileNotFoundError:
            return None

    def _init_pose_tracker(self):
        from pose_detector import PoseHandTracker
        tracking = self.config["tracking"]
        self.pose_tracker = PoseHandTracker(
            smoothing_time_ms=tracking["smoothing_ms"],
            use_pose=tracking["use_pose"], use_hands=tracking["use_hands"],
            mirror=tracking["mirror"], camera_index=tracking["camera_index"],
            inference_fps=tracking["inference_fps"], hand_fps=tracking["hand_fps"],
            width=tracking["width"], height=tracking["height"], verbose=self.verbose,
        )
        self.pose_tracker.subscribe(self._on_pose_update)

    def _on_pose_update(self, pose_data):
        # Runs in the inference thread: copying to a bounded mailbox is all it does.
        self.mailbox.publish(pose_data)

    def _reload_config(self):
        stamp = self._stamp()
        if stamp == self._config_stamp:
            return
        # Remember rejected versions, too; do not retry a broken file each frame.
        self._config_stamp = stamp
        try:
            candidate = self._read_config()
            self.requested_model = candidate["active_model"]
            if not self.obj_override and candidate["active_model"] != self.config["active_model"]:
                self.visualizer.load_model(resolve_model(candidate["active_model"], self.models_dir))
                self.rendered_model = candidate["active_model"]
            restart_keys = ("antialias_samples", "fullscreen")
            needs_restart = (candidate["tracking"] != self.config["tracking"] or
                             any(candidate["render"][k] != self.config["render"][k] for k in restart_keys))
            if candidate["controls"] != self.config["controls"]:
                self.control_system = create_control_system(candidate)
            self.visualizer.apply_settings(target_fps=candidate["render"]["target_fps"],
                                           hide_cursor=candidate["render"]["hide_cursor"])
            self.config = candidate
            self.last_error = None if self.rendered_model == self.requested_model else self.last_error
            if needs_restart:
                LOG.info("Configuración guardada; reiniciando cámara/ventana")
                self.exit_code = RESTART_REQUESTED
                self.request_stop()
        except (ValueError, OSError, RuntimeError) as exc:
            self.last_error = str(exc)
            LOG.error("No se pudo aplicar la configuración; se conserva la anterior: %s", exc)

    def _write_status(self):
        if not self.status_path:
            return
        report = {"updated_at": time.time(), "selected_model": self.requested_model,
                  "rendered_model": self.rendered_model, "error": self.last_error,
                  "render": self.metrics.summary()}
        temporary = self.status_path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(report) + "\n", encoding="utf-8")
            temporary.replace(self.status_path)
        except OSError:
            LOG.debug("No se pudo escribir el estado del visor", exc_info=True)

    def _render_tick(self, task):
        now = time.monotonic()
        self.metrics.tick(now)
        output = self.control_system.process_input(self.mailbox.read())
        self.visualizer.update_model(**output)
        if now - self._last_config_check >= 1.0:
            self._last_config_check = now
            if self.pose_tracker is not None and getattr(self.pose_tracker, "last_error", None):
                self.last_error = f"Seguimiento detenido: {self.pose_tracker.last_error}"
                self._write_status()
                LOG.error("Seguimiento detenido: %s", self.pose_tracker.last_error)
                self.exit_code = 1
                self.request_stop()
                return task.done
            self._reload_config()
            self._write_status()
        if self.verbose and now - self._last_log >= 2.0:
            self._last_log = now
            LOG.info("Render: %s", self.metrics.summary())
        if self.benchmark_seconds and self._start is not None and now - self._start >= self.benchmark_seconds:
            self.request_stop()
        return task.cont

    def request_stop(self, *_):
        if self.visualizer:
            self.visualizer.taskMgr.stop()

    def run(self):
        previous_handlers = {}
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, self.request_stop)
            if not self.no_camera:
                self._init_pose_tracker()
                self.pose_tracker.run()
            self._start = time.monotonic()
            self.visualizer.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        return self.exit_code

    def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        if self.pose_tracker:
            self.pose_tracker.stop()
        report = self.metrics.summary()
        report["model"] = self.visualizer.model_path if self.visualizer else None
        report["render_settings"] = self.config["render"]
        report["tracking_settings"] = self.config["tracking"]
        report["camera_enabled"] = not self.no_camera
        if self.pose_tracker and hasattr(self.pose_tracker, "get_metrics"):
            report["tracking"] = self.pose_tracker.get_metrics()
        if self.visualizer:
            self.visualizer.destroy()
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if self.benchmark_seconds:
            print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gestur: control 3D fluido para Raspberry Pi 5")
    parser.add_argument("obj", nargs="?", help="Modelo local; sin argumento se usa la biblioteca/configuración")
    parser.add_argument("--config", default=os.environ.get("GESTUR_CONFIG"), help="JSON compartido con el portal")
    parser.add_argument("--models-dir", help="Directorio de modelos subidos")
    parser.add_argument("--control-preset", choices=["default", "continuous"], default="continuous",
                        help="Alias conservados para la configuración del capitell")
    parser.add_argument("--hands", action="store_true", help="Activar también el modelo Lite de manos")
    parser.add_argument("--windowed", action="store_true", help="Ejecutar en una ventana")
    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--no-camera", action="store_true", help="Medir o revisar sólo el renderizador")
    parser.add_argument("--benchmark-seconds", type=float, default=0)
    parser.add_argument("--metrics", help="Guardar medición JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    if args.benchmark_seconds < 0:
        parser.error("--benchmark-seconds debe ser positivo")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    overrides = {}
    if args.hands:
        overrides["tracking"] = {"use_hands": True}
    if args.windowed:
        overrides["render"] = {"fullscreen": False}
    try:
        while True:
            controller = PoseController(
                args.obj, verbose=args.verbose, config_path=args.config, models_dir=args.models_dir,
                overrides=overrides, no_camera=args.no_camera, show_fps=args.show_fps,
                benchmark_seconds=args.benchmark_seconds, metrics_path=args.metrics,
            )
            code = controller.run()
            if code != RESTART_REQUESTED:
                return code
    except (ValueError, OSError, RuntimeError) as exc:
        LOG.error("No se pudo iniciar Gestur: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
