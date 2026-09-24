"""Bounded, offline MediaPipe Lite tracking for the Raspberry Pi 5.

Capture drains the camera independently from inference. There is exactly one
pending frame: inference always uses the most recent frame and never queues old
frames. The pose and hand models have separate cadence budgets.
"""
import logging
import math
from pathlib import Path
import threading
import time

from tracking_geometry import (TrackingFilter, anatomical_hand, empty_data,
                               empty_part, hand_features, pose_features)

_LOG = logging.getLogger(__name__)
_MODEL_DIR = Path(__file__).resolve().parent / 'tracking_models'


class _MediaPipeBackend:
    def __init__(self, model_dir, use_pose, use_hands, confidence):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.cv2, self.mp = cv2, mp
        self.pose = self.hands = None
        cv2.setNumThreads(1)
        try:
            def options(filename):
                path = Path(model_dir) / filename
                if not path.is_file():
                    raise FileNotFoundError(
                        f'Falta {path}. Ejecuta python scripts/provision_models.py '
                        'durante la instalación (el seguimiento funciona sin Internet).')
                return python.BaseOptions(model_asset_path=str(path),
                                          delegate=python.BaseOptions.Delegate.CPU)
            if use_pose:
                self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=options('pose_landmarker_lite.task'),
                    running_mode=vision.RunningMode.VIDEO, num_poses=1,
                    min_pose_detection_confidence=confidence,
                    min_pose_presence_confidence=confidence,
                    min_tracking_confidence=confidence, output_segmentation_masks=False))
            if use_hands:
                self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=options('hand_landmarker_lite.task'),
                    running_mode=vision.RunningMode.VIDEO, num_hands=2,
                    min_hand_detection_confidence=confidence,
                    min_hand_presence_confidence=confidence,
                    min_tracking_confidence=confidence))
        except Exception:
            self.close()
            raise

    def image(self, frame, mirror):
        if mirror:
            frame = self.cv2.flip(frame, 1)
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

    def close(self):
        for name in ('pose', 'hands'):
            model = getattr(self, name, None)
            if model is not None:
                try:
                    model.close()
                finally:
                    setattr(self, name, None)


class PoseHandTracker:
    def __init__(self, response_time_ms=50, smoothing_time_ms=60,
                 use_pose=True, use_hands=True, mirror=False,
                 invert_hands=False, verbose=False, *, camera_index=0,
                 inference_fps=None, hand_fps=15, width=640, height=480,
                 model_dir=None, visibility_threshold=.5,
                 detection_timeout_ms=250, head_scale_min=.02, head_scale_max=.20):
        """Keep the legacy callback schema, adding hand rotation/pinch/gesture.

        Construction opens no camera or models. ``run`` allocates resources and
        can be called again after ``stop``. Frame timestamps are monotonic seconds;
        MediaPipe receives strictly increasing integer milliseconds per session.
        ``response_time_ms`` remains a compatibility alias for pose cadence.
        """
        if inference_fps is None:
            inference_fps = 1000.0 / max(1.0, response_time_ms)
        if not all(math.isfinite(float(value)) and float(value) > 0
                   for value in (inference_fps, hand_fps, width, height)):
            raise ValueError('Frecuencias y resolución deben ser mayores que cero.')
        if not all(math.isfinite(float(value)) for value in
                   (smoothing_time_ms, detection_timeout_ms, visibility_threshold, head_scale_min, head_scale_max)):
            raise ValueError('Los parámetros del tracker deben ser finitos.')
        if not 0 <= visibility_threshold <= 1 or head_scale_max <= head_scale_min:
            raise ValueError('Confianza o rango de escala no válido.')
        self.use_pose, self.use_hands = bool(use_pose), bool(use_hands)
        self.mirror, self.invert_hands, self.verbose = bool(mirror), bool(invert_hands), bool(verbose)
        self.camera_index, self.width, self.height = camera_index, int(width), int(height)
        self.inference_fps, self.hand_fps = float(inference_fps), float(hand_fps)
        self.response_time = 1 / self.inference_fps
        self.smoothing_time = max(0, smoothing_time_ms / 1000)
        # A slow configured cadence must not expire between scheduled inferences.
        self.detection_timeout = max(detection_timeout_ms / 1000,
                                     2 / min(self.inference_fps, self.hand_fps))
        self.visibility_threshold = visibility_threshold
        self.head_scale_min, self.head_scale_max = head_scale_min, head_scale_max
        self.model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        self.listeners = []
        self.current_data = empty_data()
        self.running = False
        self.thread = self.capture_thread = None
        self.cap = self._backend = None
        self.pose_model = self.hands_model = None
        self.last_error = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._latest_frame = None
        self._sequence = 0
        self._metrics = {}
        self._filter = TrackingFilter(self.smoothing_time, self.detection_timeout)

    def _open_camera(self):
        import cv2
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f'No se pudo acceder a la cámara {self.camera_index}.')
        # Not all backends implement buffer size; the one-frame slot also bounds
        # application memory and latency if the driver ignores this property.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, max(self.inference_fps, self.hand_fps))
        return cap

    def _build_backend(self):
        return _MediaPipeBackend(self.model_dir, self.use_pose, self.use_hands,
                                self.visibility_threshold)

    def run(self):
        with self._lifecycle_lock:
            if self.running:
                return
            if any(t and t.is_alive() for t in (self.thread, self.capture_thread)):
                raise RuntimeError('La sesión anterior todavía está cerrando la cámara.')
            self._stop_event.clear()
            self.last_error = None
            self._latest_frame, self._sequence = None, 0
            self._filter = TrackingFilter(self.smoothing_time, self.detection_timeout)
            self.current_data = empty_data()
            self._metrics = dict(captured_frames=0, pose_frames=0, hand_frames=0,
                                 capture_failures=0, inference_ms=0.0, frame_age_ms=0.0)
            if not self.use_pose and not self.use_hands:
                return
            try:
                self._backend = self._build_backend()
                self.pose_model, self.hands_model = self._backend.pose, self._backend.hands
                self.cap = self._open_camera()
                self.running = True
                self.capture_thread = threading.Thread(target=self._capture_loop,
                                                        name='gestur-camera', daemon=True)
                self.thread = threading.Thread(target=self._process_loop,
                                               name='gestur-inference', daemon=True)
                self.capture_thread.start()
                self.thread.start()
            except Exception:
                self.running = False
                self._stop_event.set()
                if self.capture_thread and self.capture_thread.is_alive():
                    self.capture_thread.join(timeout=2)
                elif self.cap is not None:
                    self.cap.release()
                self._close_backend()
                raise

    def _capture_loop(self):
        failures = 0
        try:
            while not self._stop_event.is_set():
                success, frame = self.cap.read()
                timestamp = time.monotonic()
                if not success or frame is None:
                    failures += 1
                    with self._state_lock:
                        self._metrics['capture_failures'] += 1
                    if failures >= 10:
                        raise RuntimeError('La cámara no entrega imágenes (10 lecturas fallidas).')
                    self._stop_event.wait(.05)
                    continue
                failures = 0
                with self._condition:
                    self._sequence += 1
                    self._latest_frame = (self._sequence, timestamp, frame)
                    self._condition.notify()
                with self._state_lock:
                    self._metrics['captured_frames'] += 1
        except Exception as error:
            self.last_error = error
            _LOG.error('Error de captura: %s', error)
            self._stop_event.set()
        finally:
            self.cap.release()
            with self._condition:
                self._condition.notify_all()

    def _process_loop(self):
        sequence, timestamp_ms = 0, -1
        next_pose = next_hand = 0.0
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                due_pose = self.use_pose and now >= next_pose
                due_hand = self.use_hands and now >= next_hand
                due_times = ([next_pose] if self.use_pose else []) + ([next_hand] if self.use_hands else [])
                delay = max(0, min(due_times)-now)
                with self._condition:
                    latest = self._latest_frame
                    fresh = latest is not None and latest[0] != sequence
                    if not fresh or not (due_pose or due_hand):
                        self._condition.wait(timeout=min(.05, delay) if delay > 0 else .05)
                        latest = None
                if latest is None:
                    self._filter.expire(time.monotonic())
                    self._publish()
                    continue
                sequence, captured_at, frame = latest
                now = time.monotonic()
                if now-captured_at > self.detection_timeout:
                    self._filter.expire(now)
                    self._publish()
                    continue
                started = now
                timestamp_ms = max(timestamp_ms+1, int(captured_at*1000))
                image = self._backend.image(frame, self.mirror)
                aspect = frame.shape[1] / frame.shape[0]
                if due_pose:
                    result = self.pose_model.detect_for_video(image, timestamp_ms)
                    normalized = result.pose_landmarks[0] if result.pose_landmarks else []
                    world = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
                    features = pose_features(normalized, world, self.visibility_threshold, aspect,
                                             self.head_scale_min, self.head_scale_max)
                    for key, sample in features.items():
                        self._filter.update(key, sample, captured_at)
                    next_pose = max(started + 1 / self.inference_fps, time.monotonic())
                    with self._state_lock:
                        self._metrics['pose_frames'] += 1
                if due_hand:
                    hand_started = time.monotonic()
                    result = self.hands_model.detect_for_video(image, timestamp_ms)
                    self._update_hands(result, captured_at, aspect)
                    next_hand = max(hand_started + 1 / self.hand_fps, time.monotonic())
                    with self._state_lock:
                        self._metrics['hand_frames'] += 1
                completed = time.monotonic()
                self._filter.expire(completed)
                with self._state_lock:
                    self._metrics['inference_ms'] = (completed-started)*1000
                    self._metrics['frame_age_ms'] = (completed-captured_at)*1000
                self._publish()
        except Exception as error:
            self.last_error = error
            _LOG.exception('Error en el seguimiento: %s', error)
        finally:
            self._stop_event.set()
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2)
            self._close_backend()
            self._filter = TrackingFilter(self.smoothing_time, self.detection_timeout)
            self._publish()
            self.running = False

    def _update_hands(self, result, timestamp, aspect):
        samples = {'left_hand': empty_part(hand=True), 'right_hand': empty_part(hand=True)}
        scores = {'left_hand': -1, 'right_hand': -1}
        for normalized, world, categories in zip(result.hand_landmarks,
                result.hand_world_landmarks, result.handedness):
            if not categories:
                continue
            category = max(categories, key=lambda c: c.score)
            label = anatomical_hand(category.category_name, self.mirror, self.invert_hands)
            if label is None or category.score < self.visibility_threshold:
                continue
            key = label.lower() + '_hand'
            if category.score > scores[key]:
                samples[key] = hand_features(normalized, world, category.category_name, aspect)
                scores[key] = category.score
        for key, sample in samples.items():
            self._filter.update(key, sample, timestamp)

    def _close_backend(self):
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                _LOG.exception('No se pudo cerrar MediaPipe correctamente.')
            self._backend = None
        self.pose_model = self.hands_model = None

    def _publish(self):
        snapshot = self._filter.snapshot()
        with self._state_lock:
            self.current_data = snapshot
            listeners = tuple(self.listeners)
        for callback in listeners:
            try:
                callback({key: value.copy() for key, value in snapshot.items()})
            except Exception:
                _LOG.exception('Error en callback de seguimiento')

    def get_current_data(self):
        with self._state_lock:
            return {key: value.copy() for key, value in self.current_data.items()}

    def get_metrics(self):
        with self._state_lock:
            return self._metrics.copy()

    def subscribe(self, listener_fn):
        if callable(listener_fn):
            with self._state_lock:
                if listener_fn not in self.listeners:
                    self.listeners.append(listener_fn)

    def unsubscribe(self, listener_fn):
        with self._state_lock:
            if listener_fn in self.listeners:
                self.listeners.remove(listener_fn)

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        # The capture thread owns release(); avoid closing a camera while a
        # driver is inside read(). Do not close models while inference uses them.
        current = threading.current_thread()
        for thread in (self.thread, self.capture_thread):
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=3)
        if not any(t and t.is_alive() for t in (self.thread, self.capture_thread)):
            self.running = False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tracker = PoseHandTracker(mirror=True)
    tracker.subscribe(lambda data: print(data))
    try:
        tracker.run()
        while tracker.running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
