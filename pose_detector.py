"""Bounded, offline MediaPipe Lite tracking for the Raspberry Pi 5.

Capture drains the camera independently from inference. There is exactly one
pending frame: inference always uses the most recent frame and never queues old
frames. The pose and hand models have separate cadence budgets.
Their combined inference work shares a wall-time duty limit; absent parts use
slower presence probes and resume their requested cadence when detected again.
"""
import logging
import math
from pathlib import Path
import threading
import time

from primary_person import PrimaryPersonLock
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

    def image(self, frame, mirror, region=None):
        if mirror:
            frame = self.cv2.flip(frame, 1)
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        if region is not None:
            # Mask the converted buffer in place. Keeping the original canvas
            # preserves VIDEO tracking coordinates, gesture scale and aspect.
            # The captured BGR frame remains untouched for the camera thread.
            height, width = rgb.shape[:2]
            left, top, right, bottom = region
            left = max(0, min(width, math.floor(left * width)))
            right = max(left, min(width, math.ceil(right * width)))
            top = max(0, min(height, math.floor(top * height)))
            bottom = max(top, min(height, math.ceil(bottom * height)))
            rgb[:top] = 0
            rgb[bottom:] = 0
            rgb[top:bottom, :left] = 0
            rgb[top:bottom, right:] = 0
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
                 detection_timeout_ms=250, head_scale_min=.02, head_scale_max=.20,
                 inference_duty=.6, idle_fps=3, idle_after_seconds=2, pose_parts=('head',)):
        """Keep the legacy callback schema, adding hand rotation/pinch/gesture.

        Construction opens no camera or models. ``run`` allocates resources and
        can be called again after ``stop``. Frame timestamps are monotonic seconds;
        MediaPipe receives strictly increasing integer milliseconds per session.
        ``response_time_ms`` remains a compatibility alias for pose cadence.
        ``inference_duty`` reserves gaps between complete inference cycles; it
        does not cap native CPU threads or promise an operating-system CPU rate.
        After ``idle_after_seconds`` without a requested pose part / hand,
        each model independently probes at up to ``idle_fps``. ``pose_parts``
        selects head/torso presence relevant to the enabled controls; it does
        not allocate another network or change the published pose schema.
        """
        if inference_fps is None:
            inference_fps = 1000.0 / max(1.0, response_time_ms)
        if not all(math.isfinite(float(value)) and float(value) > 0
                   for value in (inference_fps, hand_fps, width, height, inference_duty, idle_fps)):
            raise ValueError('Frecuencias y resolución deben ser mayores que cero.')
        if not all(math.isfinite(float(value)) for value in
                   (smoothing_time_ms, detection_timeout_ms, visibility_threshold, head_scale_min, head_scale_max,
                    idle_after_seconds)):
            raise ValueError('Los parámetros del tracker deben ser finitos.')
        if not 0 <= visibility_threshold <= 1 or head_scale_max <= head_scale_min:
            raise ValueError('Confianza o rango de escala no válido.')
        if inference_duty > 1 or idle_after_seconds < 0:
            raise ValueError('El presupuesto de inferencia debe ser como máximo 1 y la espera no negativa.')
        if (not isinstance(pose_parts, (tuple, list))
                or any(part not in ('head', 'torso') for part in pose_parts)
                or (use_pose and not pose_parts)):
            raise ValueError('Selecciona cabeza o torso para la presencia del modelo Pose.')
        self.use_pose, self.use_hands = bool(use_pose), bool(use_hands)
        self.pose_parts = tuple(pose_parts)
        self.mirror, self.invert_hands, self.verbose = bool(mirror), bool(invert_hands), bool(verbose)
        self.camera_index, self.width, self.height = camera_index, int(width), int(height)
        self.inference_fps, self.hand_fps = float(inference_fps), float(hand_fps)
        self.response_time = 1 / self.inference_fps
        self.smoothing_time = max(0, smoothing_time_ms / 1000)
        self.inference_duty = float(inference_duty)
        self.idle_fps, self.idle_after_seconds = float(idle_fps), float(idle_after_seconds)
        self._base_detection_timeout = max(0, detection_timeout_ms / 1000)
        self._rates = {name: rate for name, enabled, rate in
                       (('pose', self.use_pose, self.inference_fps), ('hand', self.use_hands, self.hand_fps))
                       if enabled}
        # A slow configured cadence must not expire between scheduled inferences.
        self.detection_timeout = max(self._base_detection_timeout,
                                     2 / min(self._rates.values(), default=math.inf))
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
        self._waiting_for_frame = False
        self._scheduled_rates = self._rates.copy()
        self._model_capture_times = {}
        self._expiry_timeouts = {name: max(self._base_detection_timeout, 2 / rate)
                                 for name, rate in self._rates.items()}
        self._metrics = {}
        self._filter = TrackingFilter(self.smoothing_time, self.detection_timeout)
        self._primary = PrimaryPersonLock(use_pose=self.use_pose,
                                          visibility_threshold=self.visibility_threshold)
        self._primary_generation = self._primary.generation

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
        cap.set(cv2.CAP_PROP_FPS, max(self._rates.values(), default=1))
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
            self._waiting_for_frame = False
            self._scheduled_rates = self._rates.copy()
            self._model_capture_times = {}
            self._expiry_timeouts = {name: max(self._base_detection_timeout, 2 / rate)
                                     for name, rate in self._rates.items()}
            self._filter = TrackingFilter(self.smoothing_time, self.detection_timeout)
            self._primary = PrimaryPersonLock(use_pose=self.use_pose,
                                              visibility_threshold=self.visibility_threshold)
            self._primary_generation = self._primary.generation
            self.current_data = empty_data()
            self._metrics = dict(captured_frames=0, pose_frames=0, hand_frames=0,
                                 capture_failures=0, inference_ms=0.0, frame_age_ms=0.0,
                                 inference_wall_seconds=0.0, inference_duty_limit=self.inference_duty,
                                 budget_pauses=0, budget_pause_seconds=0.0,
                                 pose_scheduled_fps=self._rates.get('pose', 0),
                                 hand_scheduled_fps=self._rates.get('hand', 0),
                                 pose_idle=False, hand_idle=False,
                                 primary_person_state='searching', primary_person_generation=0,
                                 rejected_pose_frames=0, rejected_hand_candidates=0)
            if not self.use_pose and not self.use_hands:
                return
            self.cap = None
            try:
                # A missing camera must not load neural networks on every retry.
                self.cap = self._open_camera()
                self._backend = self._build_backend()
                self.pose_model, self.hands_model = self._backend.pose, self._backend.hands
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
                    if self._waiting_for_frame:
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
        next_due = dict.fromkeys(self._rates, 0.0)
        last_detected = dict.fromkeys(self._rates, time.monotonic())
        budget_ready = 0.0
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if self._expire_tracking(now):
                    self._publish()
                if self._stop_event.is_set():
                    break
                cadence_ready = min(next_due.values())
                ready = max(cadence_ready, budget_ready)
                with self._condition:
                    latest = self._latest_frame
                    fresh = latest is not None and latest[0] != sequence
                    if now < ready or not fresh:
                        # New frames cannot make a model ready before its budget.
                        # Only wake on capture when a fresh frame is all we need.
                        self._waiting_for_frame = now >= ready
                        timeout = ready - now if now < ready else .5
                        timeout = min(timeout, self._next_expiry_delay(now))
                        self._condition.wait(timeout=max(.001, timeout))
                        waited_until = time.monotonic()
                        budget_wait = max(0, min(waited_until, budget_ready) - max(now, cadence_ready))
                        if budget_wait:
                            with self._state_lock:
                                self._metrics['budget_pauses'] += 1
                                self._metrics['budget_pause_seconds'] += budget_wait
                        latest = None
                    self._waiting_for_frame = False
                if latest is None:
                    continue
                sequence, captured_at, frame = latest
                now = time.monotonic()
                if self._stop_event.is_set():
                    break
                if now-captured_at > self.detection_timeout:
                    continue
                due_pose = 'pose' in next_due and now >= next_due['pose']
                due_hand = 'hand' in next_due and now >= next_due['hand']
                started = now
                timestamp_ms = max(timestamp_ms+1, int(captured_at*1000))
                region = self._primary.region(captured_at)
                self._sync_primary_generation()
                image = (self._backend.image(frame, self.mirror, region)
                         if region is not None else self._backend.image(frame, self.mirror))
                aspect = frame.shape[1] / frame.shape[0]
                if due_pose:
                    result = self.pose_model.detect_for_video(image, timestamp_ms)
                    features = self._update_pose(result, captured_at, aspect)
                    detected = any(features[part]['detected'] for part in self.pose_parts)
                    self._schedule_model('pose', detected, captured_at,
                                         started, time.monotonic(), next_due, last_detected)
                    with self._state_lock:
                        self._metrics['pose_frames'] += 1
                if due_hand:
                    hand_started = time.monotonic()
                    result = self.hands_model.detect_for_video(image, timestamp_ms)
                    self._update_hands(result, captured_at, aspect)
                    detected = any(self._filter.data[key]['detected'] for key in ('left_hand', 'right_hand'))
                    self._schedule_model('hand', detected, captured_at,
                                         hand_started, time.monotonic(), next_due, last_detected)
                    with self._state_lock:
                        self._metrics['hand_frames'] += 1
                completed = time.monotonic()
                elapsed = completed - started
                # One combined duty budget covers conversion, geometry and both
                # native models. It is wall time, not a claim about CPU usage.
                pause = elapsed * (1 / self.inference_duty - 1)
                budget_ready = completed + pause
                inferred = (['pose'] if due_pose else []) + (['hand'] if due_hand else [])
                self._update_expiry_limits(inferred, captured_at, elapsed)
                self._expire_tracking(completed)
                with self._state_lock:
                    self._metrics['inference_ms'] = elapsed*1000
                    self._metrics['inference_wall_seconds'] += elapsed
                    self._metrics['frame_age_ms'] = (completed-captured_at)*1000
                    self._metrics['primary_person_state'] = self._primary.state
                    self._metrics['primary_person_generation'] = self._primary.generation
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

    def _schedule_model(self, name, detected, captured_at, started, completed, next_due, last_detected):
        if detected:
            last_detected[name] = captured_at
        idle = not detected and completed - last_detected[name] >= self.idle_after_seconds
        rate = min(self._rates[name], self.idle_fps) if idle else self._rates[name]
        self._scheduled_rates[name] = rate
        next_due[name] = max(started + 1 / rate, completed)
        with self._state_lock:
            self._metrics[f'{name}_scheduled_fps'] = rate
            self._metrics[f'{name}_idle'] = idle

    def _update_expiry_limits(self, inferred, captured_at, elapsed):
        for name in inferred:
            previous = self._model_capture_times.get(name, captured_at)
            observed_period = max(0, captured_at - previous)
            self._model_capture_times[name] = captured_at
            period = max(1 / self._scheduled_rates[name], elapsed / self.inference_duty, observed_period)
            self._expiry_timeouts[name] = max(self._base_detection_timeout, min(.8, 2 * period))
        # Only a new inference from a model may shorten its expiry. A fast hand
        # cycle must not invalidate a head held through a slower pose cadence.
        self._filter.timeout = max(self._expiry_timeouts.values())

    def _expire_tracking(self, now):
        """Expire each model independently; idle hands cannot extend head holds."""
        changed = False
        for key, sample in self._filter.data.items():
            if not sample['detected']:
                continue
            name = 'hand' if 'hand' in key else 'pose'
            seen = self._filter.last_seen[key]
            if seen is None or now - seen > self._expiry_timeouts.get(name, self.detection_timeout):
                self._filter.data[key] = empty_part(hand=name == 'hand', head=key == 'head')
                changed = True
        return changed

    def _next_expiry_delay(self, now):
        deadlines = []
        for key, sample in self._filter.data.items():
            if sample['detected']:
                name = 'hand' if 'hand' in key else 'pose'
                seen = self._filter.last_seen[key]
                if seen is not None:
                    deadlines.append(seen + self._expiry_timeouts.get(name, self.detection_timeout))
        return max(.001, min(deadlines) - now) if deadlines else math.inf

    def _update_hands(self, result, timestamp, aspect):
        samples = {'left_hand': empty_part(hand=True), 'right_hand': empty_part(hand=True)}
        scores = {'left_hand': -1, 'right_hand': -1}
        candidates = []
        for normalized, world, categories in zip(result.hand_landmarks,
                result.hand_world_landmarks, result.handedness):
            if not categories:
                continue
            category = max(categories, key=lambda c: c.score)
            label = anatomical_hand(category.category_name, self.mirror)
            if label is None or category.score < self.visibility_threshold:
                continue
            candidates.append((normalized, world, category, label))
        selected = self._primary.select_hands(
            [(points, label, category.score) for points, _, category, label in candidates],
            timestamp, aspect)
        self._sync_primary_generation()
        with self._state_lock:
            self._metrics['rejected_hand_candidates'] = (
                self._metrics.get('rejected_hand_candidates', 0) + len(candidates) - len(selected))
        for index in selected:
            normalized, world, category, label = candidates[index]
            # Ownership uses anatomical labels; optional control inversion must
            # not change which physical person's hand is accepted.
            if self.invert_hands:
                label = 'Left' if label == 'Right' else 'Right'
            key = label.lower() + '_hand'
            if category.score > scores[key]:
                samples[key] = hand_features(normalized, world, category.category_name, aspect)
                scores[key] = category.score
        for key, sample in samples.items():
            self._filter.update(key, sample, timestamp)

    def _update_pose(self, result, timestamp, aspect):
        normalized = result.pose_landmarks[0] if result.pose_landmarks else []
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
        accepted = self._primary.update_pose(normalized, timestamp, aspect)
        self._sync_primary_generation()
        features = (pose_features(normalized, world, self.visibility_threshold, aspect,
                                  self.head_scale_min, self.head_scale_max)
                    if accepted else {'head': empty_part(head=True), 'torso': empty_part()})
        for key, sample in features.items():
            self._filter.update(key, sample, timestamp)
        # Hands have their own cadence and ownership checks. An intermittent
        # pose must not invalidate a still-visible, already associated hand.
        # Changing owner clears every part in _sync_primary_generation.
        with self._state_lock:
            self._metrics['rejected_pose_frames'] = (
                self._metrics.get('rejected_pose_frames', 0) + int(bool(normalized) and not accepted))
        return features

    def _sync_primary_generation(self):
        """Never smooth samples from different people together."""
        if self._primary.generation != self._primary_generation:
            self._primary_generation = self._primary.generation
            self._filter = TrackingFilter(self.smoothing_time, self.detection_timeout)

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
