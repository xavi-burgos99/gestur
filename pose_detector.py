import warnings
import logging

# Configurar logging para silenciar warnings conocidos
logging.getLogger('mmengine').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Failed to search registry.*")
warnings.filterwarnings("ignore", message=".*TorchScript.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

import time
import threading
import cv2
import numpy as np
import torch
import math
from typing import List, Dict, Optional, Tuple
from mmpose.apis import MMPoseInferencer
from collections import deque


class SmoothedValue:
    def __init__(self, window_ms=500, emit_interval_ms=10):
        self.window_ms = window_ms
        self.emit_interval_ms = emit_interval_ms
        self.values = deque()  # stores tuples (timestamp, value)
        self.last_emit_time = 0

    def add_value(self, value):
        now = time.time() * 1000  # ms
        self.values.append((now, value))
        # Remove old values outside the window
        while self.values and self.values[0][0] < now - self.window_ms:
            self.values.popleft()

    def get_smoothed_value(self):
        now = time.time() * 1000
        if now - self.last_emit_time < self.emit_interval_ms:
            return None  # Not time to emit yet
        self.last_emit_time = now

        if not self.values:
            return None

        # Average all values in the window
        vals = np.array([v for _, v in self.values])
        if vals.ndim == 1:
            return np.mean(vals)
        else:
            return np.mean(vals, axis=0)


class PoseDetector:
    def __init__(self, verbose: float = 0, smoothing_window_ms: int = 500,
                 emit_interval_ms: int = 10, hold_time_sec: float = 3.0):
        self.verbose = verbose
        self.smoothing_window_ms = smoothing_window_ms
        self.emit_interval_ms = emit_interval_ms
        self.hold_time_sec = hold_time_sec

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Inicializar inferenciadores
        self.head_inferencer = MMPoseInferencer(pose2d='body', device=self.device)
        self.hand_inferencer = MMPoseInferencer(pose2d='hand', device=self.device)

        # Cámara compartida
        self.camera = None
        self.camera_lock = threading.Lock()

        # Smoothers para cada tipo de dato
        self.head_smoother = SmoothedValue(window_ms=smoothing_window_ms, emit_interval_ms=emit_interval_ms)
        self.left_hand_smoother = SmoothedValue(window_ms=smoothing_window_ms, emit_interval_ms=emit_interval_ms)
        self.right_hand_smoother = SmoothedValue(window_ms=smoothing_window_ms, emit_interval_ms=emit_interval_ms)

        # Sistema de hold para mantener últimos valores
        self.last_head_detection_time = 0
        self.last_left_hand_detection_time = 0
        self.last_right_hand_detection_time = 0
        self.last_head_data = None
        self.last_left_hand_data = None
        self.last_right_hand_data = None

        # Datos compartidos y locks
        self.data_lock = threading.Lock()
        self.pose_data = {
            'head': {'x': 0.0, 'y': 0.0},
            'hands': {
                'left': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
                'right': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            }
        }

        # Control de hilos
        self.running = False
        self.detection_thread = None

        # Índices de keypoints de manos para cálculo de orientación
        self.hand_keypoint_indices = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20,
            'middle_mcp': 9,
            'index_mcp': 5,
        }

    def _get_face_keypoint_ids(self, meta) -> List[int]:
        """Obtiene los IDs de los keypoints faciales relevantes."""
        if not meta or 'keypoint_labels' not in meta:
            return [0, 1, 2]

        target_keypoints = {'nose', 'left_eye', 'right_eye'}
        face_ids = [
            i for i, label in enumerate(meta['keypoint_labels'])
            if label in target_keypoints
        ]
        return face_ids if face_ids else [0, 1, 2]

    def _calculate_hand_rotation(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calcula los ángulos de rotación de la mano (yaw, pitch, roll)."""
        try:
            # Obtener puntos clave
            wrist = keypoints[self.hand_keypoint_indices['wrist']][:2]
            middle_mcp = keypoints[self.hand_keypoint_indices['middle_mcp']][:2]
            middle_tip = keypoints[self.hand_keypoint_indices['middle_tip']][:2]
            index_mcp = keypoints[self.hand_keypoint_indices['index_mcp']][:2]

            # Vector principal: de muñeca a base del dedo medio
            main_vector = middle_mcp - wrist

            # Vector del dedo medio: de base a punta
            finger_vector = middle_tip - middle_mcp

            # Vector lateral: de base del medio a base del índice
            lateral_vector = index_mcp - middle_mcp

            # Calcular YAW (rotación en plano horizontal)
            yaw = math.degrees(math.atan2(main_vector[1], main_vector[0]))

            # Calcular PITCH (inclinación hacia arriba/abajo del dedo)
            finger_length = np.linalg.norm(finger_vector)
            if finger_length > 0:
                horizontal_component = np.sqrt(finger_vector[0] ** 2 + finger_vector[1] ** 2)
                if horizontal_component > 0:
                    pitch = math.degrees(math.atan2(-finger_vector[1], horizontal_component))
                else:
                    pitch = 0.0
            else:
                pitch = 0.0

            # Calcular ROLL (rotación de la mano alrededor del eje del brazo)
            lateral_length = np.linalg.norm(lateral_vector)
            if lateral_length > 0:
                roll = math.degrees(
                    math.atan2(lateral_vector[1], lateral_vector[0]) - math.atan2(main_vector[1], main_vector[0]))
            else:
                roll = 0.0

            # Normalizar ángulos a rango [-180, 180]
            yaw = ((yaw + 180) % 360) - 180
            pitch = np.clip(pitch, -90, 90)
            roll = ((roll + 180) % 360) - 180

            return {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll)
            }

        except Exception as e:
            print(f"Error calculando rotación de mano: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

    def _process_head_detection(self, frame) -> Optional[Dict[str, float]]:
        """Procesa la detección de cabeza y devuelve coordenadas normalizadas."""
        if not self.camera:
            return None

        W = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cx, cy = W / 2, H / 2

        for output in self.head_inferencer(frame):
            predictions = output['predictions'][0]
            if not predictions:
                continue

            keypoints = np.asarray(predictions[0]['keypoints'])

            # Verificar confianza si está disponible
            if keypoints.shape[1] == 3 and keypoints[:, 2].mean() < 0.25:
                continue

            # Obtener keypoints faciales
            face_ids = self._get_face_keypoint_ids(output.get('dataset_meta', {}))
            face_keypoints = keypoints[face_ids]

            # Verificar confianza de keypoints faciales
            if face_keypoints.shape[1] == 3 and face_keypoints[:, 2].mean() < 0.25:
                continue

            # Calcular centro de la cara
            x_mean = face_keypoints[:, 0].mean()
            y_mean = face_keypoints[:, 1].mean()

            # Normalizar coordenadas (-1 a 1)
            head_x = (x_mean - cx) / cx
            head_y = (y_mean - cy) / cy

            return {
                'x': float(np.clip(head_x, -1, 1)),
                'y': float(np.clip(head_y, -1, 1))
            }

        return None

    def _process_hand_detection(self, frame) -> Optional[Dict]:
        """Procesa la detección de manos y calcula orientaciones."""
        try:
            hands_data = {
                'left': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
                'right': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            }

            for output in self.hand_inferencer(frame):
                predictions = output['predictions'][0]
                if not predictions:
                    continue

                for i, pred in enumerate(predictions):
                    keypoints = np.asarray(pred['keypoints'])

                    # Verificar que tenemos 21 keypoints
                    if keypoints.shape[0] != 21:
                        continue

                    # Verificar confianza mínima
                    if keypoints.shape[1] == 3:
                        confidence = keypoints[:, 2].mean()
                        if confidence < 0.3:
                            continue

                    # Calcular rotaciones
                    rotations = self._calculate_hand_rotation(keypoints)

                    # Determinar si es mano izquierda o derecha
                    wrist_x = keypoints[0, 0]
                    frame_center_x = frame.shape[1] / 2

                    if wrist_x < frame_center_x:
                        hand_type = 'right'
                    else:
                        hand_type = 'left'

                    hands_data[hand_type] = {
                        'detected': True,
                        'yaw': rotations['yaw'],
                        'pitch': rotations['pitch'],
                        'roll': rotations['roll']
                    }

            return hands_data

        except Exception as e:
            print(f"Error en detección de manos: {e}")
            return None

    def _update_smoothed_data(self, head_result, hand_result):
        """Actualiza los datos suavizados y aplica el sistema de hold."""
        now = time.time()

        # Procesar datos de cabeza
        if head_result:
            self.last_head_detection_time = now
            self.last_head_data = head_result
            self.head_smoother.add_value(np.array([head_result['x'], head_result['y']]))

        # Procesar datos de manos
        if hand_result:
            left_hand = hand_result.get('left', {})
            right_hand = hand_result.get('right', {})

            if left_hand.get('detected', False):
                self.last_left_hand_detection_time = now
                self.last_left_hand_data = left_hand
                self.left_hand_smoother.add_value(np.array([
                    left_hand['yaw'], left_hand['pitch'], left_hand['roll']
                ]))

            if right_hand.get('detected', False):
                self.last_right_hand_detection_time = now
                self.last_right_hand_data = right_hand
                self.right_hand_smoother.add_value(np.array([
                    right_hand['yaw'], right_hand['pitch'], right_hand['roll']
                ]))

        # Obtener valores suavizados
        smoothed_head = self.head_smoother.get_smoothed_value()
        smoothed_left_hand = self.left_hand_smoother.get_smoothed_value()
        smoothed_right_hand = self.right_hand_smoother.get_smoothed_value()

        # Aplicar sistema de hold y componer salida
        output = {
            'head': {'x': 0.0, 'y': 0.0},
            'hands': {
                'left': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
                'right': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            }
        }

        # Cabeza con hold
        if smoothed_head is not None:
            output['head'] = {'x': float(smoothed_head[0]), 'y': float(smoothed_head[1])}
        elif self.last_head_data and (now - self.last_head_detection_time) < self.hold_time_sec:
            output['head'] = self.last_head_data.copy()

        # Mano izquierda con hold
        if smoothed_left_hand is not None:
            output['hands']['left'] = {
                'detected': True,
                'yaw': float(smoothed_left_hand[0]),
                'pitch': float(smoothed_left_hand[1]),
                'roll': float(smoothed_left_hand[2])
            }
        elif self.last_left_hand_data and (now - self.last_left_hand_detection_time) < self.hold_time_sec:
            output['hands']['left'] = self.last_left_hand_data.copy()

        # Mano derecha con hold
        if smoothed_right_hand is not None:
            output['hands']['right'] = {
                'detected': True,
                'yaw': float(smoothed_right_hand[0]),
                'pitch': float(smoothed_right_hand[1]),
                'roll': float(smoothed_right_hand[2])
            }
        elif self.last_right_hand_data and (now - self.last_right_hand_detection_time) < self.hold_time_sec:
            output['hands']['right'] = self.last_right_hand_data.copy()

        # Actualizar datos compartidos
        with self.data_lock:
            self.pose_data = output

    def _detection_loop(self):
        """Loop principal de detección en hilo separado."""
        last_verbose_time = 0.0

        while self.running:
            with self.camera_lock:
                if not self.camera or not self.camera.isOpened():
                    time.sleep(0.01)
                    continue

                ret, frame = self.camera.read()
                if not ret:
                    continue

            # Procesar detecciones
            head_result = self._process_head_detection(frame)
            hand_result = self._process_hand_detection(frame)

            # Actualizar datos suavizados
            self._update_smoothed_data(head_result, hand_result)

            # Verbose output mejorado
            if self.verbose and time.time() - last_verbose_time >= self.verbose:
                last_verbose_time = time.time()
                self._print_verbose_data()

            # Control de salida
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                self.stop()
                break

    def _print_verbose_data(self):
        """Imprime todos los datos de pose detectados con indicadores de estado."""
        with self.data_lock:
            head = self.pose_data['head']
            hands = self.pose_data['hands']

            now = time.time()

            print(f"\n{'=' * 70}")
            print(f"POSE DATA (SMOOTHED) - {time.strftime('%H:%M:%S')}")
            print(f"{'=' * 70}")

            # Estado de cabeza
            head_age = now - self.last_head_detection_time if self.last_head_detection_time > 0 else float('inf')
            head_status = "🟢 LIVE" if head_age < 0.1 else "🟡 HOLD" if head_age < self.hold_time_sec else "🔴 LOST"
            print(f"HEAD     | {head_status} | X: {head['x']:+6.2f} | Y: {head['y']:+6.2f}")

            # Estado de manos
            for hand_type in ['left', 'right']:
                hand_data = hands[hand_type]
                detection_time = getattr(self, f'last_{hand_type}_hand_detection_time')
                hand_age = now - detection_time if detection_time > 0 else float('inf')

                if hand_data['detected']:
                    hand_status = "🟢 LIVE" if hand_age < 0.1 else "🟡 HOLD" if hand_age < self.hold_time_sec else "🔴 LOST"
                else:
                    hand_status = "⚫ NONE"

                print(f"HAND {hand_type.upper():>4} | {hand_status} | " +
                      f"Yaw: {hand_data['yaw']:+6.1f}° | " +
                      f"Pitch: {hand_data['pitch']:+6.1f}° | " +
                      f"Roll: {hand_data['roll']:+6.1f}°")

            print(f"{'=' * 70}")
            print(
                f"📊 Window: {self.smoothing_window_ms}ms | Emit: {self.emit_interval_ms}ms | Hold: {self.hold_time_sec}s")

    def start(self, camera_id: int = 0):
        """Inicia la detección."""
        if self.running:
            return

        # Inicializar cámara
        with self.camera_lock:
            self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                raise RuntimeError('No se pudo abrir la cámara')

        # Iniciar hilo de detección
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()

        print("🚀 Detector con suavizado iniciado - Presiona 'Q' para salir")
        print(
            f"📊 Configuración: Ventana {self.smoothing_window_ms}ms | Emisión {self.emit_interval_ms}ms | Hold {self.hold_time_sec}s")
        print("📹 Detectando cabeza y manos...")

    def stop(self):
        """Detiene la detección."""
        self.running = False

        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)

        with self.camera_lock:
            if self.camera:
                self.camera.release()
                self.camera = None

        print("\n🛑 Detector detenido")

    def get_pose_data(self) -> Dict:
        """Devuelve los datos de pose actuales suavizados."""
        with self.data_lock:
            return self.pose_data.copy()

    def __del__(self):
        self.stop()
