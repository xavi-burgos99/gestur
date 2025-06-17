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
    """Clase para suavizar valores en una ventana temporal deslizante."""

    def __init__(self, window_ms=500):
        self.window_ms = window_ms
        self.values = deque()  # almacena tuplas (timestamp, value)

    def add_value(self, value):
        """Añade un nuevo valor con timestamp actual."""
        now = time.time() * 1000  # tiempo actual en ms
        self.values.append((now, value))
        self._remove_old(now)

    def _remove_old(self, now):
        """Elimina valores más antiguos que la ventana."""
        while self.values and (now - self.values[0][0]) > self.window_ms:
            self.values.popleft()

    def get_smoothed(self):
        """Devuelve el promedio de todos los valores en la ventana."""
        if not self.values:
            return None
        # Promediar todos los valores en la ventana
        vals = np.array([v for _, v in self.values])
        return np.mean(vals, axis=0)


class DataRetentionWithSmoothing:
    """Clase para mantener datos durante un tiempo de retención Y continuar suavizando."""

    def __init__(self, retention_ms=3000):
        self.retention_ms = retention_ms
        self.last_detected_data = None
        self.last_detection_time = 0
        self.currently_detected = False

    def update(self, detected_data):
        """
        Actualiza con nuevos datos detectados y determina qué datos usar para suavizado.

        Returns:
            (data_for_smoothing, is_within_retention)
        """
        now = time.time() * 1000

        if detected_data is not None:
            # Nueva detección: actualizar datos y tiempo
            self.last_detected_data = detected_data
            self.last_detection_time = now
            self.currently_detected = True
            return detected_data, True
        else:
            # No hay detección: verificar si estamos dentro del período de retención
            time_since_last = now - self.last_detection_time

            if time_since_last <= self.retention_ms and self.last_detected_data is not None:
                # Dentro del período de retención: usar últimos datos para suavizado
                self.currently_detected = False
                return self.last_detected_data, True
            else:
                # Fuera del período de retención
                self.currently_detected = False
                return None, False


class PoseDetector:
    def __init__(self, verbose: float = 0.5, smoothing_window_ms: int = 500,
                 emit_interval_ms: int = 50, retention_ms: int = 3000):
        self.verbose = verbose
        self.smoothing_window_ms = smoothing_window_ms
        self.emit_interval_ms = emit_interval_ms
        self.retention_ms = retention_ms

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Inicializar inferenciadores
        self.head_inferencer = MMPoseInferencer(pose2d='body', device=self.device)
        self.hand_inferencer = MMPoseInferencer(pose2d='hand', device=self.device)

        # Cámara compartida
        self.camera = None
        self.camera_lock = threading.Lock()

        # Sistema de suavizado
        self.smoothed_head = SmoothedValue(window_ms=smoothing_window_ms)
        self.smoothed_hands = SmoothedValue(window_ms=smoothing_window_ms)

        # Sistema de retención de datos con suavizado continuo
        self.retention_head = DataRetentionWithSmoothing(retention_ms=retention_ms)
        self.retention_hands = DataRetentionWithSmoothing(retention_ms=retention_ms)

        # Control de emisión y datos
        self.data_lock = threading.Lock()
        self.last_emit_time = 0
        self.current_smoothed_data = {
            'head': {'x': 0.0, 'y': 0.0},
            'hands': {
                'left': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'openness': 0.0},
                'right': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'openness': 0.0}
            }
        }

        # Control de hilos
        self.running = False
        self.detection_thread = None

        # Índices de keypoints de manos (formato MMPose de 21 puntos)
        self.hand_keypoint_indices = {
            'wrist': 0,
            # Pulgar
            'thumb_mcp': 1, 'thumb_pip': 2, 'thumb_dip': 3, 'thumb_tip': 4,
            # Índice
            'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
            # Medio
            'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
            # Anular
            'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
            # Meñique
            'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
        }

        # Configuración para cálculo de apertura
        self.finger_configs = [
            ('thumb', 'thumb_mcp', 'thumb_tip'),
            ('index', 'index_mcp', 'index_tip'),
            ('middle', 'middle_mcp', 'middle_tip'),
            ('ring', 'ring_mcp', 'ring_tip'),
            ('pinky', 'pinky_mcp', 'pinky_tip')
        ]

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

    def _calculate_hand_openness(self, keypoints: np.ndarray) -> float:
        """
        Calcula la apertura de la mano basada en la extensión de los dedos.

        Args:
            keypoints: Array de keypoints de forma (21, 2) o (21, 3)

        Returns:
            Float entre 0.0 (cerrado) y 1.0 (abierto)
        """
        try:
            wrist = keypoints[self.hand_keypoint_indices['wrist']][:2]
            finger_extensions = []

            for finger_name, mcp_key, tip_key in self.finger_configs:
                mcp = keypoints[self.hand_keypoint_indices[mcp_key]][:2]
                tip = keypoints[self.hand_keypoint_indices[tip_key]][:2]

                # Distancia de la muñeca al metacarpo (base del dedo)
                base_distance = np.linalg.norm(mcp - wrist)

                # Distancia de la muñeca a la punta del dedo
                tip_distance = np.linalg.norm(tip - wrist)

                # Si la base está muy cerca de la muñeca, saltar este dedo
                if base_distance < 1e-3:
                    continue

                # Ratio de extensión: cuánto se extiende el dedo respecto a su base
                extension_ratio = tip_distance / base_distance

                # Normalizar: típicamente un dedo cerrado tiene ratio ~1.0, abierto ~1.5-2.0
                if finger_name == 'thumb':
                    # El pulgar tiene geometría diferente
                    normalized_extension = np.clip((extension_ratio - 0.8) / 0.7, 0, 1)
                else:
                    # Dedos normales
                    normalized_extension = np.clip((extension_ratio - 1.0) / 0.8, 0, 1)

                finger_extensions.append(normalized_extension)

            if not finger_extensions:
                return 0.0

            # Promedio de extensión de todos los dedos
            average_extension = np.mean(finger_extensions)

            # Aplicar curva suave para hacer más natural la transición
            openness = 1 / (1 + np.exp(-6 * (average_extension - 0.5)))

            return float(np.clip(openness, 0.0, 1.0))

        except Exception as e:
            print(f"Error calculando apertura de mano: {e}")
            return 0.0

    def _calculate_hand_rotation(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calcula los ángulos de rotación de la mano (yaw, pitch, roll) basado en keypoints."""
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
        """Procesa la detección de manos y calcula orientaciones y apertura."""
        try:
            hands_data = {
                'left': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'openness': 0.0},
                'right': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'openness': 0.0}
            }

            for output in self.hand_inferencer(frame):
                predictions = output['predictions'][0]
                if not predictions:
                    continue

                for i, pred in enumerate(predictions):
                    keypoints = np.asarray(pred['keypoints'])

                    # Verificar que tenemos 21 keypoints (formato estándar de mano)
                    if keypoints.shape[0] != 21:
                        continue

                    # Verificar confianza mínima
                    if keypoints.shape[1] == 3:
                        confidence = keypoints[:, 2].mean()
                        if confidence < 0.3:
                            continue

                    # Calcular rotaciones y apertura
                    rotations = self._calculate_hand_rotation(keypoints)
                    openness = self._calculate_hand_openness(keypoints)

                    # Determinar si es mano izquierda o derecha
                    wrist_x = keypoints[0, 0]  # X de la muñeca
                    frame_center_x = frame.shape[1] / 2

                    if wrist_x < frame_center_x:
                        hand_type = 'right'  # Mano en lado izquierdo de la imagen
                    else:
                        hand_type = 'left'  # Mano en lado derecho de la imagen

                    hands_data[hand_type] = {
                        'detected': True,
                        'yaw': rotations['yaw'],
                        'pitch': rotations['pitch'],
                        'roll': rotations['roll'],
                        'openness': openness
                    }

            return hands_data

        except Exception as e:
            print(f"Error en detección de manos: {e}")
            return None

    def _update_smoothing(self, head_data, hands_data):
        """
        Actualiza los valores suavizados con las nuevas detecciones.
        IMPORTANTE: Ahora mantiene suavizando con últimos valores durante retención.
        """
        # Procesar cabeza con retención y suavizado continuo
        head_for_smoothing, head_in_retention = self.retention_head.update(head_data)
        if head_for_smoothing is not None and head_in_retention:
            head_array = np.array([head_for_smoothing['x'], head_for_smoothing['y']])
            self.smoothed_head.add_value(head_array)

        # Procesar manos con retención y suavizado continuo
        hands_for_smoothing, hands_in_retention = self.retention_hands.update(hands_data)
        if hands_for_smoothing is not None and hands_in_retention:
            # Convertir datos de manos a array plano para suavizado
            left = hands_for_smoothing.get('left', {'detected': False, 'yaw': 0, 'pitch': 0, 'roll': 0, 'openness': 0})
            right = hands_for_smoothing.get('right',
                                            {'detected': False, 'yaw': 0, 'pitch': 0, 'roll': 0, 'openness': 0})

            # Marcar como detectadas si estamos usando datos retenidos de detecciones recientes
            left_detected = 1.0 if left['detected'] else 0.0
            right_detected = 1.0 if right['detected'] else 0.0

            hands_array = np.array([
                left_detected, left['yaw'], left['pitch'], left['roll'], left['openness'],
                right_detected, right['yaw'], right['pitch'], right['roll'], right['openness']
            ])
            self.smoothed_hands.add_value(hands_array)

    def _get_smoothed_data(self):
        """Obtiene datos suavizados con control de emisión."""
        now = time.time() * 1000

        # Control de intervalos de emisión (50ms por defecto)
        if now - self.last_emit_time < self.emit_interval_ms:
            return None  # Aún no es momento de emitir

        self.last_emit_time = now

        # Obtener valores suavizados
        smoothed_head_array = self.smoothed_head.get_smoothed()
        smoothed_hands_array = self.smoothed_hands.get_smoothed()

        # Construir resultado final
        result = {}

        # Procesar cabeza
        if smoothed_head_array is not None:
            result['head'] = {
                'x': float(smoothed_head_array[0]),
                'y': float(smoothed_head_array[1])
            }
        else:
            result['head'] = {'x': 0.0, 'y': 0.0}

        # Procesar manos
        if smoothed_hands_array is not None:
            # Determinar si las manos están siendo detectadas actualmente o retenidas
            left_currently_detected = self.retention_hands.currently_detected if hasattr(self.retention_hands,
                                                                                         'currently_detected') else False
            right_currently_detected = left_currently_detected  # Para simplificar, usar el mismo estado

            result['hands'] = {
                'left': {
                    'detected': bool(round(smoothed_hands_array[0])) and left_currently_detected,
                    'yaw': float(smoothed_hands_array[1]),
                    'pitch': float(smoothed_hands_array[2]),
                    'roll': float(smoothed_hands_array[3]),
                    'openness': float(np.clip(smoothed_hands_array[4], 0.0, 1.0))
                },
                'right': {
                    'detected': bool(round(smoothed_hands_array[5])) and right_currently_detected,
                    'yaw': float(smoothed_hands_array[6]),
                    'pitch': float(smoothed_hands_array[7]),
                    'roll': float(smoothed_hands_array[8]),
                    'openness': float(np.clip(smoothed_hands_array[9], 0.0, 1.0))
                }
            }
        else:
            result['hands'] = {
                'left': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'openness': 0.0},
                'right': {'detected': False, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'openness': 0.0}
            }

        return result

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

            # Procesar detecciones RAW (sin suavizado aún)
            head_result = self._process_head_detection(frame)
            hand_result = self._process_hand_detection(frame)

            # Actualizar sistema de suavizado (ahora incluye retención activa)
            self._update_smoothing(head_result, hand_result)

            # Obtener datos suavizados
            smoothed_data = self._get_smoothed_data()

            # Actualizar datos compartidos solo si hay datos válidos
            if smoothed_data is not None:
                with self.data_lock:
                    self.current_smoothed_data = smoothed_data

            # Verbose output mejorado
            if self.verbose and time.time() - last_verbose_time >= self.verbose:
                last_verbose_time = time.time()
                self._print_verbose_data()

            # Control de salida
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                self.stop()
                break

    def _get_openness_indicator(self, openness: float) -> str:
        """Devuelve un indicador visual para el nivel de apertura."""
        if openness < 0.2:
            return "✊"  # Puño cerrado
        elif openness < 0.4:
            return "👋"  # Parcialmente abierto
        elif openness < 0.7:
            return "🖐"  # Mayormente abierto
        else:
            return "✋"  # Completamente abierto

    def _get_detection_status_indicator(self, hand_type: str) -> str:
        """Devuelve indicador de estado de detección (detectando vs reteniendo)."""
        retention_obj = self.retention_hands

        if hasattr(retention_obj, 'currently_detected'):
            if retention_obj.currently_detected:
                return "🟢"  # Detectando activamente
            elif retention_obj.last_detected_data is not None:
                return "🟡"  # Reteniendo (usando últimos datos)
            else:
                return "/"  # Sin datos
        else:
            return "?"  # Estado desconocido

    def _print_verbose_data(self):
        """Imprime todos los datos de pose detectados (suavizados) incluyendo estado de retención."""
        with self.data_lock:
            head = self.current_smoothed_data['head']
            hands = self.current_smoothed_data['hands']

            print(f"\n{'=' * 90}")
            print(f"POSE DATA (SMOOTHED + RETENTION) - {time.strftime('%H:%M:%S')}")
            print(f"{'=' * 90}")

            # Datos de cabeza
            head_status = self._get_detection_status_indicator('head')
            print(f"HEAD     {head_status} | X: {head['x']:+6.2f} | Y: {head['y']:+6.2f}")

            # Datos de manos con apertura y estado de retención
            for hand_type in ['left', 'right']:
                hand_data = hands[hand_type]
                status = "✓" if hand_data['detected'] else "✗"
                icon = "🤚" if hand_type == "left" else "✋"
                openness_icon = self._get_openness_indicator(hand_data['openness'])
                detection_status = self._get_detection_status_indicator(hand_type)

                print(f"HAND {hand_type.upper():>4} {detection_status} | {status} | " +
                      f"Yaw: {hand_data['yaw']:+6.1f}° | " +
                      f"Pitch: {hand_data['pitch']:+6.1f}° | " +
                      f"Roll: {hand_data['roll']:+6.1f}° | " +
                      f"Open: {hand_data['openness']:5.2f}")

            print(f"{'=' * 90}")
            print(f"Smoothing: {self.smoothing_window_ms}ms | " +
                  f"Emit: {self.emit_interval_ms}ms | " +
                  f"Retention: {self.retention_ms}ms")
            print(f"🟢 = Detectando | 🟡 = Reteniendo | / = Sin datos")

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

        print("Detector iniciado con suavizado continuo - Presiona 'Q' para salir")
        print(f"Detectando cabeza, manos y apertura con retención suavizada...")
        print(
            f"Suavizado: {self.smoothing_window_ms}ms | Emisión: {self.emit_interval_ms}ms | Retención: {self.retention_ms}ms")
        print(f"El suavizado CONTINÚA durante la retención para reconexión suave")

    def stop(self):
        """Detiene la detección."""
        self.running = False

        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)

        with self.camera_lock:
            if self.camera:
                self.camera.release()
                self.camera = None

        print("\nDetector detenido")

    def get_pose_data(self) -> Dict:
        """Devuelve los datos de pose suavizados actuales en formato JSON."""
        with self.data_lock:
            return self.current_smoothed_data.copy()

    def __del__(self):
        self.stop()
