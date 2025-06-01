import time
import threading
import cv2
import numpy as np
import torch
import warnings
import logging
from typing import List, Dict, Optional
from mmpose.apis import MMPoseInferencer

# Configurar logging para silenciar warnings conocidos
logging.getLogger('mmengine').setLevel(logging.ERROR)

# Silenciar warnings específicos
warnings.filterwarnings("ignore", message=".*Failed to search registry.*")
warnings.filterwarnings("ignore", message=".*TorchScript.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")


class PoseDetector:
    def __init__(self, verbose: float = 0.5):
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Inicializar inferenciadores
        self.head_inferencer = MMPoseInferencer(pose2d='body', device=self.device)
        # TODO: Futuro inferenciador de manos
        # self.hand_inferencer = MMPoseInferencer(pose2d='hand', device=self.device)

        # Cámara compartida
        self.camera = None
        self.camera_lock = threading.Lock()

        # Datos compartidos
        self.data_lock = threading.Lock()
        self.pose_data = {
            'head': {'x': 0.0, 'y': 0.0},
            'hands': None  # Preparado para futuro uso
        }

        # Control de hilos
        self.running = False
        self.detection_thread = None

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
        """
        Procesa la detección de manos (preparado para implementación futura).

        TODO: Implementar cuando se añada el inferenciador de manos
        """
        # Estructura preparada para futuro desarrollo
        # for output in self.hand_inferencer(frame):
        #     # Procesar detecciones de manos
        #     pass

        return None

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

            # Actualizar datos compartidos
            if head_result or hand_result:
                with self.data_lock:
                    if head_result:
                        self.pose_data['head'] = head_result
                    if hand_result:
                        self.pose_data['hands'] = hand_result

            # Verbose output
            if self.verbose and head_result and time.time() - last_verbose_time >= self.verbose:
                last_verbose_time = time.time()
                print(f"Head X: {head_result['x']:+.2f}  Head Y: {head_result['y']:+.2f}")

            # Control de salida
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                self.stop()
                break

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

    def stop(self):
        """Detiene la detección."""
        self.running = False

        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)

        with self.camera_lock:
            if self.camera:
                self.camera.release()
                self.camera = None

    def get_pose_data(self) -> Dict:
        """Devuelve los datos de pose actuales en formato JSON."""
        with self.data_lock:
            return self.pose_data.copy()

    def __del__(self):
        self.stop()
