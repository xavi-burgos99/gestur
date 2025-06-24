import cv2
import mediapipe as mp
import math
import time
import threading


class PoseHandTracker:
    def __init__(self, response_time_ms=50, smoothing_time_ms=500,
                 use_pose=True, use_hands=True,
                 mirror=False, invert_hands=False, verbose=False):
        """Inicializa el tracker de pose y manos en tiempo real."""
        # Configuración de parámetros
        self.use_pose = use_pose
        self.use_hands = use_hands
        # Opciones extra
        self.mirror = mirror            # espejo horizontal del frame
        self.invert_hands = invert_hands  # intercambiar izquierda/derecha
        self.response_time = response_time_ms / 1000.0  # en segundos
        self.smoothing_time = smoothing_time_ms / 1000.0  # en segundos
        # Factor de suavizado (exponencial): α = dt / smoothing_time
        # Si smoothing_time es 0 (sin suavizado), ponemos α=1
        self.alpha = 1.0 if self.smoothing_time <= 0 else min(1.0, self.response_time / self.smoothing_time)

        # Inicializar MediaPipe Pose y Hands si corresponden
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        self.pose_model = None
        self.hands_model = None
        if self.use_pose:
            # Modelo pose: usar modelo ligero (model_complexity=0) para rendimiento
            self.pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                           enable_segmentation=False,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
        if self.use_hands:
            # Modelo hands: max 2 manos, modelo complejo=1 (se podría bajar a 0 si fuera necesario)
            self.hands_model = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Estructura para datos actuales (filtrados)
        self.current_data = {
            "head":  {"detected": False, "x": None, "y": None,
                      "pitch": None, "yaw": None, "roll": None},
            "torso": {"detected": False, "x": None, "y": None,
                      "pitch": None, "yaw": None, "roll": None},
            "left_hand":  {"detected": False, "x": None, "y": None,
                           "pitch": None, "yaw": None, "roll": None},
            "right_hand": {"detected": False, "x": None, "y": None,
                           "pitch": None, "yaw": None, "roll": None}
        }

        # Lista de listeners para callback
        self.listeners = []

        # Verbose option
        self.verbose = verbose
        if self.verbose:
            print(f"[PoseHandTracker] Init | pose={self.use_pose} hands={self.use_hands} "
                  f"| mirror={self.mirror} invert_hands={self.invert_hands} "
                  f"| dt={self.response_time*1000:.0f} ms smooth={self.smoothing_time*1000:.0f} ms")

        # Configurar cámara (webcam 0)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo acceder a la cámara.")
        # Opcional: reducir resolución para mejorar rendimiento
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Control diferido: el seguimiento se inicia con run()
        self.running = False
        self.thread = None
        # Timestamp de la última detección por mano (se reinicia en run)
        self.last_left_time = 0.0
        self.last_right_time = 0.0

    def run(self):
        """Arranca el hilo de captura/procesamiento si aún no está en marcha."""
        if self.running:
            if self.verbose:
                print("[PoseHandTracker] run() ya fue llamado; ignorando.")
            return
        # Reiniciar marcas de tiempo
        now = time.time()
        self.last_left_time = now
        self.last_right_time = now
        # Iniciar hilo
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        if self.verbose:
            print("[PoseHandTracker] Tracking iniciado.")

    def _process_loop(self):
        """Bucle interno que captura frames y actualiza datos a la frecuencia deseada."""
        prev_time = time.time()
        while self.running:
            success, frame = self.cap.read()
            # Aplicar espejo si se solicita
            if self.mirror and success:
                frame = cv2.flip(frame, 1)  # 1 == flip horizontal
            if not success:
                continue  # si falla lectura, intentar siguiente

            # Convertir a RGB (MediaPipe espera color RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[0], frame.shape[1]

            # Resultados de MediaPipe
            pose_res = None
            hand_res = None
            if self.use_pose and self.pose_model:
                pose_res = self.pose_model.process(frame_rgb)
            if self.use_hands and self.hands_model:
                hand_res = self.hands_model.process(frame_rgb)

            # Variables locales para nuevos valores calculados (sin filtrar)
            head_pitch = head_yaw = head_roll = 0.0
            torso_pitch = torso_yaw = torso_roll = 0.0
            head_x = head_y = None
            torso_x = torso_y = None
            left_x = left_y = None
            right_x = right_y = None
            left_pitch = left_yaw = left_roll = 0.0
            right_pitch = right_yaw = right_roll = 0.0

            # Reset detection flags for this frame
            left_detected = False
            right_detected = False
            head_detected = False
            torso_detected = False

            # Cálculo de datos de pose (cabeza y torso) si disponible
            if pose_res and pose_res.pose_world_landmarks:
                head_detected = True
                torso_detected = True
                # Extraer landmarks necesarios
                # Nota: accederemos a pose_world_landmarks (3D) para cálculos de ángulos,
                # y pose_landmarks (2D) para posiciones en pixeles si hiciera falta.
                lm = pose_res.pose_world_landmarks.landmark  # lista de 33 landmarks 3D
                lm2d = pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None  # lista 2D normalizados

                # Puntos de interés
                nose = lm[0]
                left_ear = lm[7]
                right_ear = lm[8]
                left_shoulder = lm[11]
                right_shoulder = lm[12]
                left_hip = lm[23]
                right_hip = lm[24]

                # Calcular orientación de la cabeza (usando nariz y orejas)
                # Vector de dirección de la cabeza: desde centro de orejas hacia la nariz
                ear_mid_x = (left_ear.x + right_ear.x) / 2.0
                ear_mid_y = (left_ear.y + right_ear.y) / 2.0
                ear_mid_z = (left_ear.z + right_ear.z) / 2.0
                dir_head = (
                    nose.x - ear_mid_x,
                    nose.y - ear_mid_y,
                    nose.z - ear_mid_z
                )
                # Yaw de la cabeza: ángulo de dir_head proyectado en plano X-Z
                # Tomamos eje Z hacia adelante de la cámara. Suponiendo que el eje Z de landmarks
                # apunta hacia la cámara (negativo hacia adelante).
                dir_x, dir_y, dir_z = dir_head
                # Proyección en plano horizontal (x-z)
                proj_xz = math.sqrt(dir_x ** 2 + dir_z ** 2)
                if proj_xz > 1e-6:
                    head_yaw = math.degrees(math.atan2(dir_x, -dir_z))
                else:
                    head_yaw = 0.0
                # Pitch de la cabeza: ángulo entre vector y su proyección horizontal (y vs xz)
                head_pitch = math.degrees(math.atan2(dir_y, proj_xz))
                # Roll de la cabeza: ángulo de inclinación de la línea de orejas
                # Vector oreja->oreja
                ear_vec = (
                    right_ear.x - left_ear.x,
                    right_ear.y - left_ear.y,
                    right_ear.z - left_ear.z
                )
                # Roll calculado con atan2(diferencia_y, diferencia_z) respecto al plano horizontal,
                # pero dado que la cámara puede no estar perfectamente alineada, simplificaremos
                # usando solo la componente Y vs X (en imagen 2D esto sería la pendiente).
                # Usamos landmarks 2D para mayor precisión en imagen:
                if lm2d:
                    left_ear_2d = lm2d[7];
                    right_ear_2d = lm2d[8]
                    dy = right_ear_2d.y - left_ear_2d.y
                    dx = right_ear_2d.x - left_ear_2d.x
                else:
                    dy = ear_vec[1];
                    dx = ear_vec[0]
                head_roll = math.degrees(math.atan2(dy, dx))

                # Posición 2D normalizada de la cabeza (nariz) y del torso (punto medio caderas)
                if pose_res.pose_landmarks:
                    nose2d = pose_res.pose_landmarks.landmark[0]  # nariz índice 0
                    head_x, head_y = nose2d.x, nose2d.y
                    hip_mid2d_x = (pose_res.pose_landmarks.landmark[23].x +
                                    pose_res.pose_landmarks.landmark[24].x) / 2.0
                    hip_mid2d_y = (pose_res.pose_landmarks.landmark[23].y +
                                    pose_res.pose_landmarks.landmark[24].y) / 2.0
                    torso_x, torso_y = hip_mid2d_x, hip_mid2d_y

                # Calcular orientación del torso (usando hombros y caderas)
                # Vectores base del torso
                shoulder_mid = (
                    (left_shoulder.x + right_shoulder.x) / 2.0,
                    (left_shoulder.y + right_shoulder.y) / 2.0,
                    (left_shoulder.z + right_shoulder.z) / 2.0
                )
                hip_mid = (
                    (left_hip.x + right_hip.x) / 2.0,
                    (left_hip.y + right_hip.y) / 2.0,
                    (left_hip.z + right_hip.z) / 2.0
                )
                vertical_axis = (
                    shoulder_mid[0] - hip_mid[0],
                    shoulder_mid[1] - hip_mid[1],
                    shoulder_mid[2] - hip_mid[2]
                )
                lateral_axis = (
                    right_shoulder.x - left_shoulder.x,
                    right_shoulder.y - left_shoulder.y,
                    right_shoulder.z - left_shoulder.z
                )
                # Vector frontal mediante producto cruz: vertical x lateral
                forward_axis = (
                    vertical_axis[1] * lateral_axis[2] - vertical_axis[2] * lateral_axis[1],
                    vertical_axis[2] * lateral_axis[0] - vertical_axis[0] * lateral_axis[2],
                    vertical_axis[0] * lateral_axis[1] - vertical_axis[1] * lateral_axis[0]
                )
                # Yaw del torso: ángulo del forward_axis en plano X-Z
                fwd_x, fwd_y, fwd_z = forward_axis
                proj_xz_torso = math.sqrt(fwd_x ** 2 + fwd_z ** 2)
                if proj_xz_torso > 1e-6:
                    torso_yaw = math.degrees(math.atan2(fwd_x, -fwd_z))
                else:
                    torso_yaw = 0.0
                # Pitch del torso: ángulo de forward_axis vs horizontal
                torso_pitch = math.degrees(math.atan2(fwd_y, proj_xz_torso)) if proj_xz_torso > 1e-6 else 0.0
                # Roll del torso: inclinación de la línea de hombros
                # Usamos diferencia de alturas entre hombros
                shoulder_dy = right_shoulder.y - left_shoulder.y
                shoulder_dx = right_shoulder.x - left_shoulder.x
                torso_roll = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
            # Cálculo de datos de manos si disponible
            if hand_res and hand_res.multi_hand_landmarks:
                # Recorremos cada mano detectada
                for hand_landmarks, hand_handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                    # Determinar si es mano izquierda o derecha según el modelo
                    label = hand_handedness.classification[0].label  # "Left" o "Right"
                    # Si se pidió invertir manos, intercambiar la etiqueta
                    if self.invert_hands:
                        label = "Left" if label == "Right" else "Right"
                    # Actualizar flags de detección
                    if label == "Left":
                        left_detected = True
                    else:
                        right_detected = True
                    # Obtener coordenadas normalizadas de la muñeca (landmark 0)
                    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                    # Guardar coordenadas normalizadas (0‑1) como porcentaje
                    wrist_x_pct = wrist.x  # 0‑1
                    wrist_y_pct = wrist.y  # 0‑1
                    # Calcular orientaciones de la mano (pitch, yaw, roll)
                    # Usaremos varios puntos de la mano para estimar el plano de la palma.
                    # Puntos base: muñeca (0), base dedo índice (5), base dedo meñique (17)
                    index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
                    pinky_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]
                    # Vector muñeca -> centro de la palma (promedio entre índice y meñique)
                    palm_center = (
                        (index_mcp.x + pinky_mcp.x) / 2.0,
                        (index_mcp.y + pinky_mcp.y) / 2.0,
                        (index_mcp.z + pinky_mcp.z) / 2.0
                    )
                    vec_palm = (
                        palm_center[0] - wrist.x,
                        palm_center[1] - wrist.y,
                        palm_center[2] - wrist.z
                    )
                    # Pitch de la mano: ángulo de vec_palm vs plano horizontal (eje Y vs X-Z)
                    vec_x, vec_y, vec_z = vec_palm
                    proj_xz_hand = math.sqrt(vec_x ** 2 + vec_z ** 2)
                    hand_pitch_val = math.degrees(math.atan2(-vec_y, proj_xz_hand)) if proj_xz_hand > 1e-6 else 0.0
                    # (Usamos -vec_y porque si la palma se inclina hacia abajo, y aumenta, pero queremos pitch positivo)
                    # Yaw de la mano: ángulo de vec_palm en plano horizontal (X-Z)
                    hand_yaw_val = math.degrees(math.atan2(vec_x, vec_z)) if proj_xz_hand > 1e-6 else 0.0
                    # Roll de la mano: rotación de la palma alrededor de vec_palm.
                    # Calculamos el normal de la palma usando cross product (wrist->index) x (wrist->pinky)
                    v_wrist_index = (
                        index_mcp.x - wrist.x,
                        index_mcp.y - wrist.y,
                        index_mcp.z - wrist.z
                    )
                    v_wrist_pinky = (
                        pinky_mcp.x - wrist.x,
                        pinky_mcp.y - wrist.y,
                        pinky_mcp.z - wrist.z
                    )
                    normal_palm = (
                        v_wrist_index[1] * v_wrist_pinky[2] - v_wrist_index[2] * v_wrist_pinky[1],
                        v_wrist_index[2] * v_wrist_pinky[0] - v_wrist_index[0] * v_wrist_pinky[2],
                        v_wrist_index[0] * v_wrist_pinky[1] - v_wrist_index[1] * v_wrist_pinky[0]
                    )
                    # Para calcular roll, medimos el ángulo del normal respecto a la vertical (eje Y).
                    # Proyectar normal en plano Y-Z (plano vertical mirando de lado)
                    norm_yz = math.sqrt(normal_palm[1] ** 2 + normal_palm[2] ** 2)
                    hand_roll_val = math.degrees(math.atan2(normal_palm[0], norm_yz)) if norm_yz > 1e-6 else 0.0
                    # Asignar a la mano correspondiente
                    if label == "Left":
                        left_x, left_y = wrist_x_pct, wrist_y_pct
                        left_pitch, left_yaw, left_roll = hand_pitch_val, hand_yaw_val, hand_roll_val
                    else:  # "Right"
                        right_x, right_y = wrist_x_pct, wrist_y_pct
                        right_pitch, right_yaw, right_roll = hand_pitch_val, hand_yaw_val, hand_roll_val

            # Guardar posiciones (sin suavizado)
            self.current_data["head"]["x"] = head_x
            self.current_data["head"]["y"] = head_y
            self.current_data["torso"]["x"] = torso_x
            self.current_data["torso"]["y"] = torso_y
            # Si no se detectó cabeza/torso en este frame, vaciar valores
            if not head_detected:
                head_pitch = head_yaw = head_roll = None
            if not torso_detected:
                torso_pitch = torso_yaw = torso_roll = None

            # Aplicar suavizado (exponential smoothing) a cada valor antes de actualizar current_data
            # Cabeza
            if head_detected:
                self.current_data["head"]["pitch"] = (1 - self.alpha) * (self.current_data["head"]["pitch"] or 0) + self.alpha * head_pitch
                self.current_data["head"]["yaw"]   = (1 - self.alpha) * (self.current_data["head"]["yaw"] or 0)   + self.alpha * head_yaw
                self.current_data["head"]["roll"]  = (1 - self.alpha) * (self.current_data["head"]["roll"] or 0)  + self.alpha * head_roll
            elif not self.current_data["head"]["detected"]:
                self.current_data["head"]["pitch"] = self.current_data["head"]["yaw"] = self.current_data["head"]["roll"] = None
            # Torso
            if torso_detected:
                self.current_data["torso"]["pitch"] = (1 - self.alpha) * (self.current_data["torso"]["pitch"] or 0) + self.alpha * torso_pitch
                self.current_data["torso"]["yaw"]   = (1 - self.alpha) * (self.current_data["torso"]["yaw"] or 0)   + self.alpha * torso_yaw
                self.current_data["torso"]["roll"]  = (1 - self.alpha) * (self.current_data["torso"]["roll"] or 0)  + self.alpha * torso_roll
            elif not self.current_data["torso"]["detected"]:
                self.current_data["torso"]["pitch"] = self.current_data["torso"]["yaw"] = self.current_data["torso"]["roll"] = None
            # Manos
            # Para manos, si no se detectó mano en este frame, mantenemos la posición previa (no la actualizamos para no poner None a mitad de suavizado).
            if left_x is not None and left_y is not None:
                self.current_data["left_hand"]["x"] = left_x
                self.current_data["left_hand"]["y"] = left_y
            if right_x is not None and right_y is not None:
                self.current_data["right_hand"]["x"] = right_x
                self.current_data["right_hand"]["y"] = right_y
            # Remember last detection time stamps
            now = time.time()
            if left_detected:
                self.last_left_time = now
            if right_detected:
                self.last_right_time = now

            # Determine if hand is still considered "present"
            left_present = (now - self.last_left_time) < self.smoothing_time
            right_present = (now - self.last_right_time) < self.smoothing_time

            # Suavizar orientaciones solo si la mano está presente (dentro del periodo de smoothing)
            if left_present:
                prev_lp = self.current_data["left_hand"]["pitch"] or 0.0
                prev_ly = self.current_data["left_hand"]["yaw"]   or 0.0
                prev_lr = self.current_data["left_hand"]["roll"]  or 0.0
                self.current_data["left_hand"]["pitch"] = (1 - self.alpha) * prev_lp + self.alpha * left_pitch
                self.current_data["left_hand"]["yaw"]   = (1 - self.alpha) * prev_ly + self.alpha * left_yaw
                self.current_data["left_hand"]["roll"]  = (1 - self.alpha) * prev_lr + self.alpha * left_roll
            if right_present:
                prev_rp = self.current_data["right_hand"]["pitch"] or 0.0
                prev_ry = self.current_data["right_hand"]["yaw"]   or 0.0
                prev_rr = self.current_data["right_hand"]["roll"]  or 0.0
                self.current_data["right_hand"]["pitch"] = (1 - self.alpha) * prev_rp + self.alpha * right_pitch
                self.current_data["right_hand"]["yaw"]   = (1 - self.alpha) * prev_ry + self.alpha * right_yaw
                self.current_data["right_hand"]["roll"]  = (1 - self.alpha) * prev_rr + self.alpha * right_roll

            # Update detection flags exposed to the exterior
            self.current_data["left_hand"]["detected"] = left_present
            self.current_data["right_hand"]["detected"] = right_present
            self.current_data["head"]["detected"]  = head_detected
            self.current_data["torso"]["detected"] = torso_detected

            # Clear coordinates and rotations when the hand is no longer present
            if not left_present:
                self.current_data["left_hand"]["x"] = None
                self.current_data["left_hand"]["y"] = None
                self.current_data["left_hand"]["pitch"] = None
                self.current_data["left_hand"]["yaw"] = None
                self.current_data["left_hand"]["roll"] = None
            if not right_present:
                self.current_data["right_hand"]["x"] = None
                self.current_data["right_hand"]["y"] = None
                self.current_data["right_hand"]["pitch"] = None
                self.current_data["right_hand"]["yaw"] = None
                self.current_data["right_hand"]["roll"] = None

            # Notificar a los listeners con los nuevos datos
            for callback in self.listeners:
                try:
                    callback(self.current_data)
                except Exception as e:
                    print(f"Error en callback de listener: {e}")

            # Verbose pretty print
            if self.verbose:
                self._print_verbose_data()

            # Controlar la tasa de refresco para aproximarla a response_time_ms
            current_time = time.time()
            elapsed = current_time - prev_time
            # Si el procesamiento fue más rápido que el intervalo deseado, dormir el resto
            sleep_time = self.response_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            prev_time = time.time()
        # Fin del while
        # Al salir del loop, liberar la cámara y modelos
        self.cap.release()
        if self.pose_model:
            self.pose_model.close()
        if self.hands_model:
            self.hands_model.close()

    def _print_verbose_data(self):
        """Imprime datos actuales: posición (%) y rotación, en formato tabulado."""
        d = self.current_data
        # Helper for row logging
        def log_row(hand):
            # Choose flag symbol
            if hand["detected"]:
                det_flag = "✔"
            elif hand["x"] is not None:
                det_flag = "·"   # smoothing grace period
            else:
                det_flag = "✖"
            # x/y
            if hand["x"] is not None:
                x_disp = f"{hand['x']*100:5.1f}%"
                y_disp = f"{hand['y']*100:5.1f}%"
            else:
                x_disp = "-"
                y_disp = "-"
            # rotation
            def f(val):
                return f"{val:7.2f}°" if val is not None else "   -   "
            return (f"{det_flag}  x={x_disp:>6}  y={y_disp:>6} | "
                    f"pitch={f(hand['pitch'])}  yaw={f(hand['yaw'])}  "
                    f"roll={f(hand['roll'])}")

        print("\n[PoseHandTracker]")
        print(f"  HEAD   : {log_row(d['head'])}")
        print(f"  TORSO  : {log_row(d['torso'])}")
        print(f"  LEFT H : {log_row(d['left_hand'])}")
        print(f"  RIGHT H: {log_row(d['right_hand'])}")
        print("-"*64)

    def get_current_data(self):
        """Devuelve el diccionario JSON actual con los datos de rotación/posición."""
        return self.current_data

    def subscribe(self, listener_fn):
        """Registra una función de callback para escuchar datos nuevos."""
        if callable(listener_fn):
            self.listeners.append(listener_fn)

    def unsubscribe(self, listener_fn):
        """Elimina una función de callback de la lista de suscriptores."""
        if listener_fn in self.listeners:
            self.listeners.remove(listener_fn)

    def stop(self):
        """Detiene el bucle de procesamiento y libera recursos."""
        if not self.running:
            return
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

if __name__ == "__main__":
    tracker = PoseHandTracker(response_time_ms=100, smoothing_time_ms=500,
                              use_pose=True, use_hands=True,
                              mirror=True, invert_hands=False,
                              verbose=True)

    tracker.run()  # iniciar procesamiento

    # Ejemplo de callback para imprimir datos
    def print_data(data):
        print("Head (pitch,yaw,roll):", data["head"],
              " Torso:", data["torso"],
              " Left hand:", data["left_hand"],
              " Right hand:", data["right_hand"])

    #tracker.subscribe(print_data)

    # ... ejecutar por algún tiempo ...
    time.sleep(20)  # mantener 20 segundos recibiendo datos
    tracker.stop()  # detener el seguimiento y liberar recursos