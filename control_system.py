# control_system.py (modificado para combinar funcionalidades)

import math
import time
from abc import ABC, abstractmethod


class Smoother(ABC):
    """Clase base para algoritmos de suavizado"""

    @abstractmethod
    def update(self, value):
        pass

    @abstractmethod
    def reset(self):
        pass


class ExponentialSmoother(Smoother):
    """Suavizado exponencial con decaimiento al centro"""

    def __init__(self, alpha=0.3, decay_rate=0.1, center_value=0.5):
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.center_value = center_value
        self.smoothed_value = center_value
        self.has_data = False

    def update(self, value):
        if value is not None:
            if not self.has_data:
                self.smoothed_value = value
            else:
                self.smoothed_value = self.alpha * value + (1 - self.alpha) * self.smoothed_value
            self.has_data = True
        else:
            if self.has_data:
                self.smoothed_value = (1 - self.decay_rate) * self.smoothed_value + self.decay_rate * self.center_value
                if abs(self.smoothed_value - self.center_value) < 1e-4:
                    self.smoothed_value = self.center_value
                    self.has_data = False
        return self.smoothed_value

    def reset(self):
        self.smoothed_value = self.center_value
        self.has_data = False


# En tu control_system.py, modifica la clase HybridRotationController:

class HybridRotationController:
    """
    Controlador híbrido que combina rotación proporcional + continua con velocidad gradual
    """

    def __init__(self, max_degrees=30.0, left_threshold=0.25, right_threshold=0.75,
                 continuous_speed_degrees_per_second=45.0, center=0.5, invert=False):
        self.max_degrees = max_degrees
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold
        self.max_continuous_speed = continuous_speed_degrees_per_second
        self.center = center
        self.invert = invert

        # Estado para rotación continua
        self.continuous_rotation = 0.0
        self.last_update_time = time.time()
        self.is_in_continuous_mode = False
        self.continuous_direction = 0  # -1, 0, 1

        # Para transición suave
        self.proportional_rotation_at_threshold = 0.0

    def calculate_continuous_speed_factor(self, head_x_value):
        """
        Calcula el factor de velocidad continua basado en la distancia desde el umbral

        Args:
            head_x_value: Posición X de la cabeza (0.0 a 1.0)

        Returns:
            Factor de velocidad (0.0 a 1.0)
        """
        if head_x_value > self.right_threshold:
            # Zona derecha: 0.75 -> 1.0
            # Factor va de 0.0 (en 0.75) a 1.0 (en 1.0)
            distance_from_threshold = head_x_value - self.right_threshold
            max_distance = 1.0 - self.right_threshold  # 0.25
            factor = min(1.0, distance_from_threshold / max_distance)
            return factor

        elif head_x_value < self.left_threshold:
            # Zona izquierda: 0.25 -> 0.0
            # Factor va de 0.0 (en 0.25) a 1.0 (en 0.0)
            distance_from_threshold = self.left_threshold - head_x_value
            max_distance = self.left_threshold  # 0.25
            factor = min(1.0, distance_from_threshold / max_distance)
            return factor
        else:
            # Zona normal: sin rotación continua
            return 0.0

    def update(self, head_x_value):
        """
        Actualiza la rotación híbrida con velocidad gradual

        Args:
            head_x_value: Posición X de la cabeza (0.0 a 1.0)

        Returns:
            Rotación total en grados
        """
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time

        if head_x_value is None:
            # Sin detección -> resetear todo
            self.is_in_continuous_mode = False
            self.continuous_direction = 0
            self.continuous_rotation = 0.0
            return 0.0

        # Determinar zona actual
        in_left_extreme = head_x_value < self.left_threshold
        in_right_extreme = head_x_value > self.right_threshold
        in_normal_zone = self.left_threshold <= head_x_value <= self.right_threshold

        if in_normal_zone:
            # MODO NORMAL: Rotación proporcional (comportamiento "asomar")
            if self.is_in_continuous_mode:
                # Saliendo de modo continuo -> resetear
                self.is_in_continuous_mode = False
                self.continuous_direction = 0
                print("🛑 Saliendo de modo continuo -> modo proporcional")

            # Calcular rotación proporcional normal
            offset = head_x_value - self.center
            if self.invert:
                offset = -offset
            proportional_rotation = offset * 2.0 * self.max_degrees
            proportional_rotation = max(-self.max_degrees, min(self.max_degrees, proportional_rotation))

            return self.continuous_rotation + proportional_rotation

        else:
            # MODO EXTREMO: Rotación continua con velocidad gradual
            current_direction = -1 if in_left_extreme else 1

            if not self.is_in_continuous_mode:
                # Entrando en modo continuo por primera vez
                self.is_in_continuous_mode = True
                self.continuous_direction = current_direction

                # Calcular la rotación proporcional en el umbral para transición suave
                threshold_value = self.left_threshold if in_left_extreme else self.right_threshold
                offset = threshold_value - self.center
                if self.invert:
                    offset = -offset
                self.proportional_rotation_at_threshold = offset * 2.0 * self.max_degrees
                self.proportional_rotation_at_threshold = max(-self.max_degrees, min(self.max_degrees,
                                                                                     self.proportional_rotation_at_threshold))

                print(f"🔄 Entrando en modo CONTINUO GRADUAL {'izquierda' if in_left_extreme else 'derecha'}")

            elif self.continuous_direction != current_direction:
                # Cambio de dirección en extremos
                self.continuous_direction = current_direction
                print(f"🔄 Cambiando dirección continua a {'izquierda' if in_left_extreme else 'derecha'}")

            # NUEVA LÓGICA: Calcular velocidad gradual
            speed_factor = self.calculate_continuous_speed_factor(head_x_value)
            current_speed = self.max_continuous_speed * speed_factor

            # Aplicar rotación continua con velocidad gradual
            if self.is_in_continuous_mode and current_speed > 0:
                rotation_increment = self.continuous_direction * current_speed * delta_time
                if self.invert:
                    rotation_increment = -rotation_increment
                self.continuous_rotation += rotation_increment

                # Mantener en rango razonable
                while self.continuous_rotation > 360:
                    self.continuous_rotation -= 360
                while self.continuous_rotation < -360:
                    self.continuous_rotation += 360

                if abs(speed_factor - 1.0) < 0.01:  # Casi al máximo
                    if not hasattr(self, '_max_speed_logged'):
                        print(f"🚀 Velocidad máxima alcanzada: {current_speed:.1f}°/s")
                        self._max_speed_logged = True
                else:
                    self._max_speed_logged = False

            return self.continuous_rotation + self.proportional_rotation_at_threshold



class DataProcessor:
    """Procesador de datos que enriquece la información de entrada"""

    def __init__(self):
        pass

    def process_hands(self, left_hand_data, right_hand_data):
        """
        Procesa datos de manos y calcula información derivada

        Returns:
            Dict con datos procesados de manos
        """
        left_detected = left_hand_data.get('detected', False)
        right_detected = right_hand_data.get('detected', False)

        result = {
            'left_detected': left_detected,
            'right_detected': right_detected,
            'any_detected': left_detected or right_detected,
            'both_detected': left_detected and right_detected,
            'count': sum([left_detected, right_detected])
        }

        # Datos de mano izquierda
        if left_detected:
            result['left_x'] = left_hand_data.get('x')
            result['left_y'] = left_hand_data.get('y')
        else:
            result['left_x'] = None
            result['left_y'] = None

        # Datos de mano derecha
        if right_detected:
            result['right_x'] = right_hand_data.get('x')
            result['right_y'] = right_hand_data.get('y')
        else:
            result['right_x'] = None
            result['right_y'] = None

        # Calcular posición promedio
        valid_x = [x for x in [result['left_x'], result['right_x']] if x is not None]
        valid_y = [y for y in [result['left_y'], result['right_y']] if y is not None]

        if valid_x and valid_y:
            result['center_x'] = sum(valid_x) / len(valid_x)
            result['center_y'] = sum(valid_y) / len(valid_y)
        else:
            result['center_x'] = None
            result['center_y'] = None

        # Calcular distancia entre manos
        if result['both_detected']:
            dx = result['right_x'] - result['left_x']
            dy = result['right_y'] - result['left_y']
            result['distance'] = math.sqrt(dx * dx + dy * dy)
            result['separation_x'] = abs(dx)
            result['separation_y'] = abs(dy)
        else:
            result['distance'] = None
            result['separation_x'] = None
            result['separation_y'] = None

        return result


class ControlMapping:
    """Define una relación entre entrada y salida con procesamiento"""

    def __init__(self, name, input_extractor, output_applier, smoother=None, enabled=True):
        """
        Args:
            name: Nombre descriptivo del mapeo
            input_extractor: Función que extrae valor de los datos de entrada
            output_applier: Función que aplica el valor procesado al estado de salida
            smoother: Instancia de Smoother opcional
            enabled: Si el mapeo está activo
        """
        self.name = name
        self.input_extractor = input_extractor
        self.output_applier = output_applier
        self.smoother = smoother
        self.enabled = enabled

    def process(self, input_data, output_state):
        """Procesa un mapeo completo de entrada a salida"""
        if not self.enabled:
            return

        # Extraer valor de entrada
        raw_value = self.input_extractor(input_data)

        # Aplicar suavizado si está disponible
        if self.smoother:
            processed_value = self.smoother.update(raw_value)
        else:
            processed_value = raw_value

        # Aplicar al estado de salida
        if processed_value is not None:
            self.output_applier(processed_value, output_state)


class ControlSystem:
    """Sistema de control que procesa entrada y genera salida para el visualizador"""

    def __init__(self):
        self.mappings = []
        self.data_processor = DataProcessor()

    def add_mapping(self, mapping):
        """Añade un mapeo de control"""
        self.mappings.append(mapping)

    def remove_mapping(self, name):
        """Remueve un mapeo por nombre"""
        self.mappings = [m for m in self.mappings if m.name != name]

    def enable_mapping(self, name, enabled=True):
        """Habilita/deshabilita un mapeo"""
        for mapping in self.mappings:
            if mapping.name == name:
                mapping.enabled = enabled

    def process_input(self, pose_data):
        """
        Procesa datos de pose y genera comandos para el visualizador

        Args:
            pose_data: Datos del detector de pose

        Returns:
            Dict con parámetros para visualizer.update_model()
        """
        # Enriquecer datos de entrada
        enriched_data = self._enrich_input_data(pose_data)

        # Estado de salida inicial
        output_state = {
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0],  # [yaw, pitch, roll]
            'scale': 1.0
        }

        # Procesar cada mapeo
        for mapping in self.mappings:
            mapping.process(enriched_data, output_state)

        return output_state

    def _enrich_input_data(self, pose_data):
        """Enriquece los datos de pose con información procesada"""
        enriched = pose_data.copy()

        # Procesar manos si están disponibles
        left_hand = pose_data.get('left_hand', {})
        right_hand = pose_data.get('right_hand', {})
        enriched['hands'] = self.data_processor.process_hands(left_hand, right_hand)

        return enriched

    def get_mappings_info(self):
        """Retorna información sobre los mapeos activos"""
        info = []
        for m in self.mappings:
            mapping_type = "Hybrid" if isinstance(m, HybridRotationMapping) else "Standard"
            info.append(f"{'✓' if m.enabled else '✗'} {m.name} ({mapping_type})")
        return info


# ===============================
# FACTORY FUNCTIONS
# ===============================

def create_extractors():
    """Factory de funciones extractoras comunes"""

    def head_x(data):
        head = data.get('head', {})
        return head.get('x') if head.get('detected') else None

    def head_y(data):
        head = data.get('head', {})
        return head.get('y') if head.get('detected') else None

    def hands_center_x(data):
        hands = data.get('hands', {})
        return hands.get('center_x')

    def hands_center_y(data):
        hands = data.get('hands', {})
        return hands.get('center_y')

    def hands_distance(data):
        hands = data.get('hands', {})
        return hands.get('distance')

    def hands_separation_x(data):
        hands = data.get('hands', {})
        return hands.get('separation_x')

    return {
        'head_x': head_x,
        'head_y': head_y,
        'hands_center_x': hands_center_x,
        'hands_center_y': hands_center_y,
        'hands_distance': hands_distance,
        'hands_separation_x': hands_separation_x,
    }


def create_appliers():
    """Factory de funciones aplicadoras comunes"""

    def rotation_yaw(max_degrees=30.0, center=0.5, invert=False):
        def applier(value, output_state):
            offset = value - center
            if invert:
                offset = -offset
            rotation = offset * 2.0 * max_degrees
            output_state['rotation'][0] = max(-max_degrees, min(max_degrees, rotation))

        return applier

    def rotation_pitch(max_degrees=30.0, center=0.5, invert=False):
        def applier(value, output_state):
            offset = value - center
            if invert:
                offset = -offset
            rotation = offset * 2.0 * max_degrees
            output_state['rotation'][1] = max(-max_degrees, min(max_degrees, rotation))

        return applier

    def rotation_roll(max_degrees=30.0, center=0.5, invert=False):
        def applier(value, output_state):
            offset = value - center
            if invert:
                offset = -offset
            rotation = offset * 2.0 * max_degrees
            output_state['rotation'][2] = max(-max_degrees, min(max_degrees, rotation))

        return applier

    def position_x(scale=10.0, center=0.5, invert=False):
        def applier(value, output_state):
            offset = value - center
            if invert:
                offset = -offset
            output_state['position'][0] = offset * scale

        return applier

    def position_y(scale=10.0, center=0.5, invert=False):
        def applier(value, output_state):
            offset = value - center
            if invert:
                offset = -offset
            output_state['position'][1] = offset * scale

        return applier

    def position_z(scale=10.0, center=0.5, invert=False):
        def applier(value, output_state):
            offset = value - center
            if invert:
                offset = -offset
            output_state['position'][2] = offset * scale

        return applier

    def scale_uniform(min_scale=0.3, max_scale=3.0, center_distance=0.3):
        def applier(value, output_state):
            if value is None:
                output_state['scale'] = 1.0
            else:
                # Mapear distancia a escala
                if value <= center_distance:
                    # Distancia pequeña -> escala pequeña
                    scale_factor = value / center_distance
                    output_state['scale'] = min_scale + scale_factor * (1.0 - min_scale)
                else:
                    # Distancia grande -> escala grande
                    scale_factor = (value - center_distance) / (1.0 - center_distance)
                    output_state['scale'] = 1.0 + scale_factor * (max_scale - 1.0)

                # Limitar rango
                output_state['scale'] = max(min_scale, min(max_scale, output_state['scale']))

        return applier

    return {
        'rotation_yaw': rotation_yaw,
        'rotation_pitch': rotation_pitch,
        'rotation_roll': rotation_roll,
        'position_x': position_x,
        'position_y': position_y,
        'position_z': position_z,
        'scale_uniform': scale_uniform,
    }


# En tu control_system.py, modifica estas clases:

class HybridRotationMapping:
    """
    Mapeo híbrido para rotación proporcional + continua en ROLL
    """

    def __init__(self, name, rotation_controller, smoother=None, enabled=True):
        self.name = name
        self.rotation_controller = rotation_controller
        self.smoother = smoother
        self.enabled = enabled

    def process(self, input_data, output_state):
        """Procesa rotación híbrida aplicada al ROLL"""
        if not self.enabled:
            return

        # Extraer posición X de cabeza
        head = input_data.get('head', {})
        head_x = head.get('x') if head.get('detected') else None

        # Aplicar suavizado básico si está disponible
        if self.smoother:
            head_x = self.smoother.update(head_x)

        # Actualizar rotación híbrida
        current_rotation = self.rotation_controller.update(head_x)

        # CAMBIO: Aplicar al ROLL (índice 2) en lugar de YAW (índice 0)
        output_state['rotation'][2] = current_rotation


def create_default_control_system():
    """
    Sistema de control híbrido que combina "asomar" + rotación continua en ROLL
    """
    system = ControlSystem()
    extractors = create_extractors()
    appliers = create_appliers()

    # Crear smoothers
    head_x_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.2, center_value=0.5)
    head_y_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.2, center_value=0.5)
    head_x_yaw_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.2, center_value=0.5)  # Para YAW normal
    hands_dist_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.2, center_value=0.3)

    # Controlador híbrido para ROLL
    hybrid_roll_controller = HybridRotationController(
        max_degrees=30.0,  # Rotación máxima en zona normal
        left_threshold=0.25,  # 25% umbral izquierdo
        right_threshold=0.75,  # 75% umbral derecho
        continuous_speed_degrees_per_second=100.0,  # Velocidad rotación continua
        center=0.5,
        invert=True  # Mantener el comportamiento original
    )

    # MAPEO HÍBRIDO: Cabeza X -> Rotación ROLL (normal + continua)
    system.add_mapping(HybridRotationMapping(
        name="head_x_hybrid_roll",
        rotation_controller=hybrid_roll_controller,
        smoother=head_x_smoother
    ))

    # MAPEO NORMAL: Cabeza X -> Rotación YAW (solo proporcional)
    system.add_mapping(ControlMapping(
        name="head_x_to_yaw",
        input_extractor=extractors['head_x'],
        output_applier=appliers['rotation_yaw'](max_degrees=10.0, invert=True),
        smoother=head_x_yaw_smoother
    ))

    # Mapeos normales mantenidos
    system.add_mapping(ControlMapping(
        name="head_y_to_pitch",
        input_extractor=extractors['head_y'],
        output_applier=appliers['rotation_pitch'](max_degrees=30.0),
        smoother=head_y_smoother
    ))

    # NOTA: Removí head_x_to_roll ya que ahora el roll lo maneja el controlador híbrido

    # Mapeo: Distancia entre manos -> Escala
    system.add_mapping(ControlMapping(
        name="hands_distance_to_scale",
        input_extractor=extractors['hands_distance'],
        output_applier=appliers['scale_uniform'](min_scale=0.3, max_scale=2.5),
        smoother=hands_dist_smoother
    ))

    return system
