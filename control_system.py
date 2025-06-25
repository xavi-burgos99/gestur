# control_system.py
import math
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
        return [f"{'✓' if m.enabled else '✗'} {m.name}" for m in self.mappings]


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
        'hands_separation_x': hands_separation_x
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
        'scale_uniform': scale_uniform
    }


def create_default_control_system():
    """Crea un sistema de control con configuración por defecto"""
    system = ControlSystem()
    extractors = create_extractors()
    appliers = create_appliers()

    # Crear smoothers
    head_x_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.1, center_value=0.5)
    head_y_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.1, center_value=0.5)
    hands_x_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.1, center_value=0.5)
    hands_y_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.1, center_value=0.5)
    hands_dist_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.1, center_value=0.3)

    # Mapeos: Cabeza -> Rotación
    system.add_mapping(ControlMapping(
        name="head_x_to_yaw",
        input_extractor=extractors['head_x'],
        output_applier=appliers['rotation_yaw'](max_degrees=10.0, invert=True),
        smoother=head_x_smoother
    ))

    system.add_mapping(ControlMapping(
        name="head_y_to_pitch",
        input_extractor=extractors['head_y'],
        output_applier=appliers['rotation_pitch'](max_degrees=10.0),
        smoother=head_y_smoother
    ))

    system.add_mapping(ControlMapping(
        name="head_x_to_roll",
        input_extractor=extractors['head_x'],
        output_applier=appliers['rotation_roll'](max_degrees=45.0, invert=True),
        smoother=head_x_smoother
    ))

    # Mapeos: Manos -> Posición
    #system.add_mapping(ControlMapping(
    #    name="hands_x_to_position_x",
    #    input_extractor=extractors['hands_center_x'],
    #    output_applier=appliers['position_x'](scale=8.0, invert=True),
    #    smoother=hands_x_smoother
    #))

    #system.add_mapping(ControlMapping(
    #    name="hands_y_to_position_z",
    #    input_extractor=extractors['hands_center_y'],
    #    output_applier=appliers['position_z'](scale=8.0, invert=True),
    #    smoother=hands_y_smoother
    #))

    # Mapeo: Distancia entre manos -> Escala
    #system.add_mapping(ControlMapping(
    #    name="hands_distance_to_scale",
    #    input_extractor=extractors['hands_distance'],
    #    output_applier=appliers['scale_uniform'](min_scale=0.3, max_scale=2.5),
    #    smoother=hands_dist_smoother
    #))

    return system


def create_gaming_control_system():
    """Sistema de control optimizado para gaming (más sensible)"""
    system = ControlSystem()
    extractors = create_extractors()
    appliers = create_appliers()

    # Smoothers más rápidos para gaming
    head_x_smoother = ExponentialSmoother(alpha=0.6, decay_rate=0.2, center_value=0.5)
    head_y_smoother = ExponentialSmoother(alpha=0.6, decay_rate=0.2, center_value=0.5)
    hands_x_smoother = ExponentialSmoother(alpha=0.5, decay_rate=0.15, center_value=0.5)
    hands_y_smoother = ExponentialSmoother(alpha=0.5, decay_rate=0.15, center_value=0.5)

    # Rotaciones más amplias para gaming
    system.add_mapping(ControlMapping(
        name="head_x_to_yaw_gaming",
        input_extractor=extractors['head_x'],
        output_applier=appliers['rotation_yaw'](max_degrees=60.0),
        smoother=head_x_smoother
    ))

    system.add_mapping(ControlMapping(
        name="head_y_to_pitch_gaming",
        input_extractor=extractors['head_y'],
        output_applier=appliers['rotation_pitch'](max_degrees=45.0),
        smoother=head_y_smoother
    ))

    # Movimiento más amplio
    #system.add_mapping(ControlMapping(
    #    name="hands_x_to_position_x_gaming",
    #    input_extractor=extractors['hands_center_x'],
    #    output_applier=appliers['position_x'](scale=15.0, invert=True),
    #    smoother=hands_x_smoother
    #))

    #system.add_mapping(ControlMapping(
    #    name="hands_y_to_position_z_gaming",
    #    input_extractor=extractors['hands_center_y'],
    #    output_applier=appliers['position_z'](scale=12.0, invert=True),
    #    smoother=hands_y_smoother
    #))

    return system


def create_precise_control_system():
    """Sistema de control para trabajo de precisión (más suave)"""
    system = ControlSystem()
    extractors = create_extractors()
    appliers = create_appliers()

    # Smoothers más suaves para precisión
    head_x_smoother = ExponentialSmoother(alpha=0.15, decay_rate=0.05, center_value=0.5)
    head_y_smoother = ExponentialSmoother(alpha=0.15, decay_rate=0.05, center_value=0.5)
    hands_x_smoother = ExponentialSmoother(alpha=0.2, decay_rate=0.05, center_value=0.5)
    hands_y_smoother = ExponentialSmoother(alpha=0.2, decay_rate=0.05, center_value=0.5)
    hands_dist_smoother = ExponentialSmoother(alpha=0.2, decay_rate=0.05, center_value=0.3)

    # Rotaciones más sutiles
    system.add_mapping(ControlMapping(
        name="head_x_to_yaw_precise",
        input_extractor=extractors['head_x'],
        output_applier=appliers['rotation_yaw'](max_degrees=15.0),
        smoother=head_x_smoother
    ))

    system.add_mapping(ControlMapping(
        name="head_y_to_pitch_precise",
        input_extractor=extractors['head_y'],
        output_applier=appliers['rotation_pitch'](max_degrees=15.0),
        smoother=head_y_smoother
    ))

    # Movimiento más controlado
    system.add_mapping(ControlMapping(
        name="hands_x_to_position_x_precise",
        input_extractor=extractors['hands_center_x'],
        output_applier=appliers['position_x'](scale=5.0, invert=True),
        smoother=hands_x_smoother
    ))

    #system.add_mapping(ControlMapping(
    #    name="hands_y_to_position_z_precise",
    #    input_extractor=extractors['hands_center_y'],
    #    output_applier=appliers['position_z'](scale=5.0, invert=True),
    #    smoother=hands_y_smoother
    #))

    # Escala más sutil
    #system.add_mapping(ControlMapping(
    #    name="hands_distance_to_scale_precise",
    #    input_extractor=extractors['hands_distance'],
    #    output_applier=appliers['scale_uniform'](min_scale=0.7, max_scale=1.5),
    #    smoother=hands_dist_smoother
    #))

    return system
