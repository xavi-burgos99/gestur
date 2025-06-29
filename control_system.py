# control_system.py
import math
import time
from abc import ABC, abstractmethod


class Smoother(ABC):
    @abstractmethod
    def update(self, value):
        pass

    @abstractmethod
    def reset(self):
        pass


class ExponentialSmoother(Smoother):
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


class HybridRotationController:
    def __init__(self, max_degrees=30.0, left_threshold=0.25, right_threshold=0.75,
                 continuous_speed_degrees_per_second=45.0, center=0.5, invert=False,
                 reset_timeout_seconds=3.0):
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
        self.continuous_direction = 0
        self.proportional_rotation_at_threshold = 0.0

        # Estado para reseteo completo
        self.is_resetting = False
        self.reset_start_time = None
        self.reset_start_rotation = 0.0

    def update(self, head_x_value):
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time

        if head_x_value is None:
            # **INICIAR O CONTINUAR RESETEO**
            if not self.is_resetting:
                self.is_resetting = True
                self.reset_start_time = current_time
                self.reset_start_rotation = self.continuous_rotation + self.proportional_rotation_at_threshold

            # **RESETEO GRADUAL A 0 GRADOS**
            reset_duration = 1.0  # 1 segundo para llegar a 0
            reset_elapsed = current_time - self.reset_start_time

            if reset_elapsed >= reset_duration:
                # Reseteo completado
                self.continuous_rotation = 0.0
                self.proportional_rotation_at_threshold = 0.0
                self.is_in_continuous_mode = False
                self.continuous_direction = 0
                return 0.0
            else:
                # Interpolación suave hacia 0
                progress = reset_elapsed / reset_duration
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                current_rotation = self.reset_start_rotation * (1.0 - smooth_progress)
                return current_rotation

        # **CABEZA DETECTADA - CANCELAR RESETEO**
        self.is_resetting = False

        # **RESTO DE LA LÓGICA NORMAL**
        in_left_extreme = head_x_value < self.left_threshold
        in_right_extreme = head_x_value > self.right_threshold
        in_normal_zone = self.left_threshold <= head_x_value <= self.right_threshold

        if in_normal_zone:
            if self.is_in_continuous_mode:
                self.is_in_continuous_mode = False
                self.continuous_direction = 0

            offset = head_x_value - self.center
            if self.invert:
                offset = -offset
            proportional_rotation = offset * 2.0 * self.max_degrees
            proportional_rotation = max(-self.max_degrees, min(self.max_degrees, proportional_rotation))
            return self.continuous_rotation + proportional_rotation
        else:
            current_direction = -1 if in_left_extreme else 1

            if not self.is_in_continuous_mode:
                self.is_in_continuous_mode = True
                self.continuous_direction = current_direction
                threshold_value = self.left_threshold if in_left_extreme else self.right_threshold
                offset = threshold_value - self.center
                if self.invert:
                    offset = -offset
                self.proportional_rotation_at_threshold = offset * 2.0 * self.max_degrees
                self.proportional_rotation_at_threshold = max(-self.max_degrees,
                                                              min(self.max_degrees,
                                                                  self.proportional_rotation_at_threshold))
            elif self.continuous_direction != current_direction:
                self.continuous_direction = current_direction

            speed_factor = self.calculate_continuous_speed_factor(head_x_value)
            current_speed = self.max_continuous_speed * speed_factor

            if self.is_in_continuous_mode and current_speed > 0:
                rotation_increment = self.continuous_direction * current_speed * delta_time
                if self.invert:
                    rotation_increment = -rotation_increment
                self.continuous_rotation += rotation_increment

                while self.continuous_rotation > 360:
                    self.continuous_rotation -= 360
                while self.continuous_rotation < -360:
                    self.continuous_rotation += 360

            return self.continuous_rotation + self.proportional_rotation_at_threshold

    def calculate_continuous_speed_factor(self, head_x_value):
        if head_x_value > self.right_threshold:
            distance_from_threshold = head_x_value - self.right_threshold
            max_distance = 1.0 - self.right_threshold
            factor = min(1.0, distance_from_threshold / max_distance)
            return factor
        elif head_x_value < self.left_threshold:
            distance_from_threshold = self.left_threshold - head_x_value
            max_distance = self.left_threshold
            factor = min(1.0, distance_from_threshold / max_distance)
            return factor
        else:
            return 0.0


class DataProcessor:
    def __init__(self):
        pass

    def process_hands(self, left_hand_data, right_hand_data):
        left_detected = left_hand_data.get('detected', False)
        right_detected = right_hand_data.get('detected', False)

        result = {
            'left_detected': left_detected,
            'right_detected': right_detected,
            'any_detected': left_detected or right_detected,
            'both_detected': left_detected and right_detected,
            'count': sum([left_detected, right_detected])
        }

        # Datos de manos
        if left_detected:
            result['left_x'] = left_hand_data.get('x')
            result['left_y'] = left_hand_data.get('y')
        else:
            result['left_x'] = None
            result['left_y'] = None

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
    def __init__(self, name, input_extractor, output_applier, smoother=None, enabled=True):
        self.name = name
        self.input_extractor = input_extractor
        self.output_applier = output_applier
        self.smoother = smoother
        self.enabled = enabled

    def process(self, input_data, output_state):
        if not self.enabled:
            return

        raw_value = self.input_extractor(input_data)

        if self.smoother:
            processed_value = self.smoother.update(raw_value)
        else:
            processed_value = raw_value

        if processed_value is not None:
            self.output_applier(processed_value, output_state)


class ControlSystem:
    def __init__(self):
        self.mappings = []
        self.data_processor = DataProcessor()

    def add_mapping(self, mapping):
        self.mappings.append(mapping)

    def remove_mapping(self, name):
        self.mappings = [m for m in self.mappings if m.name != name]

    def enable_mapping(self, name, enabled=True):
        for mapping in self.mappings:
            if mapping.name == name:
                mapping.enabled = enabled

    def process_input(self, pose_data):
        enriched_data = self._enrich_input_data(pose_data)

        output_state = {
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0],
            'scale': 1.0
        }

        for mapping in self.mappings:
            mapping.process(enriched_data, output_state)

        return output_state

    def _enrich_input_data(self, pose_data):
        enriched = pose_data.copy()
        left_hand = pose_data.get('left_hand', {})
        right_hand = pose_data.get('right_hand', {})
        enriched['hands'] = self.data_processor.process_hands(left_hand, right_hand)
        return enriched

    def get_mappings_info(self):
        info = []
        for m in self.mappings:
            mapping_type = "Hybrid" if isinstance(m, HybridRotationMapping) else "Standard"
            info.append(f"{'✓' if m.enabled else '✗'} {m.name} ({mapping_type})")
        return info


def create_extractors():
    def head_x(data):
        head = data.get('head', {})
        return head.get('x') if head.get('detected') else None

    def head_y(data):
        head = data.get('head', {})
        return head.get('y') if head.get('detected') else None

    def head_scale(data):
        head = data.get('head', {})
        return head.get('scale') if head.get('detected') else None

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
        'head_scale': head_scale,
        'hands_center_x': hands_center_x,
        'hands_center_y': hands_center_y,
        'hands_distance': hands_distance,
        'hands_separation_x': hands_separation_x,
    }


def create_appliers():
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
                if value <= center_distance:
                    scale_factor = value / center_distance
                    output_state['scale'] = min_scale + scale_factor * (1.0 - min_scale)
                else:
                    scale_factor = (value - center_distance) / (1.0 - center_distance)
                    output_state['scale'] = 1.0 + scale_factor * (max_scale - 1.0)
                output_state['scale'] = max(min_scale, min(max_scale, output_state['scale']))

        return applier

    def scale_stepped_timed(threshold=0.5, small_scale=1.0, large_scale=2.0, transition_time_ms=200):
        state = {
            'current_target': small_scale,
            'transition_start_time': None,
            'transition_start_value': small_scale,
            'transition_end_value': small_scale,
            'is_transitioning': False,
            'last_output_scale': small_scale
        }

        transition_time_seconds = transition_time_ms / 1000.0

        def applier(value, output_state):
            if value is None:
                output_state['scale'] = small_scale
                state['current_target'] = small_scale
                state['is_transitioning'] = False
                state['last_output_scale'] = small_scale
                return

            current_time = time.time()
            target_scale = large_scale if value > threshold else small_scale

            if target_scale != state['current_target']:
                state['current_target'] = target_scale
                state['transition_start_time'] = current_time
                state['transition_start_value'] = state['last_output_scale']
                state['transition_end_value'] = target_scale
                state['is_transitioning'] = True

            if state['is_transitioning']:
                elapsed_time = current_time - state['transition_start_time']
                progress = min(1.0, elapsed_time / transition_time_seconds)
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)

                current_scale = (state['transition_start_value'] +
                                 smooth_progress * (state['transition_end_value'] - state['transition_start_value']))

                if progress >= 1.0:
                    state['is_transitioning'] = False
                    current_scale = state['transition_end_value']

                state['last_output_scale'] = current_scale
                output_state['scale'] = current_scale
            else:
                state['last_output_scale'] = target_scale
                output_state['scale'] = target_scale

        return applier

    return {
        'rotation_yaw': rotation_yaw,
        'rotation_pitch': rotation_pitch,
        'rotation_roll': rotation_roll,
        'position_x': position_x,
        'position_y': position_y,
        'position_z': position_z,
        'scale_uniform': scale_uniform,
        'scale_stepped': scale_stepped_timed,
    }


class HybridRotationMapping:
    def __init__(self, name, rotation_controller, smoother=None, enabled=True):
        self.name = name
        self.rotation_controller = rotation_controller
        self.smoother = smoother
        self.enabled = enabled

        # **TIMEOUT INDEPENDIENTE PARA RESETEO**
        self.last_real_detection_time = time.time()
        self.reset_timeout_seconds = 3.0

    def process(self, input_data, output_state):
        if not self.enabled:
            return

        head = input_data.get('head', {})
        head_detected = head.get('detected', False)
        head_x_raw = head.get('x') if head_detected else None

        current_time = time.time()

        # **ACTUALIZAR TIEMPO DE DETECCIÓN REAL**
        if head_detected and head_x_raw is not None:
            self.last_real_detection_time = current_time

        # **VERIFICAR SI HAN PASADO 3 SEGUNDOS SIN DETECCIÓN**
        time_since_real_detection = current_time - self.last_real_detection_time
        timeout_reached = time_since_real_detection >= self.reset_timeout_seconds

        # **PREPARAR VALOR PARA EL CONTROLADOR**
        if timeout_reached:
            # **DESPUÉS DE 3 SEGUNDOS: Enviar None para activar reseteo**
            head_x_for_controller = None
        else:
            # **ANTES DE 3 SEGUNDOS: Usar smoother normalmente**
            if self.smoother and head_x_raw is not None:
                head_x_for_controller = self.smoother.update(head_x_raw)
            elif self.smoother:
                # Permitir que el smoother maneje la interpolación
                head_x_for_controller = self.smoother.update(head_x_raw)
            else:
                head_x_for_controller = head_x_raw

        # **ACTUALIZAR CONTROLADOR**
        current_rotation = self.rotation_controller.update(head_x_for_controller)
        output_state['rotation'][2] = current_rotation


def create_default_control_system():
    system = ControlSystem()
    extractors = create_extractors()
    appliers = create_appliers()

    # Smoothers
    head_x_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.2, center_value=0.5)
    head_y_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.2, center_value=0.5)
    head_x_yaw_smoother = ExponentialSmoother(alpha=0.3, decay_rate=0.2, center_value=0.5)
    head_scale_smoother = ExponentialSmoother(alpha=0.7, decay_rate=0.3, center_value=0.3)

    # Controlador híbrido para ROLL con timeout de 3 segundos
    hybrid_roll_controller = HybridRotationController(
        max_degrees=30.0,
        left_threshold=0.25,
        right_threshold=0.75,
        continuous_speed_degrees_per_second=100.0,
        center=0.5,
        invert=True,
        reset_timeout_seconds=3
    )

    # Mapeos
    system.add_mapping(HybridRotationMapping(
        name="head_x_hybrid_roll",
        rotation_controller=hybrid_roll_controller,
        smoother=head_x_smoother
    ))

    system.add_mapping(ControlMapping(
        name="head_x_to_yaw",
        input_extractor=extractors['head_x'],
        output_applier=appliers['rotation_yaw'](max_degrees=10.0, invert=True),
        smoother=head_x_yaw_smoother
    ))

    system.add_mapping(ControlMapping(
        name="head_y_to_pitch",
        input_extractor=extractors['head_y'],
        output_applier=appliers['rotation_pitch'](max_degrees=30.0),
        smoother=head_y_smoother
    ))

    system.add_mapping(ControlMapping(
        name="head_scale_to_model_scale",
        input_extractor=extractors['head_scale'],
        output_applier=appliers['scale_stepped'](
            threshold=0.4,
            small_scale=1.0,
            large_scale=1.75,
            transition_time_ms=750
        ),
        smoother=head_scale_smoother
    ))

    return system
