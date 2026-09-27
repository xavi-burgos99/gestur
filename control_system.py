"""Frame-rate independent controls for GESTUR's latest tracking snapshot.

The render thread owns these objects; detector callbacks only replace snapshots.
All timing uses a monotonic clock, which can be replaced in deterministic tests.
"""
import math
import time
from abc import ABC, abstractmethod


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _shortest_delta(target, current, period):
    return (target - current + period / 2) % period - period / 2


class Smoother(ABC):
    @abstractmethod
    def update(self, value, now=None):
        pass

    @abstractmethod
    def reset(self):
        pass


class ExponentialSmoother(Smoother):
    """An exponential filter whose response does not change with render FPS.

    alpha/decay_rate retain their old meaning at 30 Hz. smoothing_ms optionally
    specifies the time constant directly. period enables circular angle inputs.
    decay_period allows an unoriented measurement to return to an oriented
    neutral pose instead of stopping at the opposite orientation.
    """
    def __init__(self, alpha=0.3, decay_rate=0.1, center_value=0.5,
                 smoothing_ms=None, clock=time.monotonic, period=None,
                 decay_period=None):
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.center_value = center_value
        self.clock = clock
        self.period = period
        self.decay_period = period if decay_period is None else decay_period
        self.time_constant = (smoothing_ms / 1000.0 if smoothing_ms is not None
                              else self._time_constant(alpha))
        self.decay_constant = self._time_constant(decay_rate)
        self.reset()

    @staticmethod
    def _time_constant(alpha):
        return 0.0 if alpha >= 1 else -1 / (30 * math.log(1 - max(1e-9, alpha)))

    def update(self, value, now=None):
        now = self.clock() if now is None else now
        elapsed = max(0.0, now - self.last_update_time)
        self.last_update_time = now
        value = _number(value)
        if value is not None and not self.has_data:
            self.smoothed_value = value
            self.has_data = True
            return self.smoothed_value
        target = self.center_value if value is None else value
        constant = self.decay_constant if value is None else self.time_constant
        weight = 1.0 if constant <= 0 else -math.expm1(-elapsed / constant)
        period = self.decay_period if value is None else self.period
        delta = (target - self.smoothed_value if period is None
                 else _shortest_delta(target, self.smoothed_value, period))
        self.smoothed_value += weight * delta
        if value is not None:
            self.has_data = True
        elif abs(delta) < 1e-4:
            # For circular inputs preserve the equivalent unwrapped center.
            self.smoothed_value += delta
            self.has_data = False
        return self.smoothed_value

    def reset(self):
        self.smoothed_value = self.center_value
        self.has_data = False
        self.last_update_time = self.clock()


class HybridRotationController:
    def __init__(self, max_degrees=30.0, left_threshold=0.25, right_threshold=0.75,
                 continuous_speed_degrees_per_second=45.0, center=0.5, invert=False,
                 reset_timeout_seconds=3.0, reset_duration_seconds=1.0,
                 clock=time.monotonic):
        self.max_degrees = max_degrees
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold
        self.max_continuous_speed = continuous_speed_degrees_per_second
        self.center = center
        self.invert = invert
        self.reset_timeout_seconds = reset_timeout_seconds
        self.reset_duration_seconds = reset_duration_seconds
        self.clock = clock
        self.last_update_time = clock()
        self.continuous_rotation = 0.0
        self.current_rotation = 0.0
        self.is_resetting = False
        self.is_in_continuous_mode = False
        self.continuous_direction = 0
        self.proportional_rotation_at_threshold = 0.0
        self.reset_start_time = None
        self.reset_start_rotation = 0.0
        self.reset_target_rotation = 0.0
        self._held = False
        self._has_input = False

    def _proportional(self, value):
        offset = (_clamp(value, self.left_threshold, self.right_threshold) - self.center)
        sign = -1 if self.invert else 1
        return _clamp(offset * 2.0 * self.max_degrees * sign,
                      -self.max_degrees, self.max_degrees)

    def hold(self, now=None):
        """Stop angular velocity immediately during a brief tracking loss."""
        self.last_update_time = self.clock() if now is None else now
        self._held = True
        return self.current_rotation

    def update(self, head_x_value, now=None):
        now = self.clock() if now is None else now
        # A long UI stall must not turn into a large accumulated rotation jump.
        elapsed = _clamp(now - self.last_update_time, 0.0, 0.1)
        self.last_update_time = now
        if head_x_value is None:
            if not self._has_input:
                return self.current_rotation
            if not self.is_resetting:
                self.is_resetting = True
                self.reset_start_time = now
                self.reset_start_rotation = self.current_rotation
                self.reset_target_rotation = self.current_rotation + _shortest_delta(0.0, self.current_rotation, 360.0)
            progress = _clamp((now - self.reset_start_time) / self.reset_duration_seconds, 0.0, 1.0)
            eased = progress * progress * (3.0 - 2.0 * progress)
            self.current_rotation = self.reset_start_rotation + eased * (self.reset_target_rotation - self.reset_start_rotation)
            self.continuous_rotation = self.current_rotation
            self.proportional_rotation_at_threshold = 0.0
            self.is_in_continuous_mode = False
            self.continuous_direction = 0
            return self.current_rotation

        value = _clamp(head_x_value, 0.0, 1.0)
        proportional = self._proportional(value)
        if self._has_input and (self.is_resetting or self._held):
            # Rebase at the currently visible angle, so re-acquisition cannot
            # jump back to the pre-reset accumulated rotation.
            self.continuous_rotation = self.current_rotation - proportional
            elapsed = 0.0
        self.is_resetting = False
        self._held = False
        self._has_input = True
        self.is_in_continuous_mode = value < self.left_threshold or value > self.right_threshold
        self.continuous_direction = -1 if value < self.left_threshold else 1 if value > self.right_threshold else 0
        self.proportional_rotation_at_threshold = proportional
        speed = self.max_continuous_speed * self.calculate_continuous_speed_factor(value)
        self.continuous_rotation += self.continuous_direction * speed * elapsed * (-1 if self.invert else 1)
        # Keep rotations unwrapped: wrapping at 360 can make interpolators spin.
        self.current_rotation = self.continuous_rotation + proportional
        return self.current_rotation

    def calculate_continuous_speed_factor(self, head_x_value):
        if head_x_value > self.right_threshold:
            return min(1.0, (head_x_value - self.right_threshold) / (1.0 - self.right_threshold))
        if head_x_value < self.left_threshold:
            return min(1.0, (self.left_threshold - head_x_value) / self.left_threshold)
        return 0.0


class DataProcessor:
    def process_hands(self, left_hand_data, right_hand_data):
        left = left_hand_data or {}
        right = right_hand_data or {}
        result = {}
        for name, hand in (("left", left), ("right", right)):
            detected = bool(hand.get("detected"))
            result[f"{name}_detected"] = detected
            result[f"{name}_x"] = _number(hand.get("x")) if detected else None
            result[f"{name}_y"] = _number(hand.get("y")) if detected else None
        result["any_detected"] = result["left_detected"] or result["right_detected"]
        result["both_detected"] = result["left_detected"] and result["right_detected"]
        result["count"] = int(result["left_detected"]) + int(result["right_detected"])
        for axis in ("x", "y"):
            valid = [result[f"{side}_{axis}"] for side in ("left", "right") if result[f"{side}_{axis}"] is not None]
            result[f"center_{axis}"] = sum(valid) / len(valid) if valid else None
        result.update(distance=None, separation_x=None, separation_y=None)
        if result["both_detected"] and all(result[f"{side}_{axis}"] is not None for side in ("left", "right") for axis in ("x", "y")):
            dx = result["right_x"] - result["left_x"]
            dy = result["right_y"] - result["left_y"]
            result.update(distance=math.hypot(dx, dy), separation_x=abs(dx), separation_y=abs(dy))
        return result


class ControlMapping:
    def __init__(self, name, input_extractor, output_applier, smoother=None, enabled=True):
        self.name = name
        self.input_extractor = input_extractor
        self.output_applier = output_applier
        self.smoother = smoother
        self.enabled = enabled

    def process(self, input_data, output_state, now=None):
        if not self.enabled:
            return
        value = self.input_extractor(input_data)
        if self.smoother:
            value = self.smoother.update(value, now=now)
        # Appliers also consume None to return scale to its neutral value.
        self.output_applier(value, output_state)


class HybridRotationMapping:
    def __init__(self, name, rotation_controller, smoother=None, enabled=True,
                 input_extractor=None, output_axis=2, clock=time.monotonic):
        self.name = name
        self.rotation_controller = rotation_controller
        self.smoother = smoother
        self.enabled = enabled
        self.input_extractor = input_extractor or create_extractors()["head_x"]
        self.output_axis = output_axis
        self.clock = clock
        self.last_real_detection_time = None
        self.reset_timeout_seconds = rotation_controller.reset_timeout_seconds

    def process(self, input_data, output_state, now=None):
        if not self.enabled:
            return
        now = self.clock() if now is None else now
        raw_value = self.input_extractor(input_data)
        if raw_value is not None:
            self.last_real_detection_time = now
            value = self.smoother.update(raw_value, now) if self.smoother else raw_value
            rotation = self.rotation_controller.update(value, now)
        else:
            if self.smoother:
                self.smoother.update(None, now)
            elapsed = math.inf if self.last_real_detection_time is None else now - self.last_real_detection_time
            rotation = (self.rotation_controller.update(None, now) if elapsed >= self.reset_timeout_seconds
                        else self.rotation_controller.hold(now))
        output_state["rotation"][self.output_axis] = rotation


class ControlSystem:
    def __init__(self, clock=time.monotonic):
        self.mappings = []
        self.data_processor = DataProcessor()
        self.clock = clock

    def add_mapping(self, mapping):
        self.mappings.append(mapping)

    def remove_mapping(self, name):
        self.mappings = [mapping for mapping in self.mappings if mapping.name != name]

    def enable_mapping(self, name, enabled=True):
        for mapping in self.mappings:
            if mapping.name == name:
                mapping.enabled = enabled

    def process_input(self, pose_data):
        enriched = self._enrich_input_data(pose_data or {})
        output = {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": 1.0}
        now = self.clock()
        for mapping in self.mappings:
            mapping.process(enriched, output, now=now)
        return output

    def _enrich_input_data(self, pose_data):
        enriched = pose_data.copy()
        enriched["hands"] = self.data_processor.process_hands(pose_data.get("left_hand", {}), pose_data.get("right_hand", {}))
        return enriched

    def get_mappings_info(self):
        return [f"{'✓' if mapping.enabled else '✗'} {mapping.name} ({'Hybrid' if isinstance(mapping, HybridRotationMapping) else 'Standard'})"
                for mapping in self.mappings]


def create_extractors():
    def tracked(part, key, normalize_rotation=False):
        def extractor(data):
            item = data.get(part) or {}
            value = _number(item.get(key)) if item.get("detected") else None
            if value is not None and normalize_rotation:
                return ((value + 180.0) % 360.0) / 360.0
            return value
        return extractor

    extractors = {f"head_{axis}": tracked("head", axis) for axis in ("x", "y", "scale")}
    for angle in ("pitch", "yaw", "roll"):
        extractors[f"head_{angle}"] = tracked("head", angle, True)
    for field in ("center_x", "center_y", "distance", "separation_x"):
        extractors[f"hands_{field}"] = lambda data, field=field: _number(data.get("hands", {}).get(field))
    for side in ("left", "right"):
        for angle in ("rotation", "pitch", "yaw"):
            extractors[f"{side}_hand_{angle}"] = tracked(f"{side}_hand", angle, True)
        for field in ("x", "y", "pinch", "openness"):
            extractors[f"{side}_hand_{field}"] = tracked(f"{side}_hand", field)
    return extractors


def create_appliers(clock=time.monotonic):
    def rotation(axis, max_degrees=30.0, center=0.5, invert=False, circular=False):
        def apply(value, output):
            value = center if value is None else value
            degrees = (value - center) * 2.0 * max_degrees * (-1 if invert else 1)
            output["rotation"][axis] = degrees if circular else _clamp(degrees, -max_degrees, max_degrees)
        return apply

    def position(axis, scale=10.0, center=0.5, invert=False):
        def apply(value, output):
            value = center if value is None else value
            output["position"][axis] = (value - center) * scale * (-1 if invert else 1)
        return apply

    def scale_uniform(min_scale=0.3, max_scale=3.0, center_distance=0.3):
        def apply(value, output):
            value = center_distance if value is None else value
            if value <= center_distance:
                result = min_scale + value / max(center_distance, 1e-9) * (1.0 - min_scale)
            else:
                result = 1.0 + (value - center_distance) / max(1.0 - center_distance, 1e-9) * (max_scale - 1.0)
            output["scale"] = _clamp(result, min_scale, max_scale)
        return apply

    def scale_stepped_timed(threshold=0.5, small_scale=1.0, large_scale=2.0,
                            transition_time_ms=200, hysteresis=0.02):
        state = {"large": False, "value": small_scale, "start": small_scale,
                 "target": small_scale, "time": clock()}
        duration = transition_time_ms / 1000.0

        def apply(value, output):
            now = clock()
            progress = 1.0 if duration <= 0 else _clamp((now - state["time"]) / duration, 0.0, 1.0)
            eased = progress * progress * (3 - 2 * progress)
            current = state["start"] + eased * (state["target"] - state["start"])
            large = False if value is None else state["large"]
            if value is not None:
                if value > threshold + hysteresis:
                    large = True
                elif value < threshold - hysteresis:
                    large = False
            target = large_scale if large else small_scale
            if target != state["target"]:
                state.update(large=large, start=current, target=target, time=now)
                if duration <= 0:
                    current = target
            state["value"] = current
            output["scale"] = current
        return apply

    return {
        "rotation_yaw": lambda **kwargs: rotation(0, **kwargs),
        "rotation_pitch": lambda **kwargs: rotation(1, **kwargs),
        "rotation_roll": lambda **kwargs: rotation(2, **kwargs),
        "position_x": lambda **kwargs: position(0, **kwargs),
        "position_y": lambda **kwargs: position(1, **kwargs),
        "position_z": lambda **kwargs: position(2, **kwargs),
        "scale_uniform": scale_uniform,
        "scale_stepped": scale_stepped_timed,
    }


def create_control_system(config=None, clock=time.monotonic):
    from runtime_config import default_config, validate_config
    config = validate_config(default_config() if config is None else config)
    controls = config["controls"]
    system = ControlSystem(clock=clock)
    extractors = create_extractors()
    appliers = create_appliers(clock=clock)
    axes = {"rotation_yaw": 0, "rotation_pitch": 1, "rotation_roll": 2}
    for spec in controls["mappings"]:
        circular = spec["input"].endswith(("_rotation", "_pitch", "_yaw", "_roll"))
        # The original zoom responded faster than the head position channels.
        smoothing = controls["smoothing_ms"] / 3 if spec["mode"] == "stepped" else controls["smoothing_ms"]
        # Head roll measures an unoriented eye line (180°), while all angle
        # inputs keep the same degree-to-value scale and a 360° neutral pose.
        smoother = ExponentialSmoother(smoothing_ms=smoothing, decay_rate=0.2,
                                       center_value=spec["center"], clock=clock,
                                       period=0.5 if spec["input"] == "head_roll" else 1.0 if circular else None,
                                       decay_period=1.0 if circular else None)
        if spec["mode"] == "hybrid":
            controller = HybridRotationController(max_degrees=spec["scale"],
                left_threshold=spec["left_threshold"], right_threshold=spec["right_threshold"],
                continuous_speed_degrees_per_second=spec["continuous_speed"],
                center=spec["center"], invert=spec["invert"],
                reset_timeout_seconds=controls["reset_timeout_seconds"],
                reset_duration_seconds=controls["reset_duration_seconds"], clock=clock)
            mapping = HybridRotationMapping(spec["id"], controller, smoother, spec["enabled"],
                input_extractor=extractors[spec["input"]], output_axis=axes[spec["output"]], clock=clock)
        else:
            if spec["mode"] == "stepped":
                stepped = appliers["scale_stepped"](threshold=spec["threshold"],
                    small_scale=spec["small_scale"], large_scale=spec["large_scale"],
                    transition_time_ms=spec["transition_ms"], hysteresis=spec["hysteresis"])
                def apply(value, output, spec=spec, stepped=stepped):
                    # Scale modifies sensitivity around the selected neutral input.
                    if value is not None:
                        value = spec["center"] + (value - spec["center"]) * spec["scale"] * (-1 if spec["invert"] else 1)
                    stepped(value, output)
            elif spec["output"] in axes:
                apply = appliers[spec["output"]](max_degrees=spec["scale"], center=spec["center"],
                                                   invert=spec["invert"], circular=circular)
            elif spec["output"].startswith("position_"):
                apply = appliers[spec["output"]](scale=spec["scale"], center=spec["center"], invert=spec["invert"])
            else:
                def apply(value, output, spec=spec):
                    value = spec["center"] if value is None else value
                    output["scale"] = _clamp(1.0 + (value - spec["center"]) * spec["scale"] * (-1 if spec["invert"] else 1), 0.1, 5.0)
            mapping = ControlMapping(spec["id"], extractors[spec["input"]], apply, smoother, spec["enabled"])
        system.add_mapping(mapping)
    return system


def create_default_control_system():
    return create_control_system()
