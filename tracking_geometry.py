"""Dependency-free landmark geometry and time-based filtering.

Angles use camera coordinates (x right, y down, z away from the camera).
Image positions and proximity are normalized. Hand orientation/gestures use
metric 3D landmarks, so a wide camera image does not distort the palm plane.
"""
import math


ANGLE_FIELDS = frozenset(('pitch', 'yaw', 'roll', 'rotation'))


def empty_part(hand=False, head=False):
    # Keep the legacy head argument while giving every tracked part the same
    # optional proximity field. Missing proximity does not invalidate a part.
    result = dict(detected=False, x=None, y=None, pitch=None, yaw=None, roll=None, scale=None)
    if hand:
        result.update(rotation=None, pinch=None, openness=None, gesture='unknown')
    return result


def empty_data():
    return {'head': empty_part(head=True), 'torso': empty_part(),
            'left_hand': empty_part(hand=True), 'right_hand': empty_part(hand=True)}


def xyz(point):
    return point.x, point.y, point.z


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def midpoint(a, b):
    return tuple((x + y) / 2 for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def length(a):
    return math.sqrt(dot(a, a))


def unit(a):
    norm = length(a)
    return tuple(x / norm for x in a) if norm > 1e-8 else None


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def wrap_angle(angle, period=360.0):
    return (angle + period / 2) % period - period / 2


def smooth_value(previous, value, dt, smoothing_time, circular=False, period=360.0):
    if previous is None or smoothing_time <= 0:
        return value
    alpha = -math.expm1(-max(0.0, dt) / smoothing_time)
    delta = wrap_angle(value - previous, period) if circular else value - previous
    result = previous + alpha * delta
    return wrap_angle(result, period) if circular else result


def landmarks_valid(points, indices, visibility_threshold=None):
    if not points or len(points) <= max(indices):
        return False
    for index in indices:
        p = points[index]
        if not all(math.isfinite(v) for v in xyz(p)):
            return False
        if visibility_threshold is not None:
            for field in ('visibility', 'presence'):
                value = getattr(p, field, None)
                if value is not None and (not math.isfinite(value) or value < visibility_threshold):
                    return False
    return True


def anatomical_hand(label, mirror, invert=False):
    """Tasks labels unmirrored input (opposite to the legacy Solutions graph)."""
    if label not in ('Left', 'Right'):
        return None
    if bool(mirror) != bool(invert):
        label = 'Right' if label == 'Left' else 'Left'
    return label


def apparent_proximity(normalized, world, indices, span_indices, aspect, minimum, maximum):
    """Orientation-corrected apparent span in image-width units, mapped to 0..1.

    A weak-perspective fit compares image pair lengths with the XY projection
    of relative metric landmarks. Multiplying by the 3D span removes first-order
    foreshortening without using absolute world.z (which is object-centered).
    Multiple rigid palm/torso pairs reduce sensitivity to one projected edge.
    This is a monocular size proxy, not calibrated camera distance in metres.
    """
    if not math.isfinite(aspect) or aspect <= 0:
        return None
    projected_energy = image_energy = spatial_energy = 0.0
    for offset, first in enumerate(indices):
        for second in indices[offset + 1:]:
            delta = sub(xyz(world[second]), xyz(world[first]))
            dx = normalized[second].x - normalized[first].x
            dy = (normalized[second].y - normalized[first].y) / aspect
            image_energy += dx * dx + dy * dy
            projected_energy += delta[0] * delta[0] + delta[1] * delta[1]
            spatial_energy += dot(delta, delta)
    span = length(sub(xyz(world[span_indices[1]]), xyz(world[span_indices[0]])))
    if (span < 1e-5 or spatial_energy < 1e-10 or image_energy < 1e-12
            or projected_energy < .01 * spatial_energy):
        return None
    apparent_span = span * math.sqrt(image_energy / projected_energy)
    if not math.isfinite(apparent_span):
        return None
    return clamp((apparent_span - minimum) / (maximum - minimum))


def pose_features(normalized, world, visibility_threshold=.5, aspect=4/3,
                  scale_min=.02, scale_max=.20):
    result = {'head': empty_part(head=True), 'torso': empty_part()}
    # 2 and 5 are the centres of DIFFERENT eyes (1 and 2 belong to one eye).
    head_indices = (0, 2, 5, 7, 8)
    if (landmarks_valid(normalized, head_indices, visibility_threshold)
            and landmarks_valid(world, head_indices, visibility_threshold)):
        nose, left_ear, right_ear = xyz(world[0]), xyz(world[7]), xyz(world[8])
        direction = sub(nose, midpoint(left_ear, right_ear))
        if length(direction) > 1e-8:
            nose2d = normalized[0]
            # Use the eye line as an unoriented line: upright is zero, not 180°.
            dx = (normalized[5].x - normalized[2].x) * aspect
            dy = normalized[5].y - normalized[2].y
            roll = math.degrees(math.atan2(dy, dx))
            roll = (roll + 90.0) % 180.0 - 90.0
            # Horizontal-normalized units retain the default zoom range [0, 1].
            eye_dist = math.hypot(normalized[5].x-normalized[2].x,
                                  (normalized[5].y-normalized[2].y) / aspect)
            result['head'].update(detected=True, x=nose2d.x, y=nose2d.y,
                pitch=math.degrees(math.atan2(direction[1], math.hypot(direction[0], direction[2]))),
                yaw=math.degrees(math.atan2(direction[0], -direction[2])), roll=roll,
                scale=clamp((eye_dist-scale_min) / (scale_max-scale_min)))
    torso_indices = (11, 12, 23, 24)
    if (landmarks_valid(normalized, torso_indices, visibility_threshold)
            and landmarks_valid(world, torso_indices, visibility_threshold)):
        shoulder = midpoint(xyz(world[11]), xyz(world[12]))
        hip = midpoint(xyz(world[23]), xyz(world[24]))
        up = sub(shoulder, hip)
        across = sub(xyz(world[12]), xyz(world[11]))
        normal = cross(up, across)
        if length(normal) > 1e-8:
            # Front facing is -z independent of selfie/image left-right order.
            if normal[2] > 0:
                normal = tuple(-v for v in normal)
            roll = (math.degrees(math.atan2(across[1], across[0])) + 90) % 180 - 90
            result['torso'].update(detected=True,
                x=(normalized[23].x+normalized[24].x)/2,
                y=(normalized[23].y+normalized[24].y)/2,
                pitch=math.degrees(math.atan2(normal[1], math.hypot(normal[0],normal[2]))),
                yaw=math.degrees(math.atan2(normal[0], -normal[2])), roll=roll,
                scale=apparent_proximity(normalized, world, torso_indices, (11, 12),
                                         aspect, .12, .80))
    return result


def hand_features(normalized, world, label, aspect=4/3):
    result = empty_part(hand=True)
    if (label not in ('Left', 'Right') or not math.isfinite(aspect) or aspect <= 0
            or not landmarks_valid(normalized, range(21))
            or not landmarks_valid(world, range(21))):
        return result
    points = [xyz(p) for p in world]
    up = unit(sub(points[9], points[0]))
    across = sub(points[17], points[5])
    # Use model handedness here (before anatomical/mirror correction), since
    # this geometry describes the image passed to the network.
    if label == 'Left':
        across = tuple(-v for v in across)
    palm_width = length(sub(points[5], points[17]))
    if up is None or palm_width < 1e-5:
        return result
    # Gram-Schmidt separates finger direction from the knuckle line. Simply
    # taking two atan2 angles of the palm normal couples pitch and yaw when the
    # wrist also rolls. A complete orthogonal frame retains that third axis.
    along_fingers = dot(across, up)
    orthogonal = tuple(across[i] - along_fingers * up[i] for i in range(3))
    if length(orthogonal) / palm_width < 1e-3:
        return result  # Nearly collinear landmarks cannot define a palm plane.
    right = unit(orthogonal)
    if right is None:
        return result
    down = tuple(-v for v in up)
    away = cross(right, down)
    # Columns [right, down, away] form a camera-space rotation matrix. Neutral
    # means fingers up and palm facing the camera. Euler order is
    # Rz(roll) @ Ry(-yaw) @ Rx(pitch), preserving the previous single-axis signs.
    horizontal = math.hypot(right[0], right[1])
    yaw = math.atan2(right[2], horizontal)
    if horizontal > 1e-6:
        pitch = math.atan2(down[2], away[2])
        roll = math.atan2(right[1], right[0])
    else:
        # At yaw +/-90 degrees, pitch and roll are not independently observable
        # in this Euler convention. Choose roll=0 while retaining the same frame.
        pitch = math.atan2(-away[1], down[1])
        roll = 0.0
    image_dx = (normalized[9].x - normalized[0].x) * aspect
    image_dy = normalized[9].y - normalized[0].y
    # Existing *_hand_rotation controls use the on-screen wrist-to-middle line.
    # Keep that contract separate from 3D roll. If fingers point into the camera,
    # only this projected angle is undefined; pinch and 3D orientation remain valid.
    rotation = (wrap_angle(math.degrees(math.atan2(image_dx, -image_dy)))
                if math.hypot(image_dx, image_dy) >= 1e-6 else None)
    pinch = clamp(length(sub(points[4], points[8])) / palm_width)
    extended = 0
    for mcp, pip, dip, tip in ((5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)):
        first, second = unit(sub(points[mcp], points[pip])), unit(sub(points[tip], points[pip]))
        straight = first and second and dot(first, second) < -.75
        if straight and length(sub(points[tip], points[0])) > 1.1 * length(sub(points[pip], points[0])):
            extended += 1
    openness = extended / 4.0
    gesture = 'pinch' if pinch < .25 else 'open' if extended >= 3 else 'fist' if extended == 0 else 'unknown'
    result.update(detected=True, x=normalized[0].x, y=normalized[0].y,
        pitch=wrap_angle(math.degrees(pitch)), yaw=wrap_angle(math.degrees(yaw)),
        roll=wrap_angle(math.degrees(roll)), rotation=rotation,
        pinch=pinch, openness=openness, gesture=gesture,
        scale=apparent_proximity(normalized, world, (0, 5, 9, 13, 17), (5, 17),
                                 aspect, .025, .25))
    return result


class TrackingFilter:
    """Filters only fresh detections; loss never manufactures zero-angle samples."""
    def __init__(self, smoothing_time=.06, timeout=.25):
        self.smoothing_time = max(0, smoothing_time)
        self.timeout = max(0, timeout)
        self.data = empty_data()
        self.last_seen = {key: None for key in self.data}

    def update(self, key, sample, timestamp):
        if not sample['detected']:
            # A missing observation stops control immediately. Last seen is kept
            # only to decide whether the next observation may reuse the filter.
            self.data[key] = empty_part(hand='hand' in key, head=key == 'head')
            return
        previous = self.data[key]
        last_seen = self.last_seen[key]
        dt = timestamp-last_seen if last_seen is not None else 0
        if last_seen is None or dt > self.timeout:
            previous = {}
        filtered = {}
        for name, value in sample.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Eye and shoulder lines are unoriented: +89° and -89° differ by 2°.
                period = 180.0 if key in ('head', 'torso') and name == 'roll' else 360.0
                filtered[name] = smooth_value(previous.get(name), value, dt,
                    self.smoothing_time, name in ANGLE_FIELDS, period)
            else:
                filtered[name] = value
        self.data[key] = filtered
        self.last_seen[key] = timestamp

    def expire(self, timestamp):
        for key, seen in self.last_seen.items():
            if seen is None or timestamp-seen > self.timeout:
                self.data[key] = empty_part(hand='hand' in key, head=key == 'head')

    def snapshot(self):
        return {key: value.copy() for key, value in self.data.items()}
