"""Spatial ownership for one participant, without additional inference.

This is continuity, not biometric identification: a completely occluded person
cannot be distinguished from another person in the same place and at the same
scale. Ambiguous observations stop control instead of moving the stored region.
All coordinates refer to the original (possibly mirrored) image canvas.
"""

import math
from dataclasses import dataclass
from itertools import combinations, permutations
from statistics import median


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point(point, aspect, threshold=None):
    try:
        x, y = float(point.x), float(point.y) / aspect
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        if threshold is not None:
            for field in ("visibility", "presence"):
                value = getattr(point, field, None)
                if value is not None and (
                    not math.isfinite(value) or value < threshold
                ):
                    return None
        return x, y
    except (TypeError, ValueError, AttributeError):
        return None


@dataclass
class _Observation:
    points: dict
    center: tuple
    span: float
    region: tuple
    timestamp: float
    label: str = ""


class PrimaryPersonLock:
    """Keep the first accepted participant until a bounded absence.

    ``region`` is a normalized (left, top, right, bottom) rectangle for masking
    the unchanged image canvas. ``select_hands`` takes (points, anatomical label,
    score) tuples and returns indices into that list, independent of its order
    after acquisition. Both networks retain their existing inference schedules.
    """

    def __init__(self, use_pose=True, grace_seconds=1.5, visibility_threshold=0.5):
        if not math.isfinite(grace_seconds) or grace_seconds <= 0:
            raise ValueError("The ownership grace period must be positive.")
        if not 0 <= visibility_threshold <= 1:
            raise ValueError("The visibility threshold must be between zero and one.")
        self.use_pose = bool(use_pose)
        self.grace_seconds = grace_seconds
        self.visibility_threshold = visibility_threshold
        self.generation = 0
        self.rejected = 0
        self.state = "searching"
        self._pose = None
        self._wrists = {}
        self._hands = {}
        self._last_seen = None
        self._pose_valid = False

    def _expire(self, timestamp):
        if (
            self._last_seen is not None
            and timestamp - self._last_seen > self.grace_seconds
        ):
            self._pose, self._wrists, self._hands = None, {}, {}
            self._last_seen = None
            self._pose_valid = False
            self.state = "searching"
            self.generation += 1
        else:
            # A visible hand must not keep the other hand's abandoned area
            # indefinitely. A returning slot has to join the current anchor.
            self._hands = {
                slot: hand
                for slot, hand in self._hands.items()
                if timestamp - hand.timestamp <= self.grace_seconds
            }

    def region(self, timestamp):
        self._expire(timestamp)
        observations = ([self._pose] if self.use_pose else []) + list(
            self._hands.values()
        )
        observations = [item for item in observations if item is not None]
        if not observations:
            return None
        return (
            min(item.region[0] for item in observations),
            min(item.region[1] for item in observations),
            max(item.region[2] for item in observations),
            max(item.region[3] for item in observations),
        )

    @staticmethod
    def _bounds(points, padding, aspect):
        return (
            max(0.0, min(p[0] for p in points) - padding),
            max(0.0, (min(p[1] for p in points) - padding) * aspect),
            min(1.0, max(p[0] for p in points) + padding),
            min(1.0, (max(p[1] for p in points) + padding) * aspect),
        )

    def _pose_observation(self, landmarks, timestamp, aspect):
        if len(landmarks) < 25 or not math.isfinite(aspect) or aspect <= 0:
            return None, {}
        visible = {
            i: p
            for i, landmark in enumerate(landmarks)
            if (p := _point(landmark, aspect, self.visibility_threshold)) is not None
            and -0.1 <= p[0] <= 1.1
            and -0.1 <= p[1] * aspect <= 1.1
        }
        core = {i: visible[i] for i in (0, 2, 5, 7, 8, 11, 12, 23, 24) if i in visible}
        if len(core) < 3:
            return None, {}
        center = tuple(median(p[axis] for p in core.values()) for axis in (0, 1))
        span = max(
            (_distance(a, b) for a, b in combinations(core.values(), 2)), default=0
        )
        if span < 0.008:
            return None, {}
        # Keep hands/legs that are actually visible, plus room for normal motion.
        # Never enlarge the stored rectangle using a rejected observation.
        region = self._bounds(list(visible.values()), max(0.16, span * 0.8), aspect)
        wrists = {}
        for wrist, elbow in ((15, 13), (16, 14)):
            if wrist in visible:
                forearm = (
                    _distance(visible[wrist], visible[elbow]) if elbow in visible else 0
                )
                wrists[wrist] = (visible[wrist], max(0.045, span * 0.23, forearm * 0.7))
        return _Observation(core, center, span, region, timestamp), wrists

    @staticmethod
    def _compatible(previous, candidate, hand=False):
        common = previous.points.keys() & candidate.points.keys()
        if len(common) < 3:
            return False
        # Compare the same visible anchors. A partial occlusion must not move the
        # centre merely because the set of visible landmarks changed.
        displacement = median(
            _distance(previous.points[i], candidate.points[i]) for i in common
        )
        old_span = max(
            _distance(previous.points[a], previous.points[b])
            for a, b in combinations(common, 2)
        )
        new_span = max(
            _distance(candidate.points[a], candidate.points[b])
            for a, b in combinations(common, 2)
        )
        if old_span < 0.005 or not 0.5 <= new_span / old_span <= 2.0:
            return False
        dt = max(0.0, min(0.25, candidate.timestamp - previous.timestamp))
        # Hands can travel much faster than the torso. At 10 Hz this allows a
        # 500 px/s gesture on a 640 px image, but not a jump across the frame.
        margin = (
            max(0.025, old_span * 0.45) + min(0.2, dt)
            if hand
            else max(0.025, old_span * 0.35) + 0.35 * dt
        )
        return displacement <= margin

    def update_pose(self, landmarks, timestamp, aspect):
        self._expire(timestamp)
        if not self.use_pose:
            return False
        candidate, wrists = self._pose_observation(landmarks, timestamp, aspect)
        accepted = candidate is not None and (
            self._pose is None or self._compatible(self._pose, candidate)
        )
        self._pose_valid = accepted
        if not accepted:
            if candidate is not None:
                self.rejected += 1
            self.state = "lost" if self._pose is not None else "searching"
            return False
        if self._pose is None:
            self.generation += 1
        else:
            # Removing an occluded arm/torso from the visible set must not crop
            # it out of every subsequent inference. Move the established view
            # with common anchors; retain its context until ownership expires.
            common = self._pose.points.keys() & candidate.points.keys()
            dx = median(
                candidate.points[i][0] - self._pose.points[i][0] for i in common
            )
            dy = (
                median(candidate.points[i][1] - self._pose.points[i][1] for i in common)
                * aspect
            )
            left, top, right, bottom = self._pose.region
            new_left, new_top, new_right, new_bottom = candidate.region
            width, height = right - left, bottom - top
            new_width, new_height = new_right - new_left, new_bottom - new_top
            center_x, center_y = (left + right) / 2 + dx, (top + bottom) / 2 + dy
            # Retain dimensions rather than unioning translated rectangles:
            # small landmark jitter must not accumulate an ever-growing ROI.
            if new_width > width + 0.02:
                center_x = (new_left + new_right) / 2
            if new_height > height + 0.02:
                center_y = (new_top + new_bottom) / 2
            width, height = max(width, new_width), max(height, new_height)
            # A fresh, complete observation also bounds drift of the retained
            # centre; partial observations retain their established context.
            if candidate.points.keys() == self._pose.points.keys():
                center_x = (new_left + new_right) / 2
                center_y = (new_top + new_bottom) / 2
            candidate.region = (
                max(0.0, center_x - width / 2),
                max(0.0, center_y - height / 2),
                min(1.0, center_x + width / 2),
                min(1.0, center_y + height / 2),
            )
        self._pose, self._wrists = candidate, wrists
        self._last_seen = timestamp
        self.state = "tracking"
        return True

    @classmethod
    def _hand_observation(cls, points, label, timestamp, aspect):
        if (
            label not in ("Left", "Right")
            or len(points) < 21
            or not math.isfinite(aspect)
            or aspect <= 0
        ):
            return None
        palm = {i: _point(points[i], aspect) for i in (0, 5, 9, 13, 17)}
        if any(p is None for p in palm.values()):
            return None
        if not all(
            -0.1 <= p[0] <= 1.1 and -0.1 <= p[1] * aspect <= 1.1 for p in palm.values()
        ):
            return None
        span = max(_distance(palm[0], palm[9]), _distance(palm[5], palm[17]))
        if span < 0.006:
            return None
        padding = max(0.07, span * 2.5)
        return _Observation(
            palm,
            palm[0],
            span,
            cls._bounds(list(palm.values()), padding, aspect),
            timestamp,
            label,
        )

    @staticmethod
    def _assignment(costs, slots):
        """At most two hands: prefer a complete, cheapest one-to-one match."""
        best_count, options = 0, []
        for count in range(1, min(2, len(costs), len(slots)) + 1):
            for indices in combinations(costs, count):
                for assigned in permutations(slots, count):
                    total = sum(
                        costs[i].get(slot, math.inf)
                        for i, slot in zip(indices, assigned)
                    )
                    if math.isfinite(total):
                        if count > best_count:
                            best_count, options = count, []
                        options.append((total, list(zip(indices, assigned))))
        if not options:
            return []
        best_cost, best = min(options, key=lambda option: option[0])
        # A near tie cannot establish which hand owns a slot. Retain only
        # assignments shared by all equally plausible solutions.
        certain = set(best)
        for cost, alternative in options:
            if cost <= best_cost + 0.03:
                certain.intersection_update(alternative)
        return [match for match in best if match in certain]

    def select_hands(self, candidates, timestamp, aspect):
        self._expire(timestamp)
        observed = {
            i: hand
            for i, (points, label, _score) in enumerate(candidates)
            if (hand := self._hand_observation(points, label, timestamp, aspect))
            is not None
        }
        if self.use_pose:
            if self._pose is None:
                return []
            costs = {}
            for i, hand in observed.items():
                costs[i] = {}
                for slot, previous in self._hands.items():
                    if hand.label == previous.label and self._compatible(
                        previous, hand, hand=True
                    ):
                        costs[i][slot] = _distance(hand.center, previous.center) / max(
                            previous.span, 0.01
                        )
                if self._pose_valid:
                    for slot, (wrist, radius) in self._wrists.items():
                        # Palm size covers small Pose wrist errors without
                        # accepting an arbitrarily distant or giant new hand.
                        radius = max(
                            radius, min(hand.span * 0.9, self._pose.span * 0.4)
                        )
                        distance = _distance(hand.center, wrist)
                        if distance <= radius:
                            costs[i][slot] = min(
                                costs[i].get(slot, math.inf), distance / radius
                            )
            # Match to either anatomical wrist, not image-left/right. Mirroring
            # and occasional handedness errors cannot select another owner.
            matches = self._assignment(costs, set(self._wrists) | set(self._hands))
            for i, slot in matches:
                self._hands[slot] = observed[i]
            if matches:
                self._last_seen = timestamp
                self.state = "tracking"
            return [i for i, _slot in matches]
        if not observed:
            self.state = "lost" if self._hands else "searching"
            return []
        if not self._hands:
            # First simultaneous detections have no history. Seed one hand;
            # the other must be spatially close before it can join the session.
            first = max(observed, key=lambda i: (candidates[i][2], -i))
            hand = observed[first]
            self._hands[hand.label] = hand
            self._last_seen = timestamp
            self.generation += 1
            matches = [(first, hand.label)]
        else:
            costs = {}
            for i, hand in observed.items():
                costs[i] = {
                    slot: _distance(hand.center, previous.center)
                    / max(previous.span, 0.01)
                    for slot, previous in self._hands.items()
                    if hand.label == slot
                    and self._compatible(previous, hand, hand=True)
                }
            matches = self._assignment(costs, self._hands)
        used = {i for i, _slot in matches}
        occupied = set(self._hands)
        if matches and len(occupied) == 1:
            anchor = observed[matches[0][0]]
            # An unassociated hand never expands the capture region. This is a
            # proximity heuristic only, not proof that two hands share a body.
            nearby = [
                i
                for i, hand in observed.items()
                if i not in used
                and hand.label not in occupied
                and 0.45 <= hand.span / anchor.span <= 2.2
                and _distance(hand.center, anchor.center)
                <= min(0.5, max(0.18, 7 * anchor.span))
            ]
            if nearby:
                second = min(
                    nearby, key=lambda i: _distance(observed[i].center, anchor.center)
                )
                matches.append((second, observed[second].label))
        for i, slot in matches:
            self._hands[slot] = observed[i]
        if matches:
            self._last_seen = timestamp
            self.state = "tracking"
        else:
            self.rejected += len(observed)
            self.state = "lost"
        return [i for i, _slot in matches]

    def accepts_hand(self, points, label, timestamp, aspect):
        """Single-observation convenience; use select_hands for a complete frame."""
        return bool(self.select_hands([(points, label, 1.0)], timestamp, aspect))
