"""Ownership continuity without a second detector or image-space remapping."""
from types import SimpleNamespace

import pytest

from primary_person import PrimaryPersonLock


def point(x=0, y=0, visibility=1):
    return SimpleNamespace(x=x, y=y, z=0, visibility=visibility, presence=visibility)


def pose(dx=0, dy=0, scale=1, head_only=False):
    points = [point(visibility=0) for _ in range(33)]
    values = {0: (.5, .24), 2: (.48, .22), 5: (.52, .22), 7: (.45, .24), 8: (.55, .24)}
    if not head_only:
        values.update({11: (.42, .4), 12: (.58, .4), 23: (.44, .6), 24: (.56, .6),
                       13: (.34, .43), 14: (.66, .43), 15: (.25, .5), 16: (.75, .5)})
    for index, (x, y) in values.items():
        points[index] = point(.5 + (x-.5)*scale + dx, .4 + (y-.4)*scale + dy)
    return points


def hand(x, y=.5, size=1):
    points = [point(x, y) for _ in range(21)]
    for i, (dx, dy) in {5: (-.02, -.035), 9: (0, -.05), 13: (.01, -.04), 17: (.025, -.025)}.items():
        points[i] = point(x+dx*size, y+dy*size)
    return points


def candidate(x, label='Left', score=.9, **kwargs):
    return hand(x, **kwargs), label, score


def test_first_pose_keeps_ownership_when_another_person_is_reported():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(), 0, 4/3)
    generation, original = lock.generation, lock.region(0)
    assert not lock.update_pose(pose(dx=.35), .1, 4/3)
    assert lock.state == 'lost'
    assert lock.region(.1) == original
    assert lock.generation == generation
    assert lock.rejected == 1
    assert lock.update_pose(pose(dx=.025), .2, 4/3)
    assert lock.state == 'tracking' and lock.generation == generation


def test_pose_loss_preserves_existing_hands_but_does_not_admit_new_hands():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(), 0, 4/3)
    region = lock.region(0)
    assert lock.accepts_hand(hand(.25), 'Left', .01, 4/3)
    assert not lock.update_pose([], .1, 4/3)
    assert lock.accepts_hand(hand(.25), 'Left', .2, 4/3)
    assert not lock.accepts_hand(hand(.75), 'Right', .3, 4/3)
    assert lock.region(1.4) == region
    assert lock.update_pose(pose(), 1.4, 4/3)
    assert lock.accepts_hand(hand(.25), 'Left', 1.4, 4/3)


def test_release_and_new_acquisition_each_change_generation():
    lock = PrimaryPersonLock()
    assert lock.region(0) is None
    assert lock.update_pose(pose(), 0, 1)
    assert lock.generation == 1
    assert lock.region(1.51) is None
    assert lock.generation == 2 and lock.state == 'searching'
    assert lock.region(2) is None and lock.generation == 2
    assert lock.update_pose(pose(dx=.3), 2, 1)
    assert lock.generation == 3


def test_scale_jump_does_not_change_the_region():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(head_only=True), 0, 1)
    region = lock.region(0)
    assert not lock.update_pose(pose(head_only=True, scale=3), .1, 1)
    assert lock.region(.1) == region


def test_head_only_and_partial_occlusion_use_common_anchors():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(), 0, 4/3)
    assert lock.update_pose(pose(head_only=True, dx=.01), .1, 4/3)
    assert lock.update_pose(pose(head_only=True, dx=.02), .2, 4/3)


@pytest.mark.parametrize('aspect', [4/3, 16/9, 9/16])
def test_motion_and_mirroring_keep_canvas_coordinates(aspect):
    lock = PrimaryPersonLock()
    points = pose()
    for p in points:
        p.x = 1-p.x
    assert lock.update_pose(points, 0, aspect)
    left, top, right, bottom = lock.region(0)
    for p in points:
        if p.visibility:
            assert left <= p.x <= right
            assert top <= p.y <= bottom
    # Ownership uses physical wrist proximity; handedness can be reversed.
    assert lock.select_hands([candidate(.75, 'Right'), candidate(.25, 'Left')], .01, aspect) == [0, 1]


def test_extended_arms_remain_inside_the_roi_and_associate_to_their_wrists():
    lock = PrimaryPersonLock()
    points = pose()
    points[15], points[16] = point(.03, .3), point(.97, .3)
    assert lock.update_pose(points, 0, 4/3)
    assert lock.region(0)[0] == 0 and lock.region(0)[2] == 1
    assert lock.select_hands([candidate(.03, y=.3), candidate(.97, 'Right', y=.3)], .01, 4/3) == [0, 1]


def test_pose_hand_assignment_is_one_to_one_and_ignores_foreign_hands():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(), 0, 4/3)
    # Two candidates around the same wrist cannot count as a pair.
    accepted = lock.select_hands([candidate(.25), candidate(.26, 'Right')], .1, 4/3)
    assert accepted == [0]
    # A central foreign hand is not near either of this participant's wrists.
    assert lock.select_hands([candidate(.5)], .2, 4/3) == []


def test_missing_wrists_do_not_attach_unowned_hands_by_label():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(head_only=True), 0, 4/3)
    assert lock.select_hands([candidate(.25), candidate(.75, 'Right')], .1, 4/3) == []


def test_visibility_threshold_and_invalid_points_cannot_seed_a_lock():
    lock = PrimaryPersonLock(visibility_threshold=.8)
    points = pose()
    for p in points:
        p.visibility = .7
    assert not lock.update_pose(points, 0, 1)
    for p in points:
        p.x = float('nan')
    assert not lock.update_pose(points, .1, 1)
    assert lock.region(.1) is None and lock.generation == 0


def test_hands_only_tracks_initial_nearby_pair_despite_result_order_changes():
    lock = PrimaryPersonLock(use_pose=False)
    assert lock.select_hands([candidate(.35), candidate(.55, 'Right')], 0, 4/3) == [0, 1]
    generation = lock.generation
    assert lock.select_hands([candidate(.56, 'Right'), candidate(.36)], .1, 4/3) == [0, 1]
    assert lock.generation == generation == 1


def test_hands_only_does_not_combine_distant_participants():
    lock = PrimaryPersonLock(use_pose=False)
    assert lock.select_hands([candidate(.2), candidate(.9, 'Right')], 0, 4/3) == [0]
    region = lock.region(0)
    assert lock.select_hands([candidate(.9, 'Right', score=.999)], .1, 4/3) == []
    assert lock.region(.1) == region
    assert lock.state == 'lost'


def test_hands_only_does_not_replace_a_hand_with_a_more_confident_distant_one():
    lock = PrimaryPersonLock(use_pose=False)
    assert lock.select_hands([candidate(.3)], 0, 4/3) == [0]
    assert lock.select_hands([candidate(.75, score=.999), candidate(.31, score=.55)], .1, 4/3) == [1]
    assert lock.state == 'tracking'


def test_hands_only_keeps_missing_slot_while_other_hand_is_visible():
    lock = PrimaryPersonLock(use_pose=False)
    assert lock.select_hands([candidate(.35), candidate(.55, 'Right')], 0, 4/3) == [0, 1]
    for timestamp in (.5, 1, 1.5, 2):
        assert lock.select_hands([candidate(.35), candidate(.9, 'Right')], timestamp, 4/3) == [0]
    assert lock.generation == 1
    assert lock.select_hands([candidate(.35), candidate(.56, 'Right')], 2.1, 4/3) == [0, 1]


def test_hands_only_reacquires_after_bounded_absence():
    lock = PrimaryPersonLock(use_pose=False)
    assert lock.select_hands([candidate(.3)], 0, 4/3) == [0]
    assert lock.select_hands([], .1, 4/3) == []
    assert lock.select_hands([candidate(.8)], 1, 4/3) == []
    assert lock.select_hands([candidate(.8)], 1.6, 4/3) == [0]
    assert lock.generation == 3


def test_hands_only_invalid_observations_cannot_acquire_a_slot():
    lock = PrimaryPersonLock(use_pose=False)
    bad = hand(.3)
    bad[0].x = float('nan')
    assert lock.select_hands([(bad, 'Left', 1), (hand(.4), 'Unknown', 1)], 0, 4/3) == []
    assert lock.generation == 0


def test_fast_hand_motion_keeps_owner_but_cross_frame_jump_is_rejected():
    lock = PrimaryPersonLock(use_pose=False)
    assert lock.select_hands([candidate(.2)], 0, 4/3) == [0]
    # 50 px / 100 ms on a 640 px camera: 500 px/s.
    for step in range(1, 5):
        assert lock.select_hands([candidate(.2 + step * 50/640)], step*.1, 4/3) == [0]
    assert lock.select_hands([candidate(.95)], .5, 4/3) == []
    assert lock.generation == 1


def test_abandoned_hand_slot_expires_without_releasing_the_other_hand():
    lock = PrimaryPersonLock(use_pose=False)
    assert lock.select_hands([candidate(.35), candidate(.55, 'Right')], 0, 4/3) == [0, 1]
    # The right hand travels away while the left one is absent.
    for step in range(1, 9):
        assert lock.select_hands([candidate(.55 + step*.04, 'Right')], step*.2, 4/3) == [0]
    region = lock.region(1.6)
    assert region[0] > .5
    # Another left hand at the abandoned location is not the old participant.
    assert lock.select_hands([candidate(.35), candidate(.87, 'Right')], 1.7, 4/3) == [1]
    assert lock.generation == 1


def test_equal_hand_matches_stop_control_instead_of_choosing_list_order():
    lock = PrimaryPersonLock(use_pose=False)
    assert lock.select_hands([candidate(.5)], 0, 4/3) == [0]
    original = lock.region(0)
    assert lock.select_hands([candidate(.48), candidate(.52)], .1, 4/3) == []
    assert lock.select_hands([candidate(.52), candidate(.48)], .2, 4/3) == []
    assert lock.region(.2) == original
    assert lock.state == 'lost'


def test_partial_pose_observation_does_not_shrink_the_established_context():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(), 0, 4/3)
    original = lock.region(0)
    assert lock.update_pose(pose(head_only=True), .1, 4/3)
    current = lock.region(.1)
    assert current[0] <= original[0] and current[1] <= original[1]
    assert current[2] >= original[2] and current[3] >= original[3]


def test_small_landmark_jitter_does_not_accumulate_a_full_frame_region():
    lock = PrimaryPersonLock()
    points = pose(head_only=True)
    for index in (7, 8):
        points[index].visibility = 0
    assert lock.update_pose(points, 0, 4/3)
    for step in range(100):
        sample = pose(head_only=True)
        for index in (7, 8):
            sample[index].visibility = 0
        offsets = ((0, 0, 0), (.004, .004, 0), (.004, 0, .004), (0, 0, .004))[step % 4]
        for index, offset in zip((0, 2, 5), offsets):
            sample[index].x += offset
        assert lock.update_pose(sample, (step+1)*.1, 4/3)
    left, _top, right, _bottom = lock.region(10)
    assert right-left < .37


def test_owned_hands_keep_session_alive_through_pose_loss_and_rejected_intruder():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(), 0, 4/3)
    assert lock.select_hands([candidate(.25), candidate(.75, 'Right')], 0, 4/3) == [0, 1]
    generation = lock.generation
    for timestamp in (.5, 1, 1.5, 2, 2.5):
        assert not lock.update_pose(pose(dx=.35) if timestamp == 2 else [], timestamp, 4/3)
        assert lock.select_hands([candidate(.25), candidate(.75, 'Right')], timestamp, 4/3) == [0, 1]
    assert lock.generation == generation
    assert lock.region(4.01) is None
    assert lock.generation == generation + 1


def test_foreign_hand_cannot_keep_a_lost_pose_session_alive():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(), 0, 4/3)
    assert lock.select_hands([candidate(.25)], 0, 4/3) == [0]
    assert not lock.update_pose([], .1, 4/3)
    for timestamp in (.5, 1, 1.4):
        assert lock.select_hands([candidate(.8)], timestamp, 4/3) == []
    assert lock.region(1.6) is None


def test_palm_size_tolerates_small_wrist_error_when_seeding_a_hand():
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(), 0, 4/3)
    assert lock.select_hands([candidate(.33, size=2)], .1, 4/3) == [0]
    lock = PrimaryPersonLock()
    assert lock.update_pose(pose(head_only=True), 0, 4/3)
    # No wrist observation still means no new hand, regardless of its size.
    assert lock.select_hands([candidate(.5, size=2)], .1, 4/3) == []
