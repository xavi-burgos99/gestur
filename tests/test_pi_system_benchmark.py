import threading

import pytest

from scripts.benchmark_pi_system import (
    ReplayCapture, TrialGuard, arguments, model_inventory, parse_throttled, read_json, trial_config,
)


def runtime(count=1, error=None):
    return {'error': error, 'tracking': {'state': 'running', 'metrics': {
        'captured_frames': count, 'pose_frames': count, 'hand_frames': count}}}


def test_thermal_guard_and_historical_throttling_are_distinct():
    throttle = parse_throttled('throttled=0x50005\n')
    assert throttle['now']['undervoltage'] and throttle['now']['throttled']
    assert throttle['since_boot']['undervoltage'] and throttle['since_boot']['throttled']
    assert not throttle['now']['frequency_capped']
    assert parse_throttled('unavailable') is None
    guard = TrialGuard()
    assert guard.check(1, {'cpu_temperature_c': 85, 'throttling': throttle}, runtime(), 0) is None
    assert guard.check(2, {'cpu_temperature_c': 85.1}, runtime(2), 0) == 'temperature_exceeded_85C'


def test_persistent_errors_have_grace_and_reset_when_recovered():
    guard = TrialGuard()
    assert guard.check(1, {}, runtime(error='camera'), 0) is None
    assert guard.check(15, {}, runtime(error='camera'), 0) is None
    assert guard.check(16, {}, runtime(2), 0) is None
    assert guard.check(20, {}, runtime(error='camera'), 0) is None
    assert guard.check(34, {}, runtime(error='camera'), 0) is None
    assert guard.check(35, {}, runtime(error='camera'), 0) == 'persistent_error: camera'


def test_startup_failure_and_runtime_stall_stop_after_bounded_grace():
    guard = TrialGuard()
    assert guard.check(0, {}, None, None) is None
    assert guard.check(29, {}, None, None) is None
    assert guard.check(30, {}, None, None) is None
    assert guard.check(45, {}, None, None) == 'persistent_error: runtime_status_missing_or_stale'
    guard = TrialGuard()
    assert guard.check(29, {}, runtime(), 0) is None
    assert guard.check(44, {}, runtime(), 0) is None
    assert guard.check(59, {}, runtime(), 0) == 'persistent_error: captured_frames_not_progressing'


def test_progressing_inference_can_run_without_local_hardware_sensors():
    guard = TrialGuard()
    for elapsed in range(61):
        assert guard.check(elapsed, {'cpu_temperature_c': None}, runtime(elapsed), 0) is None


def test_replay_is_paced_copies_frames_and_can_be_interrupted():
    now, waits = [1.0], []

    class Event:
        stopped = False

        def wait(self, delay):
            waits.append(delay)
            now[0] += delay
            return self.stopped

        def is_set(self):
            return self.stopped

    event = Event()
    frames = [[1], [2]]
    capture = ReplayCapture(frames, event, clock=lambda: now[0])
    ok, first = capture.read()
    first[0] = 999
    assert ok and frames == [[1], [2]]
    assert capture.read() == (True, [2])
    assert waits[-1] == pytest.approx(1 / 24)
    now[0] += 10  # Slow inference never triggers a replay catch-up burst.
    assert capture.read() == (True, [1])
    assert waits[-1] == 0
    event.stopped = True
    assert capture.read() == (False, None)
    assert not capture.isOpened()
    capture.release()
    assert capture.read() == (False, None)
    with pytest.raises(ValueError):
        ReplayCapture([], threading.Event())


@pytest.mark.parametrize('duration', ['nan', 'inf', '0', '1801', '-1'])
def test_invalid_duration_is_rejected(tmp_path, duration):
    with pytest.raises(SystemExit):
        arguments(['--duration', duration, '--output-dir', str(tmp_path / 'new')])


def test_arguments_require_local_replay_and_a_new_output_directory(tmp_path):
    image = tmp_path / 'image.png'
    image.write_bytes(b'not-decoded-by-parser')
    argv = ['--source', 'replay', '--image', str(image), '--output-dir', str(tmp_path / 'new')]
    parsed = arguments(argv)
    assert parsed.duration == 60
    assert parsed.output_dir == tmp_path / 'new'
    for invalid in (['--output-dir', str(tmp_path)],
                    ['--source', 'replay', '--output-dir', str(tmp_path / 'new')],
                    ['--image', str(image), '--output-dir', str(tmp_path / 'new')],
                    ['--output-dir', '/var/lib/gestur/benchmark-trial']):
        with pytest.raises(SystemExit):
            arguments(invalid)


def test_trial_config_is_independent_and_requests_both_detectors():
    from runtime_config import default_config
    from tracking_session import tracking_request
    before = default_config()
    config = trial_config(False, 2)
    request = tracking_request(config, has_model=True, no_camera=False)
    assert request['use_pose'] and request['use_hands']
    assert config['render']['fullscreen']
    assert config['render']['target_fps'] == 60
    assert config['render']['antialias_samples'] == 2
    assert config['tracking']['camera_index'] == 2
    assert default_config() == before


def test_status_reader_tolerates_partial_or_invalid_records(tmp_path):
    status = tmp_path / 'status.json'
    assert read_json(status) is None
    for value in ('{', '[]', 'null', '"hello"'):
        status.write_text(value)
        assert read_json(status) is None
    status.write_text('{"updated_at": 123}')
    assert read_json(status) == {'updated_at': 123}


def test_local_model_and_motion_arguments_are_validated(tmp_path):
    model = tmp_path / 'capitel.obj'
    model.write_text('local geometry')
    common = ['--output-dir', str(tmp_path / 'output')]
    parsed = arguments(common + ['--model', str(model), '--motion', 'controls'])
    assert parsed.model == model
    assert parsed.motion == 'controls'
    assert arguments(common).model is None
    assert arguments(common).motion == 'continuous'
    for invalid in (['--model', str(tmp_path / 'missing.obj')],
                    ['--model', str(tmp_path)], ['--motion', 'unknown']):
        with pytest.raises(SystemExit):
            arguments(common + invalid)


def test_model_inventory_counts_loaded_strips_and_records_actual_textures(tmp_path):
    import hashlib
    from panda3d.core import (
        Geom, GeomNode, GeomTriangles, GeomTristrips, GeomLines,
        GeomVertexData, GeomVertexFormat, NodePath, Texture,
    )
    source = tmp_path / 'source.bam'
    source.write_bytes(b'exact source bytes')
    model = NodePath('test-model')
    data = GeomVertexData('vertices', GeomVertexFormat.get_v3(), Geom.UH_static)
    data.set_num_rows(4)
    for kind, vertices in ((GeomTriangles, (0, 1, 2)),
                           (GeomTristrips, (0, 1, 2, 3)), (GeomLines, (0, 1))):
        primitive = kind(Geom.UH_static)
        for vertex in vertices:
            primitive.add_vertex(vertex)
        primitive.close_primitive()
        geom = Geom(data)
        geom.add_primitive(primitive)
        node = GeomNode('part')
        node.add_geom(geom)
        model.attach_new_node(node)
    texture = Texture('real texture')
    texture.setup_2d_texture(32, 64, Texture.T_unsigned_byte, Texture.F_rgb)
    model.set_texture(texture)
    inventory = model_inventory(model, source, synthetic=False)
    assert inventory['source'] == 'local_model'
    assert inventory['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert inventory['triangles'] == 3  # One triangle + two in a strip; lines excluded.
    assert inventory['geom_batches'] == 3
    assert inventory['textures'][0]['name'] == 'real texture'
    assert inventory['textures'][0]['loaded_size'] == [32, 64]
    model.remove_node()
