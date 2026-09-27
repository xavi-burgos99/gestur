"""Exercise portal probes and guarded fixture cleanup without device access."""
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=['api', 'import'])
def probe(request, monkeypatch):
    api = request.param == 'api'
    module = load('smoke_' + request.param, 'scripts/research/smoke_portal_api.py'
                  if api else 'scripts/check-portal-import.py')
    now = [0.0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda delay: now.__setitem__(0, now[0] + delay)))
    return SimpleNamespace(module=module, now=now, api=api,
                           failure=module.SmokeFailure if api else module.SmokeError,
                           connection=lambda: module.SmokeFailure('connection_failed')
                           if api else module.ConnectionFailure('offline'))


def client_for(probe, action):
    return SimpleNamespace(**{'request' if probe.api else 'call': action})


def test_wait_ready_retries_only_read_only_startup_connection(probe):
    calls = []

    def action(method, endpoint, **kwargs):
        calls.append((method, endpoint, kwargs))
        if len(calls) == 1:
            raise probe.connection()
        return {'authenticated': False}

    probe.module.wait_ready(client_for(probe, action), timeout=1)
    assert len(calls) == 2
    assert all(method == 'GET' and endpoint.endswith('session') for method, endpoint, _ in calls)
    assert probe.now[0] == .25


def test_wait_ready_deadline_does_not_retry_forever(probe):
    calls = []

    def action(method, endpoint, **kwargs):
        calls.append(kwargs['timeout'])
        raise probe.connection()

    with pytest.raises(probe.failure):
        probe.module.wait_ready(client_for(probe, action), timeout=1)
    assert calls == [1, .75, .5, .25]
    assert probe.now[0] == 1


def test_wait_ready_does_not_hide_other_api_errors(probe):
    calls = []

    def action(*args, **kwargs):
        calls.append(args)
        raise probe.failure('http_403')

    with pytest.raises(probe.failure, match='http_403'):
        probe.module.wait_ready(client_for(probe, action), timeout=1)
    assert len(calls) == 1 and probe.now[0] == 0


def test_wait_ready_rejects_invalid_response(probe):
    with pytest.raises(probe.failure):
        probe.module.wait_ready(client_for(probe, lambda *a, **kw: {'authenticated': 'false'}))


@pytest.mark.parametrize('initial', [
    {'ssid': 'unrelated-access-point', 'secured': False, 'active': True},
    {'ssid': 'GESTUR-0000', 'secured': True, 'active': True},
    {'ssid': 'GESTUR-0000', 'secured': False, 'active': True, 'job': {'state': 'applying'}},
])
def test_unexpected_ap_or_pending_job_never_mutates_wifi(initial):
    module = load('api_preconditions', 'scripts/research/smoke_portal_api.py')
    calls = []
    config = {'schema_version': 1, 'active_model': None, 'tracking': {}, 'render': {}, 'controls': {}}
    baseline = {'config': config, 'defaults': config}

    def request(method, endpoint, body=None, **kwargs):
        calls.append((method, endpoint))
        if endpoint == 'session':
            return {'authenticated': method != 'DELETE'}
        assert method == 'GET', 'The smoke must not write outside the expected AP state'
        return {'models': {'models': [], 'active': None}, 'config': baseline, 'wifi': initial,
                'runtime': {'online': True, 'selected_model': None, 'rendered_model': None,
                            'error': None}}[endpoint]

    report = module.run(SimpleNamespace(request=request, cookies=SimpleNamespace(clear=lambda: None)), None)
    assert report['ok'] is False
    assert report['failure'] in {'initial_access_point_not_default_gestur_ssid',
                                 'initial_access_point_not_expected_open_active', 'preexisting_wifi_job'}
    assert not any(method == 'PUT' for method, _ in calls)


@pytest.fixture
def import_cleanup(tmp_path):
    module = load('import_cleanup', 'scripts/check-portal-import.py')
    job_id = '11111111-1111-4111-8111-111111111111'
    name, marker, upload = 'owned-smoke-fixture', 'private-fixture-marker', b'original upload'
    package = tmp_path / job_id
    (package / 'source').mkdir(parents=True)
    (package / '.gestur-model.json').write_text(json.dumps({'name': name, 'entrypoint': 'model.glb'}))
    (package / 'source/smoke-owner.txt').write_text(marker)
    (package / 'original-upload.zip').write_bytes(upload)
    (tmp_path / '.import-job.json').write_text('diagnostic history')
    baseline = {'schema_version': 1, 'active_model': None, 'render': {'target_fps': 60}}
    state = SimpleNamespace(module=module, root=tmp_path, package=package, job_id=job_id,
                            name=name, marker=marker, digest=hashlib.sha256(upload).hexdigest(),
                            baseline=baseline, changed_config=False, foreign_model=False,
                            current_job={'id': job_id, 'name': name, 'state': 'completed'}, calls=[])

    def call(method, endpoint, **kwargs):
        state.calls.append((method, endpoint))
        if method == 'DELETE':
            assert endpoint == '/api/models'
            assert kwargs['value'] == {'id': job_id + '/model.glb'}
            shutil.rmtree(package)
            return {'config': copy.deepcopy(baseline)}
        assert method == 'GET', 'Cleanup must never overwrite settings or selections'
        if endpoint == '/api/imports/current':
            return {'job': copy.deepcopy(state.current_job)}
        model_id = job_id + '/model.glb'
        if endpoint == '/api/config':
            config = copy.deepcopy(baseline)
            if package.exists():
                config['active_model'] = model_id
            if state.changed_config:
                config['render']['target_fps'] = 30
            return {'config': config}
        assert endpoint == '/api/models'
        models = [{'id': model_id}] if package.exists() else []
        if state.foreign_model:
            models.append({'id': '22222222-2222-4222-8222-222222222222/model.glb'})
        return {'models': models, 'active': model_id if package.exists() else None}

    state.client = SimpleNamespace(call=call)
    state.cleanup = lambda: module.cleanup_fixture(
        state.client, tmp_path, name, marker, state.digest, job_id, 1, baseline)
    return state


def test_import_cleanup_removes_own_automatically_selected_fixture_and_reconciles(import_cleanup):
    state = import_cleanup
    result = state.cleanup()
    assert result['removed_own_package'] and result['configuration_restored']
    assert result['deleted_through_api']
    assert ('DELETE', '/api/models') in state.calls
    assert not state.package.exists()
    assert (state.root / '.import-job.json').read_text() == 'diagnostic history'
    assert state.calls[-1] == ('GET', '/api/config')
    assert sum(endpoint == '/api/config' for _, endpoint in state.calls) == 2


@pytest.mark.parametrize('interference', ['settings', 'catalog', 'unfinished_upload', 'job'])
def test_import_cleanup_preserves_fixture_after_concurrent_changes(import_cleanup, interference):
    state = import_cleanup
    if interference == 'settings':
        state.changed_config = True
    elif interference == 'catalog':
        state.foreign_model = True
    elif interference == 'unfinished_upload':
        (state.root / '.upload-22222222-2222-4222-8222-222222222222').mkdir()
    else:
        state.current_job = {'id': '22222222-2222-4222-8222-222222222222',
                             'name': 'somebody-else', 'state': 'processing'}
    before = {str(item.relative_to(state.root)) for item in state.root.rglob('*')}
    with pytest.raises(state.module.SmokeError):
        state.cleanup()
    assert {str(item.relative_to(state.root)) for item in state.root.rglob('*')} == before


@pytest.mark.parametrize('mismatch', ['marker', 'upload', 'name'])
def test_import_cleanup_requires_all_ownership_evidence(import_cleanup, mismatch):
    state = import_cleanup
    if mismatch == 'marker':
        (state.package / 'source/smoke-owner.txt').write_text('not owned')
    elif mismatch == 'upload':
        (state.package / 'original-upload.zip').write_bytes(b'not owned')
    else:
        (state.package / '.gestur-model.json').write_text(json.dumps({'name': 'not owned'}))
    with pytest.raises(state.module.SmokeError):
        state.cleanup()
    assert state.package.exists()


def test_import_cleanup_never_follows_a_package_symlink(import_cleanup):
    state = import_cleanup
    original = state.package.with_name('outside-package')
    state.package.rename(original)
    state.package.symlink_to(original, target_is_directory=True)
    with pytest.raises(state.module.SmokeError, match='Ruta insegura'):
        state.cleanup()
    assert original.is_dir() and state.package.is_symlink()


def test_import_smoke_refuses_populated_library_before_upload(tmp_path, monkeypatch, capsys):
    module = load('import_populated', 'scripts/check-portal-import.py')
    token = tmp_path / 'token'
    token.write_text('test-administrator-token-long-enough')
    models = tmp_path / 'models'
    model = models / '11111111-1111-4111-8111-111111111111' / 'model.glb'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'user-owned-model')
    model_id = model.parent.name + '/model.glb'
    config = {'schema_version': 1, 'active_model': model_id, 'render': {'target_fps': 60}}
    calls = []

    def call(method, endpoint, *args, **kwargs):
        calls.append((method, endpoint))
        if endpoint == '/api/session':
            return {'authenticated': method != 'DELETE'}
        assert method == 'GET', 'An occupied library must never receive an upload or settings write'
        return {'/api/config': {'config': config},
                '/api/models': {'models': [{'id': model_id}], 'active': model_id}}[endpoint]

    monkeypatch.setattr(module, 'Client', lambda _: SimpleNamespace(call=call))
    result = module.run(SimpleNamespace(base_url='http://127.0.0.1', token_file=token,
                                       models_dir=models, timeout=1))
    report = json.loads(capsys.readouterr().out)
    assert result == 1 and report['ok'] is False
    assert 'biblioteca debe estar vacía' in report['error']
    assert model.read_bytes() == b'user-owned-model'
    assert ('POST', '/api/models') not in calls
    assert 'cleanup' not in report
