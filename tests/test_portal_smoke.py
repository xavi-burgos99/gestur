"""Exercise startup waiting and AP preconditions with no sockets or device access."""
import importlib.util
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
