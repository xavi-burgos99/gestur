"""Exercise privileged device transactions against temporary, unprivileged files."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location('gestur_device_helper', Path(__file__).parents[1] / 'scripts/gestur-device.py')
device = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = device
SPEC.loader.exec_module(device)


class FakePlatform:
    def __init__(self):
        self.name = 'gestur-anterior'
        self.reboots = self.cancelled = 0
        self.fail_reboot = self.fail_hostname = False

    def hostname(self):
        return self.name

    def portal_url(self):
        return 'http://192.168.1.93'

    def set_hostname(self, value):
        self.name = value
        if self.fail_hostname:
            self.fail_hostname = False
            raise RuntimeError('Partial hostname failure')

    def reboot(self):
        self.reboots += 1
        if self.fail_reboot:
            raise RuntimeError('Timer failure')

    def cancel_reboot(self):
        self.cancelled += 1


class FakeWifi:
    def __init__(self):
        self.settings = {'ssid': 'Nombre actual', 'password': 'ClaveAnterior', 'active': True}
        self.applied = []
        self.restored = 0
        self.fail = False

    def identity(self):
        return 'gestur-a9bc', 'GESTUR-A9BC'

    def status(self):
        return {'ssid': self.settings['ssid'], 'secured': bool(self.settings['password']), 'active': self.settings['active']}

    def snapshot(self):
        return copy.deepcopy(self.settings)

    def apply(self, data):
        self.applied.append(copy.deepcopy(data))
        self.settings.update(data)
        if self.fail:
            raise RuntimeError('Backend error containing ' + str(data.get('password')))

    def restore(self, value):
        self.settings = value
        self.restored += 1


@pytest.fixture
def rig(tmp_path):
    settings, data, hosts = tmp_path/'etc-gestur', tmp_path/'data', tmp_path/'hosts'
    settings.mkdir(mode=0o755)
    data.mkdir(mode=0o2770)
    data.chmod(0o2770)
    hosts.write_text('127.0.0.1 localhost\n127.0.1.1 gestur-anterior other-alias # keep\n::1 localhost\n')
    defaults = {'schema_version': 1, 'active_model': None, 'tracking': {'use_hands': False}}
    (settings/'default.json').write_text(json.dumps(defaults))
    (data/'config.json').write_text('{"active_model":"previous"}')
    (data/'models').mkdir()
    (data/'models'/'valuable.glb').write_bytes(b'original mesh and textures')
    (data/'runtime.json').write_text('{"running":true}')
    (data/'presets.json').write_text('{"presets":[{"name":"Mi exposición","settings":{"render":{"exposure":73}}}]}')
    (data/'imports').mkdir()
    (data/'imports'/'partial.zip').write_bytes(b'partial')
    (data/'device-job.json').write_text('{"id":"job-current","state":"running"}')
    platform, wifi = FakePlatform(), FakeWifi()
    owners = device.Owners(os.getuid(), os.getgid(), os.getuid(), os.getgid(), os.getgid())
    manager = device.DeviceManager(device.Paths(settings, data, hosts), owners, platform, lambda: wifi)
    return manager, settings, data, hosts, platform, wifi, defaults


def save_state(settings, value):
    path = settings/'device.json'
    path.write_text(json.dumps(value))
    path.chmod(0o640)


def onboarding(**extra):
    return {'password': 'NuevaClave123', 'confirmation': 'NuevaClave123', **extra}


@pytest.mark.parametrize('value', ['a', 'gestur-a9bc', '1', 'a'*63])
def test_hostname_accepts_only_literal_dns_labels(value):
    assert device.validate_hostname(value) == value


@pytest.mark.parametrize('value', ['', '-host', 'host-', 'Upper', 'host.local', 'a'*64, 'a b', 'a\n', ';reboot', None, 12])
def test_invalid_hostname_is_rejected(value):
    with pytest.raises(ValueError):
        device.validate_hostname(value)


@pytest.mark.parametrize('value', ['short', 'á'*8, 'a'*64, 'password\n', None])
def test_password_constraints_are_shared_with_wifi(value):
    with pytest.raises(ValueError):
        device.validate_request('onboarding', {'password': value, 'confirmation': value})


def test_unknown_fields_and_missing_confirmation_are_rejected():
    for action, value in [('reset', {}), ('reset', {'confirmation': True}),
                          ('onboarding', onboarding(confirmation='different')),
                          ('hostname', {'hostname': 'valid', 'path': '/etc/passwd'}),
                          ('bootstrap', {'fresh': 1})]:
        with pytest.raises(ValueError):
            device.validate_request(action, value)


def test_status_stays_available_when_wifi_fails_and_never_exposes_credentials(rig):
    manager, settings, _, _, _, _, _ = rig
    save_state(settings, {'version': 1, 'setup_complete': True, 'legacy_auth': False,
                          'password_hash': device.password_hash('NuevaClave123')})
    def broken():
        raise RuntimeError('secret details')
    manager.wifi_factory = broken
    value = manager.status()
    assert value['hostname'] == 'gestur-anterior'
    assert value['portal_url'] == 'http://192.168.1.93'
    assert value['setup_complete'] and value['auth_mode'] == 'password'
    assert value['wifi']['available'] is False
    serialized = json.dumps(value)
    assert all(word not in serialized for word in ('password_hash', 'salt', 'secret details', 'NuevaClave123'))


def test_status_whitelists_wifi_fields_instead_of_forwarding_backend_secrets(rig):
    manager, _, _, _, _, wifi, _ = rig
    wifi.status = lambda: {'ssid': 'Test', 'active': True, 'secured': True, 'password': 'do-not-output'}
    result = manager.status()
    assert result['wifi'] == {'available': True, 'ssid': 'Test', 'active': True, 'secured': True}
    assert 'do-not-output' not in json.dumps(result)


def test_onboarding_sets_same_wifi_password_and_hash_without_plaintext(rig):
    manager, settings, data, hosts, platform, wifi, _ = rig
    save_state(settings, device.pending_state())
    before = (data/'config.json').read_bytes()
    result = manager.execute('onboarding', onboarding(hostname='sala-museo'))
    state = json.loads((settings/'device.json').read_text())
    hashed = state['password_hash']
    assert set(hashed) == {'algorithm', 'salt', 'key', 'n', 'r', 'p'}
    assert hashed['algorithm'] == 'scrypt' and len(hashed['salt']) == 32 and len(hashed['key']) == 64
    assert hashlib.scrypt(b'NuevaClave123', salt=bytes.fromhex(hashed['salt']), n=16384, r=8, p=1,
                          dklen=32, maxmem=64*1024*1024).hex() == hashed['key']
    assert wifi.settings['password'] == 'NuevaClave123' and wifi.settings['ssid'] == 'Nombre actual'
    assert platform.name == 'sala-museo' and platform.reboots == 1
    assert '127.0.1.1\tsala-museo other-alias\t# keep' in hosts.read_text()
    assert result['setup_complete'] and result['reboot_scheduled']
    assert 'NuevaClave123' not in (settings/'device.json').read_text()
    assert 'key' not in json.dumps(result) and 'salt' not in json.dumps(result)
    assert stat.S_IMODE((settings/'device.json').stat().st_mode) == 0o640
    assert (data/'config.json').read_bytes() == before


def test_blank_onboarding_hostname_keeps_current_hostname(rig):
    manager, _, _, _, platform, _, _ = rig
    manager.execute('onboarding', onboarding(hostname=''))
    assert platform.name == 'gestur-anterior'


def test_onboarding_cannot_overwrite_an_already_configured_device(rig):
    manager, settings, _, _, platform, wifi, _ = rig
    save_state(settings, {'version': 1, 'setup_complete': True, 'legacy_auth': True, 'password_hash': None})
    with pytest.raises(ValueError):
        manager.execute('onboarding', onboarding())
    assert not wifi.applied and platform.reboots == 0


@pytest.mark.parametrize('failure', ['wifi', 'hostname', 'timer'])
def test_onboarding_rolls_back_all_changes_if_any_step_fails(rig, failure):
    manager, settings, _, hosts, platform, wifi, _ = rig
    save_state(settings, device.pending_state())
    old_state, old_hosts, old_wifi = (settings/'device.json').read_bytes(), hosts.read_bytes(), wifi.snapshot()
    token = 'oldtoken'*5
    (settings/'portal-token').write_text(token)
    wifi.fail = failure == 'wifi'
    platform.fail_hostname = failure == 'hostname'
    platform.fail_reboot = failure == 'timer'
    with pytest.raises(RuntimeError, match='rollback attempted'):
        manager.execute('onboarding', onboarding(hostname='nuevo'))
    assert (settings/'device.json').read_bytes() == old_state
    assert (settings/'portal-token').read_text() == token
    assert hosts.read_bytes() == old_hosts and platform.name == 'gestur-anterior'
    assert wifi.settings == old_wifi
    if failure == 'timer':
        assert platform.cancelled == 1


def test_fresh_bootstrap_uses_permanent_mac_identity_and_is_pending(rig):
    manager, settings, _, _, platform, wifi, _ = rig
    result = manager.execute('bootstrap', {'fresh': True})
    assert platform.name == 'gestur-a9bc' and platform.reboots == 0
    assert wifi.settings['ssid'] == 'GESTUR-A9BC' and wifi.settings['password'] is None
    assert json.loads((settings/'device.json').read_text()) == device.pending_state()
    assert not result['setup_complete'] and result['auth_mode'] == 'pending'


def test_legacy_bootstrap_preserves_credentials_hostname_wifi_and_data(rig):
    manager, settings, data, _, platform, wifi, _ = rig
    (settings/'portal-token').write_text('LegacyToken'*4)
    old_wifi = wifi.snapshot()
    old_config = (data/'config.json').read_bytes()
    result = manager.execute('bootstrap', {'fresh': False})
    assert result['setup_complete'] and result['auth_mode'] == 'legacy'
    assert (settings/'portal-token').read_text() == 'LegacyToken'*4
    assert platform.name == 'gestur-anterior' and platform.reboots == 0
    assert wifi.settings == old_wifi and not wifi.applied
    assert (data/'config.json').read_bytes() == old_config


def test_bootstrap_is_idempotent_even_if_fresh_is_incorrectly_repeated(rig):
    manager, settings, _, _, platform, wifi, _ = rig
    manager.execute('onboarding', onboarding())
    original, network = (settings/'device.json').read_bytes(), wifi.snapshot()
    manager.execute('bootstrap', {'fresh': True})
    assert (settings/'device.json').read_bytes() == original and wifi.settings == network
    assert platform.name == 'gestur-anterior'


def test_legacy_bootstrap_does_not_create_a_new_legacy_token(rig):
    manager, settings, _, _, _, _, _ = rig
    with pytest.raises(ValueError, match='credentials'):
        manager.execute('bootstrap', {'fresh': False})
    assert not (settings/'portal-token').exists() and not (settings/'device.json').exists()


def test_factory_reset_clears_only_device_data_and_preserves_public_job(rig):
    manager, settings, data, _, platform, wifi, defaults = rig
    (settings/'portal-token').write_text('LegacyToken'*4)
    save_state(settings, {'version': 1, 'setup_complete': True, 'legacy_auth': True, 'password_hash': None})
    previous_mode = stat.S_IMODE(data.stat().st_mode)
    result = manager.execute('reset', {'confirmation': 'BORRAR'})
    assert sorted(item.name for item in data.iterdir()) == ['config.json', 'device-job.json', 'models']
    assert json.loads((data/'config.json').read_text()) == defaults
    assert not list((data/'models').iterdir())
    assert not (data/'presets.json').exists()
    assert 'job-current' in (data/'device-job.json').read_text()
    assert not (settings/'portal-token').exists()
    assert json.loads((settings/'device.json').read_text()) == device.pending_state()
    assert platform.name == 'gestur-a9bc' and platform.reboots == 1
    assert wifi.settings['ssid'] == 'GESTUR-A9BC' and wifi.settings['password'] is None
    assert stat.S_IMODE(data.stat().st_mode) == previous_mode
    assert stat.S_IMODE((data/'models').stat().st_mode) == 0o2770
    assert result['reboot_scheduled'] and not result['setup_complete']


def test_failed_reset_restores_models_runtime_config_credentials_and_permissions(rig):
    manager, settings, data, hosts, platform, wifi, _ = rig
    token = 'LegacyToken'*4
    (settings/'portal-token').write_text(token)
    save_state(settings, {'version': 1, 'setup_complete': True, 'legacy_auth': True, 'password_hash': None})
    old_hosts, old_wifi = hosts.read_bytes(), wifi.snapshot()
    platform.fail_reboot = True
    with pytest.raises(RuntimeError, match='rollback attempted'):
        manager.execute('reset', {'confirmation': 'BORRAR'})
    assert (data/'models'/'valuable.glb').read_bytes() == b'original mesh and textures'
    assert json.loads((data/'config.json').read_text()) == {'active_model': 'previous'}
    assert (data/'imports'/'partial.zip').read_bytes() == b'partial'
    assert (data/'runtime.json').exists()
    assert json.loads((data/'presets.json').read_text())['presets'][0]['name'] == 'Mi exposición'
    assert (settings/'portal-token').read_text() == token
    assert json.loads((settings/'device.json').read_text())['legacy_auth'] is True
    assert wifi.settings == old_wifi and hosts.read_bytes() == old_hosts
    assert platform.name == 'gestur-anterior' and platform.cancelled == 1
    assert stat.S_IMODE(data.stat().st_mode) == 0o2770
    assert not list(data.glob('.device-reset-*'))


def test_reset_never_follows_symlinks_inside_model_packages(rig, tmp_path):
    manager, _, data, _, _, _, _ = rig
    outside = tmp_path/'outside'
    outside.mkdir()
    (outside/'keep.txt').write_text('do not delete')
    (data/'models'/'external').symlink_to(outside, target_is_directory=True)
    manager.execute('reset', {'confirmation': 'BORRAR'})
    assert (outside/'keep.txt').read_text() == 'do not delete'


def test_symlink_data_root_is_rejected_without_modifying_the_target(rig, tmp_path):
    manager, _, data, _, platform, wifi, _ = rig
    alias = tmp_path/'data-alias'
    alias.symlink_to(data, target_is_directory=True)
    manager.paths = device.Paths(manager.paths.settings, alias, manager.paths.hosts)
    with pytest.raises(RuntimeError):
        manager.execute('reset', {'confirmation': 'BORRAR'})
    assert (data/'models'/'valuable.glb').exists() and not wifi.applied and platform.reboots == 0


@pytest.mark.parametrize('name', ['device.json', 'portal-token', 'default.json'])
def test_symlink_security_files_are_rejected(rig, tmp_path, name):
    manager, settings, _, _, platform, _, _ = rig
    target = tmp_path/'secret-target'
    target.write_text('{"version":1,"setup_complete":false}')
    path = settings/name
    path.unlink(missing_ok=True)
    path.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        manager.execute('reset', {'confirmation': 'BORRAR'})
    assert target.read_text() == '{"version":1,"setup_complete":false}' and platform.reboots == 0


def test_hostname_operation_preserves_wifi_and_credentials(rig):
    manager, settings, _, _, platform, wifi, _ = rig
    save_state(settings, device.pending_state())
    old_state = (settings/'device.json').read_bytes()
    manager.execute('hostname', {'hostname': 'sala-2'})
    assert platform.name == 'sala-2' and platform.reboots == 1
    assert not wifi.applied and (settings/'device.json').read_bytes() == old_state


def test_main_status_does_not_wait_for_stdin(monkeypatch, capsys):
    class NoRead:
        def read(self, *_):
            pytest.fail('status must not wait for stdin')
    monkeypatch.setattr(device.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(device.sys, 'argv', ['gestur-device', 'status'])
    monkeypatch.setattr(device.sys, 'stdin', NoRead())
    monkeypatch.setattr(device, 'DeviceManager', lambda: type('Manager', (), {'execute': lambda self, a, d: {'hostname': 'gestur-test'}})())
    device.main()
    assert json.loads(capsys.readouterr().out) == {'hostname': 'gestur-test'}


def test_main_rejects_non_root_and_oversized_requests(monkeypatch):
    monkeypatch.setattr(device.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(device.os, 'geteuid', lambda: 501)
    monkeypatch.setattr(device.sys, 'argv', ['gestur-device', 'reset'])
    with pytest.raises(ValueError, match='Not permitted'):
        device.main()
    monkeypatch.setattr(device.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(device.sys, 'stdin', io.StringIO('x'*4097))
    with pytest.raises(ValueError, match='too large'):
        device.main()


def test_wifi_rollback_restores_the_previous_client_connection():
    settings = {'connection': {'uuid': 'ap-uuid'}, '802-11-wireless': {'mode': 'ap'},
                '802-11-wireless-security': {'key-mgmt': 'wpa-psk'}}
    updated, activated, deactivated = [], [], []
    connection = SimpleNamespace(GetSecrets=lambda name: {name: {'psk': 'previous-secret'}},
                                 Update=lambda value: updated.append(copy.deepcopy(value)))
    def prop(path, _interface, key):
        if path == '/device' and key == 'ActiveConnection':
            return '/active-client'
        if path == '/active-client' and key == 'Connection':
            return '/settings/client'
        pytest.fail((path, key))
    bridge = device.WifiBridge.__new__(device.WifiBridge)
    bridge.module = SimpleNamespace(BUS_NAME='org.freedesktop.NetworkManager')
    bridge.nm = SimpleNamespace(profile=lambda: ('/device', '/settings/ap', connection, settings),
                                prop=prop, activate=lambda d, p: activated.append((d, p)),
                                nm=SimpleNamespace(DeactivateConnection=lambda p: deactivated.append(p)))
    saved = bridge.snapshot()
    bridge.restore(saved)
    assert updated[0]['802-11-wireless-security']['psk'] == 'previous-secret'
    assert activated == [('/device', '/settings/client')] and not deactivated


def test_each_reboot_timer_has_its_own_cancellation_scope(monkeypatch):
    first, second = device.Platform(), device.Platform()
    calls = []
    monkeypatch.setattr(device.Platform, '_run', staticmethod(lambda args: calls.append(args)))
    first.reboot()
    second.reboot()
    second.cancel_reboot()
    assert first.reboot_unit != second.reboot_unit
    assert '--on-active=15s' in calls[0] and '--on-active=15s' in calls[1]
    assert calls[2] == ['/usr/bin/systemctl', 'stop', second.reboot_unit + '.timer']
    assert first.reboot_unit + '.timer' not in calls[2]


def test_hostname_change_does_not_require_working_wifi(rig):
    manager, _, _, _, platform, _, _ = rig
    def broken():
        raise RuntimeError('Wi-Fi unavailable')
    manager.wifi_factory = broken
    result = manager.execute('hostname', {'hostname': 'solo-ethernet'})
    assert platform.name == 'solo-ethernet' and not result['wifi']['available']


@pytest.mark.parametrize('change', [
    {'version': True}, {'setup_complete': True}, {'legacy_auth': True},
    {'password_hash': {'algorithm': 'scrypt', 'salt': '0'*32, 'key': '0'*64, 'n': 2**30, 'r': 8, 'p': 1}},
])
def test_corrupted_state_cannot_silently_change_authentication_mode(rig, change):
    manager, settings, _, _, _, _, _ = rig
    save_state(settings, {**device.pending_state(), **change})
    with pytest.raises(ValueError):
        manager.status()


def test_termination_enters_rollback_and_ignores_repeated_signals(monkeypatch):
    handlers = []
    monkeypatch.setattr(device.signal, 'signal', lambda signum, handler: handlers.append((signum, handler)))
    with pytest.raises(RuntimeError, match='interrupted'):
        device._terminate(device.signal.SIGTERM, None)
    assert handlers == [(device.signal.SIGTERM, device.signal.SIG_IGN)]


def test_mutating_entrypoint_installs_cancellation_handler(monkeypatch, capsys):
    handlers = []
    monkeypatch.setattr(device.signal, 'signal', lambda signum, handler: handlers.append((signum, handler)))
    monkeypatch.setattr(device.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(device.sys, 'argv', ['gestur-device', 'hostname'])
    monkeypatch.setattr(device.sys, 'stdin', io.StringIO('{"hostname":"gestur-nuevo"}'))
    monkeypatch.setattr(device, 'DeviceManager', lambda: type('Manager', (), {'execute': lambda self, a, d: {'hostname': d['hostname']}})())
    device.main()
    assert handlers == [(device.signal.SIGTERM, device._terminate)]
    assert json.loads(capsys.readouterr().out)['hostname'] == 'gestur-nuevo'


def test_interruption_during_reset_restores_data_and_permissions(rig):
    manager, settings, data, _, platform, _, _ = rig
    def interrupted():
        raise KeyboardInterrupt()
    platform.reboot = interrupted
    with pytest.raises(RuntimeError, match='rollback attempted'):
        manager.execute('reset', {'confirmation': 'BORRAR'})
    assert (data/'models'/'valuable.glb').exists()
    assert json.loads((data/'config.json').read_text()) == {'active_model': 'previous'}
    assert stat.S_IMODE(data.stat().st_mode) == 0o2770
    assert not (settings/'device.json').exists()
