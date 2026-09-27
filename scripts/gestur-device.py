#!/usr/bin/python3
"""Fixed root-only device operations; JSON requests and secrets arrive on stdin.

Installed root-owned outside the application tree. No request accepts paths or
commands. Public status never includes the password hash or the legacy token.
"""
import copy
from dataclasses import dataclass
import fcntl
import grp
import hashlib
import importlib.machinery
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import pwd
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys


ACTIONS = {'status', 'hostname', 'onboarding', 'reset', 'bootstrap'}
WIFI_HELPER = Path('/usr/local/libexec/gestur-wifi')
REBOOT_UNIT = 'gestur-device-reboot'


@dataclass(frozen=True)
class Paths:
    settings: Path = Path('/etc/gestur')
    data: Path = Path('/var/lib/gestur')
    hosts: Path = Path('/etc/hosts')


@dataclass(frozen=True)
class Owners:
    root: int
    portal_group: int
    data_user: int
    data_group: int
    root_group: int = 0

    @classmethod
    def system(cls):
        return cls(0, grp.getgrnam('gestur-portal').gr_gid,
                   pwd.getpwnam('gestur').pw_uid, grp.getgrnam('gestur').gr_gid)


def validate_hostname(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', value):
        raise ValueError('Invalid hostname')
    return value


def validate_request(action, data):
    fields = {'status': set(), 'hostname': {'hostname'},
              'onboarding': {'hostname', 'password', 'confirmation'},
              'reset': {'confirmation'}, 'bootstrap': {'fresh', 'hostname'}}
    if action not in fields or not isinstance(data, dict) or set(data) - fields[action]:
        raise ValueError('Invalid request')
    if action == 'hostname':
        validate_hostname(data.get('hostname'))
    elif action in ('onboarding', 'bootstrap'):
        hostname = data.get('hostname')
        if hostname not in (None, ''):
            validate_hostname(hostname)
        if action == 'onboarding':
            password = data.get('password')
            if not isinstance(password, str) or not re.fullmatch(r'[\x20-\x7e]{8,63}', password):
                raise ValueError('Invalid password')
            if data.get('confirmation') != password:
                raise ValueError('Passwords differ')
        elif type(data.get('fresh')) is not bool:
            raise ValueError('Bootstrap requires installation type')
    elif action == 'reset' and data.get('confirmation') != 'BORRAR':
        raise ValueError('Explicit confirmation required')
    return data


def password_hash(password):
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode('ascii'), salt=salt, n=16384, r=8, p=1,
                         dklen=32, maxmem=64 * 1024 * 1024)
    return {'algorithm': 'scrypt', 'salt': salt.hex(), 'key': key.hex(),
            'n': 16384, 'r': 8, 'p': 1}


def pending_state():
    return {'version': 1, 'setup_complete': False, 'legacy_auth': False, 'password_hash': None}


def _open_directory(path, owner=None, private=False):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if owner is not None and (info.st_uid != owner or (private and info.st_mode & 0o022)):
        os.close(fd)
        raise ValueError('Unsafe directory permissions')
    return fd


def _read_at(directory, name, owner=None, missing=False, limit=1024 * 1024):
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    except FileNotFoundError:
        if missing:
            return None
        raise
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or (owner is not None and (info.st_uid != owner or info.st_mode & 0o022))):
            raise ValueError('Unsafe state file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            value = stream.read(limit + 1)
        if len(value) > limit:
            raise ValueError('State file too large')
        return value
    finally:
        os.close(fd)


def _write_at(directory, name, content, uid, gid, mode):
    temporary = '.device-' + secrets.token_hex(12)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=directory)
    try:
        os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(fd)
        os.rename(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(fd)
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


def _remove_at(directory, name):
    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if stat.S_ISDIR(info.st_mode):
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RuntimeError('Descriptor-safe deletion is unavailable')
        shutil.rmtree(name, dir_fd=directory)
    else:
        os.unlink(name, dir_fd=directory)


class Platform:
    def __init__(self):
        self.reboot_unit = None

    def hostname(self):
        return socket.gethostname()

    @staticmethod
    def _run(args):
        subprocess.run(args, check=True, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=25, env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'})

    def set_hostname(self, hostname):
        self._run(['/usr/bin/hostnamectl', 'set-hostname', hostname])

    def reboot(self):
        self.reboot_unit = REBOOT_UNIT + '-' + secrets.token_hex(8)
        self._run(['/usr/bin/systemd-run', '--quiet', '--collect',
                   '--unit=' + self.reboot_unit, '--on-active=15s',
                   '--timer-property=AccuracySec=1s', '/usr/bin/systemctl', 'reboot'])

    def cancel_reboot(self):
        if self.reboot_unit:
            self._run(['/usr/bin/systemctl', 'stop', self.reboot_unit + '.timer'])

    def portal_url(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
                # Choosing a route does not send a UDP packet.
                connection.connect(('192.0.2.1', 9))
                address = ipaddress.ip_address(connection.getsockname()[0])
                if not address.is_loopback and not address.is_link_local and not address.is_unspecified:
                    return 'http://' + str(address)
        except OSError:
            pass
        try:
            result = subprocess.run(['/usr/sbin/ip', '-j', '-4', 'address', 'show', 'up'],
                                    check=True, capture_output=True, timeout=3,
                                    env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'})
            for interface in json.loads(result.stdout):
                for item in interface.get('addr_info', []):
                    address = ipaddress.ip_address(item['local'])
                    if item.get('scope') == 'global' and not address.is_loopback and not address.is_link_local:
                        return 'http://' + str(address)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            pass
        return None


class WifiBridge:
    def __init__(self):
        # Never import a module from /opt/gestur or a caller-provided path.
        for path in (*WIFI_HELPER.parents, WIFI_HELPER):
            info = path.lstat()
            if info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode):
                raise ValueError('Untrusted Wi-Fi helper')
        loader = importlib.machinery.SourceFileLoader('_gestur_device_wifi', str(WIFI_HELPER))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        self.module = importlib.util.module_from_spec(spec)
        loader.exec_module(self.module)
        self.nm = self.module.NetworkManager()

    def identity(self):
        device = self.nm.nm.GetDeviceByIpIface('wlan0')
        mac = str(self.nm.prop(device, self.module.BUS_NAME + '.Device.Wireless', 'PermHwAddress'))
        ssid = self.module.ssid_from_mac(mac)
        return ssid.lower(), ssid

    def status(self):
        return self.nm.status()

    def snapshot(self):
        device, object_path, connection, settings = self.nm.profile()
        settings = copy.deepcopy(settings)
        if settings.get('802-11-wireless-security'):
            secured = connection.GetSecrets('802-11-wireless-security')
            settings['802-11-wireless-security'].update(secured.get('802-11-wireless-security', {}))
        active = str(self.nm.prop(device, self.module.BUS_NAME + '.Device', 'ActiveConnection'))
        previous_connection = (str(self.nm.prop(active, self.module.BUS_NAME + '.Connection.Active', 'Connection'))
                               if active != '/' else None)
        return device, object_path, connection, settings, previous_connection

    def apply(self, data):
        return self.nm.apply(data)

    def restore(self, snapshot):
        device, _object_path, connection, settings, previous_connection = snapshot
        connection.Update(settings)
        if previous_connection is not None:
            # A previously connected client network must also be restored when
            # a later hostname/state/timer step fails after activating the AP.
            self.nm.activate(device, previous_connection)
        else:
            current = str(self.nm.prop(device, self.module.BUS_NAME + '.Device', 'ActiveConnection'))
            if current != '/':
                self.nm.nm.DeactivateConnection(current)


class DeviceManager:
    def __init__(self, paths=None, owners=None, platform=None, wifi_factory=WifiBridge):
        self.paths, self.owners = paths or Paths(), owners or Owners.system()
        self.platform = platform or Platform()
        self.wifi_factory = wifi_factory

    def _state(self, settings):
        raw = _read_at(settings, 'device.json', self.owners.root, missing=True)
        if raw is None:
            return pending_state()
        value = json.loads(raw)
        if (not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != 1
                or type(value.get('setup_complete')) is not bool
                or type(value.get('legacy_auth', False)) is not bool):
            raise ValueError('Invalid device state')
        hashed = value.get('password_hash')
        if hashed is not None:
            if (not isinstance(hashed, dict) or set(hashed) != {'algorithm', 'salt', 'key', 'n', 'r', 'p'}
                    or hashed.get('algorithm') != 'scrypt'
                    or not isinstance(hashed.get('salt'), str) or not re.fullmatch('[0-9a-f]{32}', hashed['salt'])
                    or not isinstance(hashed.get('key'), str) or not re.fullmatch('[0-9a-f]{64}', hashed['key'])
                    or any(type(hashed.get(key)) is not int or hashed[key] != expected
                           for key, expected in (('n', 16384), ('r', 8), ('p', 1)))):
                raise ValueError('Invalid password state')
        legacy = value.get('legacy_auth', False)
        if (value['setup_complete'] and bool(hashed) == legacy
                or not value['setup_complete'] and (hashed is not None or legacy)):
            raise ValueError('Inconsistent device state')
        return value

    def status(self):
        settings = _open_directory(self.paths.settings, self.owners.root, private=True)
        try:
            state = self._state(settings)
        finally:
            os.close(settings)
        result = {'hostname': self.platform.hostname(), 'default_hostname': None,
                  'setup_complete': state['setup_complete'],
                  'auth_mode': ('password' if state.get('password_hash') else
                                'legacy' if state.get('legacy_auth') else 'pending'),
                  'ssid': None, 'portal_url': self.platform.portal_url(),
                  'wifi': {'available': False}}
        try:
            wifi = self.wifi_factory()
            result['default_hostname'] = wifi.identity()[0]
            network = wifi.status()
            result['ssid'] = network['ssid']
            result['wifi'] = {'available': True, 'ssid': network['ssid'],
                              'secured': bool(network.get('secured')), 'active': bool(network.get('active'))}
        except Exception:
            result['wifi']['error'] = 'No se pudo consultar el punto de acceso.'
        return result

    def _hosts(self, hostname, previous):
        fd = os.open(self.paths.hosts, os.O_RDWR | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != self.owners.root
                or info.st_nlink != 1 or info.st_mode & 0o022):
            os.close(fd)
            raise ValueError('Unsafe hosts file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            original = stream.read(65537)
        if len(original) > 65536:
            os.close(fd)
            raise ValueError('Hosts file too large')
        try:
            decoded = original.decode('utf-8')
        except UnicodeError:
            os.close(fd)
            raise ValueError('Invalid hosts file') from None
        lines, found = [], False
        for line in decoded.splitlines():
            content, marker, comment = line.partition('#')
            fields = content.split()
            if fields and fields[0] == '127.0.1.1' and not found:
                aliases = [name for name in fields[1:] if name not in (previous, previous + '.local', hostname)]
                line = '127.0.1.1\t' + ' '.join([hostname] + aliases)
                if marker:
                    line += '\t#' + comment
                found = True
            lines.append(line)
        if not found:
            lines.append('127.0.1.1\t' + hostname)
        return fd, original, ('\n'.join(lines) + '\n').encode('utf-8')

    @staticmethod
    def _write_hosts(fd, content):
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(content)
            stream.flush()
            stream.truncate()
            os.fsync(fd)

    def execute(self, action, request):
        validate_request(action, request)
        if action == 'status':
            return self.status()
        settings = _open_directory(self.paths.settings, self.owners.root, private=True)
        lock = os.open('.device.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=settings)
        try:
            info = os.fstat(lock)
            if info.st_uid != self.owners.root or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Unsafe operation lock')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._transaction(action, request, settings)
        finally:
            os.close(lock)
            os.close(settings)
        result = self.status()
        result['reboot_scheduled'] = action in ('hostname', 'onboarding', 'reset')
        return result

    def _transaction(self, action, request, settings):
        old_files = {name: _read_at(settings, name, self.owners.root, missing=True)
                     for name in ('device.json', 'portal-token')}
        state = self._state(settings)
        hostname = request.get('hostname') or self.platform.hostname()
        wifi, wifi_snapshot = None, None
        network_change = None
        write_state = action in ('onboarding', 'reset')
        remove_token = action in ('onboarding', 'reset')
        reboot = action in ('hostname', 'onboarding', 'reset')
        if action == 'onboarding':
            if state['setup_complete']:
                raise ValueError('Setup has already been completed')
            wifi = self.wifi_factory()
            network_change = {'ssid': wifi.status()['ssid'], 'password': request['password']}
            state = {'version': 1, 'setup_complete': True, 'legacy_auth': False,
                     'password_hash': password_hash(request['password'])}
        elif action == 'reset':
            wifi = self.wifi_factory()
            hostname, ssid = wifi.identity()
            network_change = {'ssid': ssid, 'password': None}
            state = pending_state()
        elif action == 'bootstrap' and old_files['device.json'] is None:
            write_state = True
            if request['fresh']:
                wifi = self.wifi_factory()
                default_hostname, ssid = wifi.identity()
                hostname = request.get('hostname') or default_hostname
                network_change = {'ssid': ssid, 'password': None}
                state = pending_state()
                remove_token = True
            else:
                if old_files['portal-token'] is None or len(old_files['portal-token'].strip()) < 24:
                    raise ValueError('Legacy credentials are missing')
                state = {'version': 1, 'setup_complete': True, 'legacy_auth': True, 'password_hash': None}
        defaults = None
        if action == 'reset':
            defaults = _read_at(settings, 'default.json', self.owners.root)
            if not isinstance(json.loads(defaults), dict):
                raise ValueError('Invalid factory defaults')
        previous_hostname = self.platform.hostname()
        hosts_fd, previous_hosts, next_hosts = self._hosts(hostname, previous_hostname)
        data_fd = quarantine_fd = None
        data_info = quarantine = None
        staged, created = [], []
        hostname_changed = wifi_changed = reboot_started = committed = False
        try:
            if wifi is not None:
                wifi_snapshot = wifi.snapshot()
            if action == 'reset':
                data_fd = _open_directory(self.paths.data)
                data_info = os.fstat(data_fd)
                # Pause unprivileged writes during the transaction, including
                # writes by a viewer holding an existing directory descriptor.
                os.fchown(data_fd, self.owners.root, self.owners.root_group)
                os.fchmod(data_fd, 0o700)
                quarantine = '.device-reset-' + secrets.token_hex(16)
                os.mkdir(quarantine, 0o700, dir_fd=data_fd)
                quarantine_fd = os.open(quarantine, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=data_fd)
                for name in os.listdir(data_fd):
                    if name == quarantine:
                        continue
                    entry = os.stat(name, dir_fd=data_fd, follow_symlinks=False)
                    # The public Node job survives long enough to acknowledge
                    # success and remain readable when the Wi-Fi reconnects.
                    if name == 'device-job.json' and stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1:
                        continue
                    os.rename(name, name, src_dir_fd=data_fd, dst_dir_fd=quarantine_fd)
                    staged.append(name)
                created.append('config.json')
                _write_at(data_fd, 'config.json', defaults, self.owners.data_user, self.owners.data_group, 0o660)
                os.mkdir('models', 0o2770, dir_fd=data_fd)
                created.append('models')
                os.chown('models', self.owners.data_user, self.owners.data_group, dir_fd=data_fd, follow_symlinks=False)
                os.chmod('models', 0o2770, dir_fd=data_fd)
            if network_change is not None:
                wifi_changed = True
                wifi.apply(network_change)
            if hostname != previous_hostname:
                hostname_changed = True
                self.platform.set_hostname(hostname)
                self._write_hosts(hosts_fd, next_hosts)
            if write_state:
                _write_at(settings, 'device.json', (json.dumps(state) + '\n').encode(),
                          self.owners.root, self.owners.portal_group, 0o640)
            if remove_token and old_files['portal-token'] is not None:
                os.unlink('portal-token', dir_fd=settings)
            if reboot:
                reboot_started = True
                self.platform.reboot()
            committed = True
        except BaseException:
            # Attempt every independent rollback even if one restoration fails.
            if reboot_started:
                try:
                    self.platform.cancel_reboot()
                except Exception:
                    pass
            for name, content in old_files.items():
                try:
                    if content is None:
                        try:
                            os.unlink(name, dir_fd=settings)
                        except FileNotFoundError:
                            pass
                    else:
                        _write_at(settings, name, content, self.owners.root, self.owners.portal_group, 0o640)
                except Exception:
                    pass
            if hostname_changed:
                try:
                    self.platform.set_hostname(previous_hostname)
                except Exception:
                    pass
                try:
                    self._write_hosts(hosts_fd, previous_hosts)
                except Exception:
                    pass
            if wifi_changed:
                try:
                    wifi.restore(wifi_snapshot)
                except Exception:
                    pass
            if quarantine_fd is not None:
                for name in created:
                    try:
                        _remove_at(data_fd, name)
                    except Exception:
                        pass
                for name in staged:
                    try:
                        os.rename(name, name, src_dir_fd=quarantine_fd, dst_dir_fd=data_fd)
                    except Exception:
                        pass
            raise RuntimeError('Device operation failed; rollback attempted') from None
        finally:
            if data_fd is not None:
                # Restore access before an already scheduled reboot. An
                # interrupted cleanup leaves old data only in root-only trash.
                os.fchown(data_fd, data_info.st_uid, data_info.st_gid)
                os.fchmod(data_fd, stat.S_IMODE(data_info.st_mode))
            if quarantine_fd is not None:
                os.close(quarantine_fd)
                try:
                    if committed:
                        _remove_at(data_fd, quarantine)
                    else:
                        os.rmdir(quarantine, dir_fd=data_fd)
                except OSError:
                    # Never follow a replacement symlink while cleaning up.
                    # Failed rollback backups must be kept, not discarded.
                    pass
            if data_fd is not None:
                os.close(data_fd)
            os.close(hosts_fd)


def _terminate(_signum, _frame):
    # One cancellation enters the normal transaction rollback. A second TERM
    # must not interrupt restoration half-way through; the caller retains a
    # separate, generous hard deadline for an actually stuck native operation.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise RuntimeError('Device operation interrupted')


def main():
    if os.geteuid() != 0 or len(sys.argv) != 2 or sys.argv[1] not in ACTIONS:
        raise ValueError('Not permitted')
    if sys.argv[1] != 'status':
        signal.signal(signal.SIGTERM, _terminate)
    raw = '' if sys.argv[1] == 'status' else sys.stdin.read(4097)
    if len(raw) > 4096:
        raise ValueError('Request too large')
    request = json.loads(raw) if raw.strip() else {}
    result = DeviceManager().execute(sys.argv[1], request)
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # D-Bus and filesystem exceptions may contain a password, never echo.
        print(json.dumps({'error': 'No se pudo completar la operación del dispositivo.'}))
        sys.exit(1)
