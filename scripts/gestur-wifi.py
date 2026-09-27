#!/usr/bin/python3
"""Narrow root-only NetworkManager bridge. Requests on stdin; secrets never in argv.
Installed outside the writable application/data tree. No arbitrary commands or profiles.
"""
import copy
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

PROFILE = Path('/etc/gestur/wifi-profile.json')
BUS_NAME = 'org.freedesktop.NetworkManager'
NM_PATH = '/org/freedesktop/NetworkManager'
SETTINGS_PATH = NM_PATH + '/Settings'
CONNECTION_IFACE = BUS_NAME + '.Settings.Connection'


def validate_request(data):
    if not isinstance(data, dict) or set(data) - {'ssid', 'password'}:
        raise ValueError('Invalid request')
    ssid = data.get('ssid')
    if not isinstance(ssid, str) or not 1 <= len(ssid.encode('utf-8')) <= 32 or re.search(r'[\x00-\x1f\x7f]', ssid):
        raise ValueError('Invalid SSID')
    password = data.get('password')
    if password is not None and (not isinstance(password, str) or not re.fullmatch(r'[\x20-\x7e]{8,63}', password)):
        raise ValueError('Invalid password')
    return data


def ssid_from_mac(mac):
    if not re.fullmatch(r'(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}', mac) or mac == '00:00:00:00:00:00':
        raise ValueError('Permanent hardware address unavailable')
    return 'GESTUR-' + mac.replace(':', '')[-4:].upper()


class NetworkManager:
    def __init__(self):
        import dbus
        self.dbus = dbus
        self.bus = dbus.SystemBus()
        self.nm = self.interface(NM_PATH, BUS_NAME)
        self.settings = self.interface(SETTINGS_PATH, BUS_NAME + '.Settings')

    def interface(self, object_path, interface):
        return self.dbus.Interface(self.bus.get_object(BUS_NAME, object_path), interface)

    def prop(self, object_path, interface, name):
        return self.interface(object_path, 'org.freedesktop.DBus.Properties').Get(interface, name)

    def connection(self, object_path):
        return self.interface(object_path, CONNECTION_IFACE)

    def find(self, profile_uuid):
        for object_path in self.settings.ListConnections():
            connection = self.connection(object_path)
            if str(connection.GetSettings()['connection']['uuid']) == profile_uuid:
                return object_path, connection
        raise ValueError('Configured access point no longer exists')

    def profile(self):
        data = json.loads(PROFILE.read_text())
        device = self.nm.GetDeviceByIpIface(data['interface'])
        object_path, connection = self.find(data['uuid'])
        settings = connection.GetSettings()
        if settings.get('802-11-wireless', {}).get('mode') != 'ap':
            raise ValueError('Profile is not an access point')
        return device, object_path, connection, settings

    def active(self, device, profile_uuid):
        active = str(self.prop(device, BUS_NAME + '.Device', 'ActiveConnection'))
        return active != '/' and str(self.prop(active, BUS_NAME + '.Connection.Active', 'Uuid')) == profile_uuid and int(self.prop(active, BUS_NAME + '.Connection.Active', 'State')) == 2

    def status(self):
        device, _, _, settings = self.profile()
        return {'ssid': bytes(settings['802-11-wireless']['ssid']).decode('utf-8'),
                'secured': bool(settings.get('802-11-wireless-security')),
                'active': self.active(device, str(settings['connection']['uuid']))}

    def active_status(self):
        result = self.status()
        if not result['active']:
            raise RuntimeError('Access point is not active')
        return result

    def wait_device_ready(self, device):
        # WirelessEnabled/rfkill can return before the supplicant is available.
        # NMDeviceState 20 (UNAVAILABLE) is transient during that startup. Wait
        # for state 30 (DISCONNECTED) or an active/activating connection before
        # asking NM to activate; never retry failed activation requests blindly.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not self.prop(NM_PATH, BUS_NAME, 'WirelessHardwareEnabled'):
                raise RuntimeError('Wireless hardware is disabled')
            if not self.prop(device, BUS_NAME + '.Device', 'Managed'):
                raise RuntimeError('Wireless device is unmanaged')
            if self.prop(device, BUS_NAME + '.Device', 'FirmwareMissing'):
                raise RuntimeError('Wireless device firmware is missing')
            state = int(self.prop(device, BUS_NAME + '.Device', 'State'))
            if state == 120:
                raise RuntimeError('Wireless device failed')
            if self.prop(NM_PATH, BUS_NAME, 'WirelessEnabled') and 30 <= state <= 100:
                return
            if state not in (0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110):
                raise RuntimeError('Wireless device cannot activate a connection')
            time.sleep(.25)
        raise RuntimeError('Wireless device readiness timed out')

    def activate(self, device, object_path):
        self.wait_device_ready(device)
        active = str(self.prop(device, BUS_NAME + '.Device', 'ActiveConnection'))
        if active != '/':
            self.nm.DeactivateConnection(active)
        active = self.nm.ActivateConnection(object_path, device, '/')
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            state = int(self.prop(active, BUS_NAME + '.Connection.Active', 'State'))
            if state == 2:
                if str(self.prop(device, BUS_NAME + '.Device', 'ActiveConnection')) != str(active):
                    raise RuntimeError('Access point is not active on the wireless device')
                return
            if state == 4:
                raise RuntimeError('Access point activation failed')
            time.sleep(.25)
        raise RuntimeError('Access point activation timed out')

    def apply(self, data):
        validate_request(data)
        device, object_path, connection, settings = self.profile()
        if settings.get('802-11-wireless-security'):
            # GetSettings intentionally omits secrets. Preserve them for rename and rollback.
            secrets = connection.GetSecrets('802-11-wireless-security')
            settings['802-11-wireless-security'].update(secrets.get('802-11-wireless-security', {}))
        old = copy.deepcopy(settings)
        settings['802-11-wireless']['ssid'] = self.dbus.ByteArray(data['ssid'].encode('utf-8'))
        if 'password' in data:
            if data['password'] is None:
                settings.pop('802-11-wireless-security', None)
                settings['802-11-wireless'].pop('security', None)
            else:
                settings['802-11-wireless-security'] = self.dbus.Dictionary({
                    'key-mgmt': 'wpa-psk', 'psk': data['password'], 'psk-flags': self.dbus.UInt32(0),
                    'proto': self.dbus.Array(['rsn'], signature='s'),
                }, signature='sv')
                settings['802-11-wireless']['security'] = '802-11-wireless-security'
        try:
            connection.Update(settings)
            self.activate(device, object_path)
            result = self.active_status()
        except Exception:
            try:
                connection.Update(old)
                self.activate(device, object_path)
            except Exception:
                pass
            raise RuntimeError('Access point update failed; rollback attempted') from None
        return result

    def bootstrap(self):
        self.interface(NM_PATH, 'org.freedesktop.DBus.Properties').Set(BUS_NAME, 'WirelessEnabled', self.dbus.Boolean(True))
        if PROFILE.exists():
            device, object_path, _, settings = self.profile()
            if not self.active(device, str(settings['connection']['uuid'])):
                self.activate(device, object_path)
            return self.active_status()  # Keep the chosen AP and its credentials on reinstall.
        device = self.nm.GetDeviceByIpIface('wlan0')
        candidates = []
        for object_path in self.settings.ListConnections():
            settings = self.connection(object_path).GetSettings()
            connection = settings['connection']
            if settings.get('802-11-wireless', {}).get('mode') == 'ap' and connection.get('interface-name', 'wlan0') == 'wlan0':
                candidates.append((object_path, settings))
        if len(candidates) > 1:
            raise ValueError('Multiple wlan0 access points: select one in /etc/gestur/wifi-profile.json')
        if candidates:
            object_path, settings = candidates[0]
        else:
            mac = str(self.prop(device, BUS_NAME + '.Device.Wireless', 'PermHwAddress'))
            name = ssid_from_mac(mac)
            settings = self.dbus.Dictionary({
                'connection': self.dbus.Dictionary({'id': 'gestur-ap', 'uuid': str(uuid.uuid4()), 'type': '802-11-wireless', 'interface-name': 'wlan0', 'autoconnect': True, 'autoconnect-priority': self.dbus.Int32(100)}, signature='sv'),
                '802-11-wireless': self.dbus.Dictionary({'ssid': self.dbus.ByteArray(name.encode()), 'mode': 'ap', 'band': 'bg'}, signature='sv'),
                'ipv4': self.dbus.Dictionary({'method': 'shared', 'address-data': self.dbus.Array([self.dbus.Dictionary({'address': '10.42.0.1', 'prefix': self.dbus.UInt32(24)}, signature='sv')], signature='a{sv}')}, signature='sv'),
                'ipv6': self.dbus.Dictionary({'method': 'disabled'}, signature='sv'),
            }, signature='sa{sv}')
            object_path = self.settings.AddConnection(settings)
        PROFILE.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        temporary = PROFILE.with_suffix('.tmp')
        temporary.write_text(json.dumps({'uuid': str(settings['connection']['uuid']), 'interface': 'wlan0'}) + '\n')
        temporary.chmod(0o600)
        temporary.replace(PROFILE)
        if not self.active(device, str(settings['connection']['uuid'])):
            self.activate(device, object_path)
        return self.active_status()


def main():
    if os.geteuid() != 0 or len(sys.argv) != 2 or sys.argv[1] not in {'status', 'apply', 'bootstrap'}:
        raise ValueError('Not permitted')
    nm = NetworkManager()
    if sys.argv[1] == 'apply':
        raw = sys.stdin.read(4097)
        if len(raw) > 4096:
            raise ValueError('Request too large')
        result = nm.apply(validate_request(json.loads(raw)))
    elif sys.argv[1] == 'bootstrap':
        result = nm.bootstrap()
    else:
        result = nm.status()
    print(json.dumps(result))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        # NetworkManager exceptions can contain secrets. Never serialize them.
        print(json.dumps({'error': 'NetworkManager operation failed'}))
        sys.exit(1)
