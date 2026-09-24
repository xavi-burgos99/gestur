"""No D-Bus, Wi-Fi hardware or elevated privileges used by these tests."""
import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
spec = importlib.util.spec_from_file_location('wifi', Path(__file__).resolve().parents[2] / 'scripts' / 'gestur-wifi.py')
wifi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wifi)

class FakeDBus:
    ByteArray = bytes
    UInt32 = int
    Int32 = int
    Boolean = bool
    @staticmethod
    def Dictionary(value, **kwargs): return dict(value)
    @staticmethod
    def Array(value, **kwargs): return list(value)

class Connection:
    def __init__(self): self.updates = []
    def GetSecrets(self, setting): return {setting: {'psk': 'previous-password'}}
    def Update(self, settings): self.updates.append(copy.deepcopy(settings))

class HelperTests(unittest.TestCase):
    def test_default_name_uses_permanent_mac_suffix(self):
        self.assertEqual(wifi.ssid_from_mac('DC:A6:32:01:ab:cd'),'GESTUR-ABCD')
        for value in ['bad','00:00:00:00:00:00']:
            with self.assertRaises(ValueError): wifi.ssid_from_mac(value)

    def test_validation(self):
        for data in [{}, {'ssid':'x','password':'short'},{'ssid':'x\n'},{'ssid':'x','password':'á'*8},{'ssid':'x','exec':'rm'}]:
            with self.assertRaises(ValueError): wifi.validate_request(data)
        wifi.validate_request({'ssid':'GESTUR-ABCD','password':None})
        wifi.validate_request({'ssid':'GESTUR-ABCD','password':'valid123'})

    def helper(self):
        helper = wifi.NetworkManager.__new__(wifi.NetworkManager)
        helper.dbus = FakeDBus
        connection = Connection()
        settings = {'connection': {'uuid':'fixed'},'802-11-wireless': {'mode':'ap','ssid':b'Old','security':'802-11-wireless-security'},'802-11-wireless-security': {'key-mgmt':'wpa-psk'}}
        helper.profile=lambda: ('device','profile',connection,copy.deepcopy(settings))
        helper.activate=lambda *args: None
        helper.status=lambda: {'ssid':'new','secured':True,'active':True}
        return helper,connection

    def test_rename_preserves_password_and_remove_clears_security(self):
        helper, connection = self.helper()
        helper.apply({'ssid':'Renamed'})
        self.assertEqual(connection.updates[0]['802-11-wireless-security']['psk'],'previous-password')
        helper.apply({'ssid':'Open','password':None})
        self.assertNotIn('802-11-wireless-security',connection.updates[1])
        self.assertNotIn('security',connection.updates[1]['802-11-wireless'])

    def test_new_password_and_failed_activation_roll_back(self):
        helper, connection = self.helper()
        calls=[]
        def activate(*args):
            calls.append(args)
            if len(calls)==1: raise RuntimeError('secret failure')
        helper.activate=activate
        with self.assertRaises(RuntimeError): helper.apply({'ssid':'New','password':'new-password'})
        self.assertEqual(connection.updates[0]['802-11-wireless-security']['psk'],'new-password')
        self.assertEqual(connection.updates[1]['802-11-wireless-security']['psk'],'previous-password')
        self.assertEqual(len(calls),2)

    def test_first_install_creates_open_mac_named_access_point(self):
        helper = wifi.NetworkManager.__new__(wifi.NetworkManager)
        helper.dbus = FakeDBus
        enabled, created, activated = [], [], []
        helper.interface = lambda *args: SimpleNamespace(Set=lambda *args: enabled.append(args))
        helper.nm = SimpleNamespace(GetDeviceByIpIface=lambda interface: 'wlan-device')
        def add(settings):
            created.append(copy.deepcopy(settings))
            return 'new-profile'
        helper.settings = SimpleNamespace(ListConnections=lambda: [], AddConnection=add)
        helper.prop = lambda *args: 'DC:A6:32:01:AB:CD'
        helper.active = lambda *args: False
        helper.activate = lambda *args: activated.append(args)
        helper.status = lambda: {'ssid': 'GESTUR-ABCD', 'secured': False, 'active': True}
        with tempfile.TemporaryDirectory() as directory, patch.object(wifi, 'PROFILE', Path(directory) / 'wifi-profile.json'):
            result = helper.bootstrap()
            self.assertTrue(wifi.PROFILE.exists())
            self.assertEqual(wifi.PROFILE.stat().st_mode & 0o777, 0o600)
        self.assertFalse(result['secured'])
        self.assertEqual(created[0]['802-11-wireless']['ssid'], b'GESTUR-ABCD')
        self.assertNotIn('802-11-wireless-security', created[0])
        self.assertEqual(created[0]['ipv4']['method'], 'shared')
        self.assertEqual(activated, [('wlan-device', 'new-profile')])
        self.assertTrue(enabled)

    def test_reinstall_reactivates_existing_profile_without_modifying_credentials(self):
        helper, connection = self.helper()
        helper.interface = lambda *args: SimpleNamespace(Set=lambda *args: None)
        helper.active = lambda *args: False
        activated = []
        helper.activate = lambda *args: activated.append(args)
        with tempfile.TemporaryDirectory() as directory, patch.object(wifi, 'PROFILE', Path(directory) / 'wifi-profile.json'):
            wifi.PROFILE.write_text('{"uuid":"fixed","interface":"wlan0"}')
            helper.bootstrap()
        self.assertEqual(connection.updates, [])
        self.assertEqual(activated, [('device', 'profile')])

if __name__ == '__main__': unittest.main()
