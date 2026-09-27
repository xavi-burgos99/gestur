"""No D-Bus, Wi-Fi hardware or elevated privileges used by these tests."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "wifi", Path(__file__).resolve().parents[2] / "scripts" / "gestur-wifi.py"
)
wifi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wifi)


class FakeDBus:
    ByteArray = bytes
    UInt32 = int
    Int32 = int
    Boolean = bool

    @staticmethod
    def Dictionary(value, **kwargs):
        return dict(value)

    @staticmethod
    def Array(value, **kwargs):
        return list(value)


class Connection:
    def __init__(self):
        self.updates = []

    def GetSecrets(self, setting):
        return {setting: {"psk": "previous-password"}}

    def Update(self, settings):
        self.updates.append(copy.deepcopy(settings))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class HelperTests(unittest.TestCase):
    def test_default_name_uses_permanent_mac_suffix(self):
        self.assertEqual(wifi.ssid_from_mac("DC:A6:32:01:ab:cd"), "GESTUR-ABCD")
        for value in ["bad", "00:00:00:00:00:00"]:
            with self.assertRaises(ValueError):
                wifi.ssid_from_mac(value)

    def test_validation(self):
        for data in [
            {},
            {"ssid": "x", "password": "short"},
            {"ssid": "x\n"},
            {"ssid": "x", "password": "á" * 8},
            {"ssid": "x", "exec": "rm"},
        ]:
            with self.assertRaises(ValueError):
                wifi.validate_request(data)
        wifi.validate_request({"ssid": "GESTUR-ABCD", "password": None})
        wifi.validate_request({"ssid": "GESTUR-ABCD", "password": "valid123"})

    def helper(self):
        helper = wifi.NetworkManager.__new__(wifi.NetworkManager)
        helper.dbus = FakeDBus
        connection = Connection()
        settings = {
            "connection": {"uuid": "fixed"},
            "802-11-wireless": {
                "mode": "ap",
                "ssid": b"Old",
                "security": "802-11-wireless-security",
            },
            "802-11-wireless-security": {"key-mgmt": "wpa-psk"},
        }
        helper.profile = lambda: (
            "device",
            "profile",
            connection,
            copy.deepcopy(settings),
        )
        helper.activate = lambda *args: None
        helper.status = lambda: {"ssid": "new", "secured": True, "active": True}
        return helper, connection

    def test_rename_preserves_password_and_remove_clears_security(self):
        helper, connection = self.helper()
        helper.apply({"ssid": "Renamed"})
        self.assertEqual(
            connection.updates[0]["802-11-wireless-security"]["psk"],
            "previous-password",
        )
        helper.apply({"ssid": "Open", "password": None})
        self.assertNotIn("802-11-wireless-security", connection.updates[1])
        self.assertNotIn("security", connection.updates[1]["802-11-wireless"])

    def test_new_password_and_failed_activation_roll_back(self):
        helper, connection = self.helper()
        calls = []

        def activate(*args):
            calls.append(args)
            if len(calls) == 1:
                raise RuntimeError("secret failure")

        helper.activate = activate
        with self.assertRaises(RuntimeError):
            helper.apply({"ssid": "New", "password": "new-password"})
        self.assertEqual(
            connection.updates[0]["802-11-wireless-security"]["psk"], "new-password"
        )
        self.assertEqual(
            connection.updates[1]["802-11-wireless-security"]["psk"],
            "previous-password",
        )
        self.assertEqual(len(calls), 2)

    def test_first_install_creates_open_mac_named_access_point(self):
        helper = wifi.NetworkManager.__new__(wifi.NetworkManager)
        helper.dbus = FakeDBus
        enabled, created, activated = [], [], []
        helper.interface = lambda *args: SimpleNamespace(
            Set=lambda *args: enabled.append(args)
        )
        helper.nm = SimpleNamespace(GetDeviceByIpIface=lambda interface: "wlan-device")

        def add(settings):
            created.append(copy.deepcopy(settings))
            return "new-profile"

        helper.settings = SimpleNamespace(ListConnections=lambda: [], AddConnection=add)
        helper.prop = lambda *args: "DC:A6:32:01:AB:CD"
        helper.active = lambda *args: False
        helper.activate = lambda *args: activated.append(args)
        helper.status = lambda: {
            "ssid": "GESTUR-ABCD",
            "secured": False,
            "active": True,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(wifi, "PROFILE", Path(directory) / "wifi-profile.json"),
        ):
            result = helper.bootstrap()
            self.assertTrue(wifi.PROFILE.exists())
            self.assertEqual(wifi.PROFILE.stat().st_mode & 0o777, 0o600)
        self.assertFalse(result["secured"])
        self.assertEqual(created[0]["802-11-wireless"]["ssid"], b"GESTUR-ABCD")
        self.assertNotIn("802-11-wireless-security", created[0])
        self.assertEqual(created[0]["ipv4"]["method"], "shared")
        self.assertEqual(activated, [("wlan-device", "new-profile")])
        self.assertTrue(enabled)

    def test_reinstall_reactivates_existing_profile_without_modifying_credentials(self):
        helper, connection = self.helper()
        helper.interface = lambda *args: SimpleNamespace(Set=lambda *args: None)
        helper.active = lambda *args: False
        activated = []
        helper.activate = lambda *args: activated.append(args)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(wifi, "PROFILE", Path(directory) / "wifi-profile.json"),
        ):
            wifi.PROFILE.write_text('{"uuid":"fixed","interface":"wlan0"}')
            helper.bootstrap()
        self.assertEqual(connection.updates, [])
        self.assertEqual(activated, [("device", "profile")])

    def activation_helper(self, clock, ready_at=0, connected_at=0):
        helper = wifi.NetworkManager.__new__(wifi.NetworkManager)
        calls = []
        active_path = "/"

        def activate(profile, device, specific):
            nonlocal active_path
            calls.append((clock.now, profile, device, specific))
            if clock.now < ready_at:
                raise RuntimeError("Device is not available")
            active_path = "active-ap"
            return active_path

        def prop(path, interface, name):
            if name in ("WirelessHardwareEnabled", "Managed"):
                return True
            if name == "WirelessEnabled":
                return clock.now >= ready_at / 2
            if name == "FirmwareMissing":
                return False
            if name == "ActiveConnection":
                return active_path
            if name == "State" and path == "device":
                return 30 if clock.now >= ready_at else 20
            if name == "State" and path == "active-ap":
                return 2 if clock.now >= connected_at else 1
            raise AssertionError((path, interface, name))

        helper.prop = prop
        helper.nm = SimpleNamespace(
            ActivateConnection=activate,
            DeactivateConnection=lambda active: self.fail("Unexpected disconnect"),
        )
        return helper, calls

    def test_activation_waits_for_radio_and_supplicant_then_for_active_ap(self):
        clock = FakeClock()
        helper, calls = self.activation_helper(clock, ready_at=0.5, connected_at=1)
        with (
            patch.object(wifi.time, "monotonic", clock.monotonic),
            patch.object(wifi.time, "sleep", clock.sleep),
        ):
            helper.activate("device", "profile")
        self.assertEqual(calls, [(0.5, "profile", "device", "/")])
        self.assertEqual(clock.now, 1)

    def test_unavailable_device_times_out_without_attempting_activation(self):
        clock = FakeClock()
        helper, calls = self.activation_helper(clock, ready_at=100)
        with (
            patch.object(wifi.time, "monotonic", clock.monotonic),
            patch.object(wifi.time, "sleep", clock.sleep),
        ):
            with self.assertRaisesRegex(RuntimeError, "readiness timed out"):
                helper.activate("device", "profile")
        self.assertEqual(calls, [])
        self.assertEqual(clock.now, 15)

    def test_permanent_device_problems_fail_without_wait_or_activation(self):
        for field, value in (
            ("WirelessHardwareEnabled", False),
            ("Managed", False),
            ("FirmwareMissing", True),
            ("State", 120),
        ):
            with self.subTest(field=field):
                clock = FakeClock()
                helper, calls = self.activation_helper(clock)
                original = helper.prop
                helper.prop = lambda path, interface, name: (
                    value if name == field else original(path, interface, name)
                )
                with (
                    patch.object(wifi.time, "monotonic", clock.monotonic),
                    patch.object(wifi.time, "sleep", clock.sleep),
                ):
                    with self.assertRaises(RuntimeError):
                        helper.activate("device", "profile")
                self.assertEqual(calls, [])
                self.assertEqual(clock.now, 0)

    def test_activation_error_is_not_retried(self):
        clock = FakeClock()
        helper, _ = self.activation_helper(clock)
        calls = []

        def fail(*args):
            calls.append(args)
            raise RuntimeError("Permanent activation error")

        helper.nm.ActivateConnection = fail
        with (
            patch.object(wifi.time, "monotonic", clock.monotonic),
            patch.object(wifi.time, "sleep", clock.sleep),
        ):
            with self.assertRaisesRegex(RuntimeError, "Permanent activation error"):
                helper.activate("device", "profile")
        self.assertEqual(len(calls), 1)
        self.assertEqual(clock.now, 0)

    def test_activation_never_succeeds_if_ap_stays_activating(self):
        clock = FakeClock()
        helper, calls = self.activation_helper(clock, connected_at=100)
        with (
            patch.object(wifi.time, "monotonic", clock.monotonic),
            patch.object(wifi.time, "sleep", clock.sleep),
        ):
            with self.assertRaisesRegex(RuntimeError, "activation timed out"):
                helper.activate("device", "profile")
        self.assertEqual(len(calls), 1)
        self.assertEqual(clock.now, 25)

    def test_reinstall_does_not_report_success_with_inactive_final_status(self):
        helper, _ = self.helper()
        helper.interface = lambda *args: SimpleNamespace(Set=lambda *args: None)
        helper.active = lambda *args: False
        helper.status = lambda: {"ssid": "Old", "secured": True, "active": False}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(wifi, "PROFILE", Path(directory) / "wifi-profile.json"),
        ):
            wifi.PROFILE.write_text('{"uuid":"fixed","interface":"wlan0"}')
            with self.assertRaisesRegex(RuntimeError, "not active"):
                helper.bootstrap()


if __name__ == "__main__":
    unittest.main()
