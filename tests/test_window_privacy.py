"""Контракт Windows API проверяется заглушками, реальный экран не захватывается."""
import ctypes
import unittest
from ctypes import wintypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import window_privacy as privacy


class CaptureProtectionTests(unittest.TestCase):
    def test_non_windows_does_not_load_native_library(self):
        with patch.object(privacy.sys, "platform", "linux"), patch.object(privacy, "_load_user32") as load:
            result = privacy.set_capture_exclusion(123, True)
        self.assertFalse(result.applied)
        self.assertFalse(result.enabled)
        self.assertIn("Windows", result.message)
        load.assert_not_called()

    def test_windows_version_boundary(self):
        for major, build, supported in ((6, 19041, False), (10, 18363, False),
                                        (10, 19041, True), (10, 22621, True)):
            with self.subTest(major=major, build=build), patch.object(privacy.sys, "platform", "win32"), \
                 patch.object(privacy.sys, "getwindowsversion", create=True,
                              return_value=SimpleNamespace(major=major, build=build)):
                self.assertEqual(privacy.capture_protection_support()[0], supported)

    def make_api(self):
        api, state = MagicMock(), {"affinity": privacy.WDA_NONE}

        def write(hwnd, affinity):
            state["affinity"] = affinity
            return 1

        def read(hwnd, output):
            output._obj.value = state["affinity"]
            return 1

        api.SetWindowDisplayAffinity.side_effect = write
        api.GetWindowDisplayAffinity.side_effect = read
        return api, state

    def test_enable_and_disable_round_trip_use_exact_flags_and_full_hwnd(self):
        api, _ = self.make_api()
        hwnd = 0x12345678ABCDEF
        with patch.object(privacy, "capture_protection_support", return_value=(True, "")), \
             patch.object(privacy, "_load_user32", return_value=api):
            enabled = privacy.set_capture_exclusion(hwnd, True)
            disabled = privacy.set_capture_exclusion(hwnd, False)
        self.assertTrue(enabled.applied)
        self.assertTrue(enabled.enabled)
        self.assertTrue(disabled.applied)
        self.assertFalse(disabled.enabled)
        api.SetWindowDisplayAffinity.assert_any_call(hwnd, 0x11)
        api.SetWindowDisplayAffinity.assert_any_call(hwnd, 0)

    def test_prototypes_are_pointer_sized(self):
        api = MagicMock()
        with patch.object(privacy.ctypes, "WinDLL", create=True, return_value=api) as load:
            self.assertIs(privacy._load_user32(), api)
        load.assert_called_once_with("user32", use_last_error=True)
        self.assertEqual(api.SetWindowDisplayAffinity.argtypes, [wintypes.HWND, wintypes.DWORD])
        self.assertEqual(api.GetWindowDisplayAffinity.argtypes, [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)])
        self.assertEqual(api.SetWindowDisplayAffinity.restype, wintypes.BOOL)

    def test_missing_hwnd_is_not_sent_to_api(self):
        with patch.object(privacy, "capture_protection_support", return_value=(True, "")), \
             patch.object(privacy, "_load_user32") as load:
            for hwnd in (0, -1):
                result = privacy.set_capture_exclusion(hwnd, True)
                self.assertFalse(result.applied)
                self.assertIsNone(result.enabled)
        load.assert_not_called()

    def test_failed_disable_preserves_known_protected_state(self):
        api, state = self.make_api()
        state["affinity"] = privacy.WDA_EXCLUDEFROMCAPTURE
        api.SetWindowDisplayAffinity.side_effect = None
        api.SetWindowDisplayAffinity.return_value = 0
        with patch.object(privacy, "capture_protection_support", return_value=(True, "")), \
             patch.object(privacy, "_load_user32", return_value=api), \
             patch.object(privacy, "_last_error", return_value=5):
            result = privacy.set_capture_exclusion(123, False)
        self.assertFalse(result.applied)
        self.assertTrue(result.enabled)
        self.assertIn("5", result.message)

    def test_unknown_state_is_not_claimed_to_be_protected(self):
        api, _ = self.make_api()
        api.GetWindowDisplayAffinity.side_effect = None
        api.GetWindowDisplayAffinity.return_value = 0
        with patch.object(privacy, "capture_protection_support", return_value=(True, "")), \
             patch.object(privacy, "_load_user32", return_value=api):
            result = privacy.set_capture_exclusion(123, True)
        self.assertFalse(result.applied)
        self.assertIsNone(result.enabled)
        self.assertIn("Не полагайтесь", result.message)

    def test_black_window_affinity_is_not_treated_as_success(self):
        api, state = self.make_api()
        api.SetWindowDisplayAffinity.side_effect = lambda hwnd, affinity: 1
        state["affinity"] = 1  # WDA_MONITOR != WDA_EXCLUDEFROMCAPTURE
        with patch.object(privacy, "capture_protection_support", return_value=(True, "")), \
             patch.object(privacy, "_load_user32", return_value=api):
            result = privacy.set_capture_exclusion(123, True)
        self.assertFalse(result.applied)
        self.assertFalse(result.enabled)

    def test_api_load_error_is_reported_instead_of_crashing_ui(self):
        with patch.object(privacy, "capture_protection_support", return_value=(True, "")), \
             patch.object(privacy, "_load_user32", side_effect=OSError("API missing")):
            result = privacy.set_capture_exclusion(123, True)
        self.assertFalse(result.applied)
        self.assertIn("API missing", result.message)


if __name__ == "__main__":
    unittest.main()
