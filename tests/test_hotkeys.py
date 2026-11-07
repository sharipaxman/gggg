"""Регистрация хоткеев без keyboard, ОС и глобальных перехватчиков."""
import unittest
from unittest.mock import MagicMock, patch, call

from app.config import HOTKEY_ATTACH_SCREENSHOT, HOTKEY_TOGGLE_LISTEN, HOTKEY_ANALYZE_SCREENSHOT
from app.hotkey_manager import HotkeyManager


class HotkeyTests(unittest.TestCase):
    def test_all_shortcuts_registered_on_release_and_removed_by_handle(self):
        keyboard = MagicMock()
        keyboard.add_hotkey.side_effect = ["history", "listen", "vision"]
        callbacks = [MagicMock() for _ in range(3)]
        manager = HotkeyManager(*callbacks)
        with patch.dict("sys.modules", {"keyboard": keyboard}):
            manager.start()
            manager.start()
            self.assertEqual(keyboard.add_hotkey.call_args_list, [
                call(HOTKEY_ATTACH_SCREENSHOT, callbacks[0], trigger_on_release=True),
                call(HOTKEY_TOGGLE_LISTEN, callbacks[1], trigger_on_release=True),
                call(HOTKEY_ANALYZE_SCREENSHOT, callbacks[2], trigger_on_release=True, suppress=True),
            ])
            manager.stop()
            manager.stop()
            self.assertEqual(keyboard.remove_hotkey.call_args_list,
                             [call("history"), call("listen"), call("vision")])

    def test_partial_registration_is_rolled_back(self):
        keyboard = MagicMock()
        keyboard.add_hotkey.side_effect = ["first", RuntimeError("reserved")]
        manager = HotkeyManager(MagicMock(), MagicMock(), MagicMock())
        with patch.dict("sys.modules", {"keyboard": keyboard}):
            with self.assertRaisesRegex(RuntimeError, "reserved"):
                manager.start()
        keyboard.remove_hotkey.assert_called_once_with("first")
        self.assertEqual(manager._handles, [])

    def test_legacy_constructor_still_registers_two_shortcuts(self):
        keyboard = MagicMock()
        manager = HotkeyManager(MagicMock(), MagicMock())
        with patch.dict("sys.modules", {"keyboard": keyboard}):
            manager.start()
            self.assertEqual(keyboard.add_hotkey.call_count, 2)
            manager.stop()


if __name__ == "__main__":
    unittest.main()
