"""
Глобальные горячие клавиши (работают даже когда окно приложения не в фокусе —
например, когда пользователь находится в Zoom/Teams).
"""
from __future__ import annotations

import logging
from typing import Callable

from app.config import HOTKEY_ATTACH_SCREENSHOT, HOTKEY_TOGGLE_LISTEN, HOTKEY_ANALYZE_SCREENSHOT

logger = logging.getLogger(__name__)


class HotkeyManager:
    def __init__(self, on_screenshot: Callable[[], None], on_toggle_listen: Callable[[], None],
                 on_analyze_screenshot: Callable[[], None] | None = None):
        self.on_screenshot = on_screenshot
        self.on_toggle_listen = on_toggle_listen
        self.on_analyze_screenshot = on_analyze_screenshot
        self._handles = []

    def start(self):
        import keyboard  # требует прав/запуска с достаточными привилегиями на Windows

        if self._handles:
            return
        bindings = [(HOTKEY_ATTACH_SCREENSHOT, self.on_screenshot),
                    (HOTKEY_TOGGLE_LISTEN, self.on_toggle_listen)]
        if self.on_analyze_screenshot:
            bindings.append((HOTKEY_ANALYZE_SCREENSHOT, self.on_analyze_screenshot))
        try:
            for shortcut, callback in bindings:
                # Отпускание клавиш не передаёт Ctrl/Shift в появившийся диалог.
                options = {"trigger_on_release": True}
                if shortcut == HOTKEY_ANALYZE_SCREENSHOT:
                    options["suppress"] = True
                handle = keyboard.add_hotkey(shortcut, callback, **options)
                self._handles.append(handle)
        except Exception:
            self.stop()
            raise
        logger.info("Глобальные хоткеи зарегистрированы: %s", ", ".join(key for key, _ in bindings))

    def stop(self):
        if not self._handles:
            return
        import keyboard

        for handle in self._handles:
            try:
                keyboard.remove_hotkey(handle)
            except (KeyError, ValueError):
                pass
        self._handles.clear()
