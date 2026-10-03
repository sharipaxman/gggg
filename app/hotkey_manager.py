"""
Глобальные горячие клавиши (работают даже когда окно приложения не в фокусе —
например, когда пользователь находится в Zoom/Teams).
"""
from __future__ import annotations

import logging
from typing import Callable

from app.config import HOTKEY_ATTACH_SCREENSHOT, HOTKEY_TOGGLE_LISTEN

logger = logging.getLogger(__name__)


class HotkeyManager:
    def __init__(self, on_screenshot: Callable[[], None], on_toggle_listen: Callable[[], None]):
        self.on_screenshot = on_screenshot
        self.on_toggle_listen = on_toggle_listen
        self._registered = False

    def start(self):
        import keyboard  # требует прав/запуска с достаточными привилегиями на Windows

        keyboard.add_hotkey(HOTKEY_ATTACH_SCREENSHOT, self.on_screenshot)
        keyboard.add_hotkey(HOTKEY_TOGGLE_LISTEN, self.on_toggle_listen)
        self._registered = True
        logger.info(
            "Глобальные хоткеи зарегистрированы: скриншот=%s, вкл/выкл прослушивание=%s",
            HOTKEY_ATTACH_SCREENSHOT, HOTKEY_TOGGLE_LISTEN,
        )

    def stop(self):
        if not self._registered:
            return
        import keyboard

        try:
            keyboard.remove_hotkey(HOTKEY_ATTACH_SCREENSHOT)
            keyboard.remove_hotkey(HOTKEY_TOGGLE_LISTEN)
        except KeyError:
            pass
        self._registered = False
