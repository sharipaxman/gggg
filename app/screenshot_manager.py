"""
Снимок экрана по горячей клавише и привязка его к следующей/последней фразе.

Логика "прикрепить к следующей фразе": пользователь жмёт хоткей —
скриншот сохраняется на диск и помечается как "ожидающий привязки".
Как только придёт следующая распознанная+переведённая фраза — скриншот
прикрепляется к ней автоматически (а не к прошлой).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import SCREENSHOTS_DIR

logger = logging.getLogger(__name__)


class ScreenshotManager:
    def __init__(self):
        self._pending_path: Optional[str] = None
        self._lock = threading.Lock()

    def capture(self) -> str:
        """Делает снимок всего экрана и сохраняет его. Возвращает путь к файлу."""
        import mss
        import mss.tools

        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        path = SCREENSHOTS_DIR / filename

        with mss.mss() as sct:
            monitor = sct.monitors[0]  # 0 = все мониторы объединённо
            shot = sct.grab(monitor)
            mss.tools.to_png(shot.rgb, shot.size, output=str(path))

        with self._lock:
            self._pending_path = str(path)

        logger.info("Снимок экрана сохранён: %s (ожидает привязки к следующей фразе)", path)
        return str(path)

    def pop_pending(self) -> Optional[str]:
        """Забирает ожидающий скриншот (если есть) и очищает ожидание — вызывается при сохранении новой фразы."""
        with self._lock:
            path = self._pending_path
            self._pending_path = None
            return path

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending_path is not None
