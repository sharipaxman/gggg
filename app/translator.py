"""
Офлайн-перевод EN -> RU через argostranslate.
При первом запуске языковой пакет нужно установить один раз (см. scripts/setup_models.py).
"""
from __future__ import annotations

import logging
import threading

from app.config import TRANSLATE_FROM, TRANSLATE_TO

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self):
        self._translation = None
        self._lock = threading.Lock()

    def _ensure_translation(self):
        if self._translation is not None:
            return
        import argostranslate.translate as argos_translate

        installed_languages = argos_translate.get_installed_languages()
        from_lang = next((l for l in installed_languages if l.code == TRANSLATE_FROM), None)
        to_lang = next((l for l in installed_languages if l.code == TRANSLATE_TO), None)

        if not from_lang or not to_lang:
            raise RuntimeError(
                "Языковой пакет Argos Translate en->ru не установлен. "
                "Запустите scripts/setup_models.py для разовой установки офлайн-моделей."
            )

        self._translation = from_lang.get_translation(to_lang)
        logger.info("Переводчик Argos Translate (%s->%s) готов", TRANSLATE_FROM, TRANSLATE_TO)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        self._ensure_translation()
        with self._lock:
            return self._translation.translate(text)
