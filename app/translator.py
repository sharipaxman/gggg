"""Офлайн-перевод EN ↔ RU через два пакета Argos Translate."""
from __future__ import annotations

import logging
import threading

from app.config import TRANSLATE_FROM, TRANSLATE_TO

logger = logging.getLogger(__name__)


def target_language(source_language: str) -> str:
    if source_language not in ("en", "ru"):
        raise ValueError(f"Язык '{source_language}' не поддерживается: ожидается EN или RU")
    return "ru" if source_language == "en" else "en"


class Translator:
    def __init__(self):
        self._translations = {}
        self._lock = threading.Lock()

    def _ensure_translation(self, source: str, target: str):
        key = (source, target)
        if key not in self._translations:
            import argostranslate.translate as argos_translate

            languages = argos_translate.get_installed_languages()
            from_lang = next((lang for lang in languages if lang.code == source), None)
            to_lang = next((lang for lang in languages if lang.code == target), None)
            translation = from_lang.get_translation(to_lang) if from_lang and to_lang else None
            if translation is None:
                raise RuntimeError(
                    f"Языковой пакет Argos Translate {source}→{target} не установлен. "
                    "Запустите python scripts/setup_models.py."
                )
            self._translations[key] = translation
            logger.info("Переводчик Argos Translate (%s→%s) готов", source, target)
        return self._translations[key]

    def translate(self, text: str, source: str = TRANSLATE_FROM, target: str = TRANSLATE_TO) -> str:
        if not text.strip():
            return ""
        if target != target_language(source):
            raise ValueError("Поддерживаются только направления EN→RU и RU→EN")
        with self._lock:
            return self._ensure_translation(source, target).translate(text)
