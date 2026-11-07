"""Локальный адаптер Stanza только для зависимостей Argos.

Argos 1.9 вызывает stanza.Pipeline на каждый абзац, новые версии делают это
через sbd.StanzaSentencizer. В обоих случаях запрещаем автоматическое скачивание
resources/models и переиспользуем pipeline. Сам модуль stanza и другие его
потребители не патчатся. Отсутствующий локальный ресурс — ошибка подготовки,
а не повод обращаться к серверу. Алгоритм разбиения предложений не меняется.
"""
from __future__ import annotations

import threading


class LocalStanza:
    def __init__(self, module):
        self._module = module
        self._pipelines = {}
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._module, name)

    def Pipeline(self, *args, **kwargs):
        kwargs["download_method"] = None  # официальный офлайн-режим Stanza
        key = repr((args, sorted(kwargs.items())))
        with self._lock:
            if key not in self._pipelines:
                self._pipelines[key] = self._module.Pipeline(*args, **kwargs)
            return self._pipelines[key]


def configure_local_stanza(argos_translate):
    # Старый Argos использует translate.stanza, новый — sbd.stanza.
    for dependency in (argos_translate, getattr(argos_translate, "sbd", None)):
        original = getattr(dependency, "stanza", None)
        if original is not None and not isinstance(original, LocalStanza):
            dependency.stanza = LocalStanza(original)
