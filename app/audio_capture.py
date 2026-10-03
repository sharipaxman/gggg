"""
Захват системного звука (loopback) на Windows через библиотеку `soundcard`.

Идея: берём не микрофон, а loopback-устройство — "запись того, что
воспроизводит динамик/наушники" (звук из звонка Zoom/Teams/Skype,
видео в браузере и т.п.). Это штатный WASAPI loopback-режим Windows,
никаких дополнительных виртуальных кабелей не требуется.

Поток работает в отдельном потоке и кладёт PCM int16 чанки в очередь,
откуда их забирает VAD-сегментатор (app/vad_segmenter.py).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

from app.config import SAMPLE_RATE, FRAME_MS

logger = logging.getLogger(__name__)

FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


class SystemAudioCapture:
    """Захватывает системный звук (loopback) и отдаёт кадры фиксированной длины."""

    def __init__(self, device_name: Optional[str] = None):
        self.device_name = device_name
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._leftover = np.zeros((0,), dtype=np.int16)

    @staticmethod
    def list_loopback_devices():
        """Возвращает список доступных loopback-микрофонов (динамиков) для выбора в UI."""
        import soundcard as sc  # импорт здесь — чтобы модуль не падал на платформах без него

        speakers = sc.all_speakers()
        return [s.name for s in speakers]

    def _get_loopback_mic(self):
        import soundcard as sc

        if self.device_name:
            return sc.get_microphone(self.device_name, include_loopback=True)
        default_speaker = sc.default_speaker()
        return sc.get_microphone(default_speaker.name, include_loopback=True)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Захват системного звука запущен (устройство: %s)", self.device_name or "по умолчанию")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Захват системного звука остановлен")

    def _run(self):
        try:
            mic = self._get_loopback_mic()
        except Exception:
            logger.exception("Не удалось открыть loopback-устройство. Проверьте, что приложение запущено на Windows.")
            return

        chunk_samples = max(FRAME_SAMPLES, 480)
        try:
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=chunk_samples) as recorder:
                while not self._stop_event.is_set():
                    data = recorder.record(numframes=chunk_samples)  # float32 [-1, 1], shape (N, 1)
                    mono = data[:, 0] if data.ndim > 1 else data
                    pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
                    self._push_frames(pcm16)
        except Exception:
            logger.exception("Ошибка во время записи системного звука")

    def _push_frames(self, pcm16: np.ndarray):
        """Нарезает входящий поток на кадры фиксированной длины FRAME_SAMPLES."""
        buf = np.concatenate([self._leftover, pcm16])
        n_full = len(buf) // FRAME_SAMPLES
        for i in range(n_full):
            frame = buf[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                logger.warning("Очередь аудио-кадров переполнена, кадр отброшен")
        self._leftover = buf[n_full * FRAME_SAMPLES:]
