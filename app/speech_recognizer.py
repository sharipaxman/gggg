"""
Распознавание речи с помощью faster-whisper — полностью локально, без интернета
(после того как модель один раз скачана/положена в data/models).
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from app.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Обёртка над faster-whisper. Модель грузится один раз и переиспользуется."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info("Загрузка модели Whisper '%s' (device=%s, compute_type=%s)...",
                    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=str(MODELS_DIR),
        )
        logger.info("Модель Whisper загружена")

    def transcribe_pcm16(self, pcm_int16: np.ndarray, sample_rate: int) -> str:
        """Принимает PCM int16 моно, возвращает распознанный английский текст."""
        self._ensure_model()
        audio_f32 = pcm_int16.astype(np.float32) / 32768.0

        with self._lock:
            segments, _info = self._model.transcribe(
                audio_f32,
                language=WHISPER_LANGUAGE,
                task="transcribe",
                vad_filter=False,   # у нас уже свой VAD с детекцией конца фразы
                beam_size=5,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
