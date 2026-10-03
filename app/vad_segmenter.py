"""
VAD-сегментатор: главная логика "дождаться, когда человек закончит фразу".

Как это работает:
1. Каждый кадр аудио (30мс) прогоняется через webrtcvad — бинарный
   детектор "есть голос / нет голоса".
2. Пока идёт речь — копим кадры в буфер текущей фразы.
3. Как только подряд набирается END_OF_SPEECH_SILENCE_MS тишины —
   считаем, что фраза закончена, и отдаём накопленный буфер целиком
   на распознавание. Это и есть требование "не по три слова, а когда
   человек реально закончил говорить".
4. Короткие случайные всплески (вдох, щелчок) короче MIN_UTTERANCE_MS
   отбрасываются и не считаются фразой.
5. Если человек говорит очень долго без пауз (MAX_UTTERANCE_MS) —
   форсированно разбиваем, чтобы не копить гигантский буфер.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import webrtcvad

from app.config import (
    SAMPLE_RATE,
    FRAME_MS,
    VAD_AGGRESSIVENESS,
    END_OF_SPEECH_SILENCE_MS,
    MIN_UTTERANCE_MS,
    MAX_UTTERANCE_MS,
    PRE_SPEECH_PADDING_MS,
)

logger = logging.getLogger(__name__)


@dataclass
class Utterance:
    """Готовая, полностью произнесённая фраза, собранная из PCM-кадров."""

    pcm_int16: np.ndarray
    duration_ms: int
    sample_rate: int = SAMPLE_RATE


class VadSegmenter:
    """Принимает кадры из очереди захвата звука, выдаёт законченные фразы."""

    def __init__(self, on_utterance: Callable[[Utterance], None]):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.on_utterance = on_utterance

        self._frames_per_ms = SAMPLE_RATE / 1000
        self._frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)
        self._silence_frames_needed = max(1, int(END_OF_SPEECH_SILENCE_MS / FRAME_MS))
        self._min_speech_frames = max(1, int(MIN_UTTERANCE_MS / FRAME_MS))
        self._max_speech_frames = max(1, int(MAX_UTTERANCE_MS / FRAME_MS))
        self._pre_pad_frames = max(0, int(PRE_SPEECH_PADDING_MS / FRAME_MS))

        self._ring_buffer: list[np.ndarray] = []   # кольцевой буфер "до начала речи"
        self._speech_buffer: list[np.ndarray] = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_frame_count = 0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, frame_queue: "queue.Queue[np.ndarray]"):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(frame_queue,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self, frame_queue: "queue.Queue[np.ndarray]"):
        while not self._stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray):
        if len(frame) != self._frame_samples:
            # добиваем нулями / отрезаем до нужной длины, webrtcvad требует точный размер
            fixed = np.zeros(self._frame_samples, dtype=np.int16)
            n = min(len(frame), self._frame_samples)
            fixed[:n] = frame[:n]
            frame = fixed

        try:
            is_speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        except Exception:
            logger.exception("Ошибка VAD на кадре, пропускаем")
            return

        if not self._in_speech:
            # копим небольшой "запас" тишины перед речью, чтобы не отрезать её начало
            self._ring_buffer.append(frame)
            if len(self._ring_buffer) > self._pre_pad_frames:
                self._ring_buffer.pop(0)

            if is_speech:
                self._in_speech = True
                self._speech_buffer = list(self._ring_buffer)
                self._speech_buffer.append(frame)
                self._speech_frame_count = len(self._speech_buffer)
                self._silence_run = 0
            return

        # мы внутри фразы
        self._speech_buffer.append(frame)
        self._speech_frame_count += 1

        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        reached_silence_end = self._silence_run >= self._silence_frames_needed
        reached_max_len = self._speech_frame_count >= self._max_speech_frames

        if reached_silence_end or reached_max_len:
            self._finalize_utterance(forced=reached_max_len and not reached_silence_end)

    def _finalize_utterance(self, forced: bool = False):
        # Отрезаем хвост тишины, который использовался только для ДЕТЕКЦИИ конца фразы —
        # иначе длительность фразы искусственно "раздувается" на END_OF_SPEECH_SILENCE_MS
        # и короткие шумовые всплески ошибочно проходят порог MIN_UTTERANCE_MS.
        trimmed_buffer = self._speech_buffer
        if not forced and self._silence_run > 0:
            trimmed_buffer = self._speech_buffer[:-self._silence_run] or self._speech_buffer

        total_frames = len(trimmed_buffer)
        if total_frames >= self._min_speech_frames:
            pcm = np.concatenate(trimmed_buffer)
            duration_ms = int(total_frames * FRAME_MS)
            reason = "максимальная длина (речь без пауз)" if forced else "обнаружена пауза — конец фразы"
            logger.info("Фраза завершена (%s), длительность %d мс", reason, duration_ms)
            try:
                self.on_utterance(Utterance(pcm_int16=pcm, duration_ms=duration_ms))
            except Exception:
                logger.exception("Ошибка в обработчике готовой фразы")
        else:
            logger.debug("Отброшен короткий всплеск (%d мс) — не фраза", total_frames * FRAME_MS)

        self._in_speech = False
        self._speech_buffer = []
        self._speech_frame_count = 0
        self._silence_run = 0
        self._ring_buffer = []
