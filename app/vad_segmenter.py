"""
VAD-сегментатор на Silero VAD: главная логика "дождаться, когда человек
закончит фразу".

Почему Silero, а не webrtcvad:
- webrtcvad — заброшенная C-библиотека, которая требует компилятор и плохо
  ставится на Python 3.11+ (нет готовых колёс);
- Silero VAD — обычный pip-пакет на PyTorch, ставится без компиляторов,
  работает точнее (нейросеть вместо энергетического детектора), отлично
  отсекает шум, музыку и щелчки, при этом обрабатывает кадр 32 мс
  примерно за 1 мс на одном ядре CPU.

Как это работает:
1. Каждое окно аудио (ровно 512 сэмплов = 32 мс при 16 кГц — другие размеры
   Silero не принимает) переводится в float32-тензор и прогоняется через
   модель. На выходе — вероятность речи от 0.0 до 1.0 (а не бинарный ответ,
   как у webrtcvad).
2. Порог с гистерезисом: чтобы начать фразу, нужна уверенная речь
   (>= VAD_THRESHOLD); чтобы признать тишину, вероятность должна упасть
   ниже VAD_NEG_THRESHOLD. Это не даёт фразе рваться на тихих словах.
3. Пока идёт речь — копим кадры в буфер текущей фразы.
4. Как только подряд набирается END_OF_SPEECH_SILENCE_MS тишины —
   считаем, что фраза закончена, и отдаём накопленный буфер целиком
   на распознавание. Это и есть требование "не по три слова, а когда
   человек реально закончил говорить". Порог 1200 мс: паузы "подумать"
   внутри предложения (обычно до секунды) не должны разрывать фразу.
   Тихий хвост в конце (POST_SPEECH_PADDING_MS) сохраняем в фразе —
   чтобы не отрезать окончания последних слов.
5. Короткие случайные всплески (вдох, щелчок) короче MIN_UTTERANCE_MS
   отбрасываются и не считаются фразой.
6. Если человек говорит очень долго без пауз (MAX_UTTERANCE_MS) —
   форсированно разбиваем, чтобы не копить гигантский буфер.

Модель Silero рекуррентная (внутри LSTM), поэтому после каждой завершённой
фразы вызываем model.reset_states() — так состояние предыдущей фразы не
влияет на следующую.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from app.config import (
    SAMPLE_RATE,
    FRAME_MS,
    VAD_WINDOW_SAMPLES,
    VAD_THRESHOLD,
    VAD_NEG_THRESHOLD,
    VAD_USE_ONNX,
    TORCH_HUB_DIR,
    TORCH_NUM_THREADS,
    END_OF_SPEECH_SILENCE_MS,
    MIN_UTTERANCE_MS,
    MAX_UTTERANCE_MS,
    PRE_SPEECH_PADDING_MS,
    POST_SPEECH_PADDING_MS,
)

logger = logging.getLogger(__name__)

INT16_MAX = 32768.0
SILERO_HUB_REPO = "snakers4/silero-vad"


def load_silero_model(use_onnx: bool = VAD_USE_ONNX):
    """Загружает модель Silero VAD.

    Порядок попыток (от самого "офлайнового" к самому "онлайновому"):
    1. pip-пакет `silero-vad` — модель (~1 МБ) лежит прямо внутри пакета,
       интернет не нужен вообще;
    2. локальный кеш torch.hub в data/models/torch_hub — если модель уже
       скачивали раньше;
    3. загрузка с GitHub через torch.hub (нужен интернет один раз,
       дальше работает из кеша).
    """
    # Модель крошечная: несколько потоков только мешают и едят CPU.
    torch.set_num_threads(TORCH_NUM_THREADS)

    try:
        from silero_vad import load_silero_vad

        model = load_silero_vad(onnx=use_onnx)
        logger.info("Silero VAD загружен из pip-пакета silero-vad (onnx=%s)", use_onnx)
        return model
    except Exception as exc:  # пакета нет или он сломан — пробуем torch.hub
        logger.warning("Не удалось загрузить Silero VAD из пакета silero-vad (%s). Пробуем torch.hub.", exc)

    TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(TORCH_HUB_DIR))

    cached_repo = TORCH_HUB_DIR / "hub" / "snakers4_silero-vad_master"
    if cached_repo.exists():
        try:
            model, _utils = torch.hub.load(
                repo_or_dir=str(cached_repo),
                model="silero_vad",
                source="local",
                onnx=use_onnx,
            )
            logger.info("Silero VAD загружен из локального кеша torch.hub: %s", cached_repo)
            return model
        except Exception as exc:
            logger.warning("Локальный кеш torch.hub непригоден (%s), скачиваем модель заново.", exc)

    model, _utils = torch.hub.load(
        repo_or_dir=SILERO_HUB_REPO,
        model="silero_vad",
        force_reload=False,
        onnx=use_onnx,
        trust_repo=True,
    )
    logger.info("Silero VAD скачан через torch.hub (кеш: %s)", TORCH_HUB_DIR)
    return model


@dataclass
class Utterance:
    """Готовая, полностью произнесённая фраза, собранная из PCM-кадров."""

    pcm_int16: np.ndarray
    duration_ms: int
    sample_rate: int = SAMPLE_RATE


class VadSegmenter:
    """Принимает кадры из очереди захвата звука, выдаёт законченные фразы.

    :param on_utterance: колбэк, которому отдаём готовую фразу.
    :param model: уже загруженная модель Silero (нужно в основном для тестов);
                  если не передана — грузится лениво при старте потока.
    """

    def __init__(self, on_utterance: Callable[[Utterance], None], model=None):
        self.on_utterance = on_utterance
        self.model = model

        self._frames_per_ms = SAMPLE_RATE / 1000
        # Silero принимает строго 512 сэмплов на 16 кГц (256 на 8 кГц)
        self._frame_samples = VAD_WINDOW_SAMPLES
        self._silence_frames_needed = max(1, int(END_OF_SPEECH_SILENCE_MS / FRAME_MS))
        self._min_speech_frames = max(1, int(MIN_UTTERANCE_MS / FRAME_MS))
        self._max_speech_frames = max(1, int(MAX_UTTERANCE_MS / FRAME_MS))
        self._pre_pad_frames = max(0, int(PRE_SPEECH_PADDING_MS / FRAME_MS))
        self._post_pad_frames = max(0, int(POST_SPEECH_PADDING_MS / FRAME_MS))

        self._ring_buffer: list[np.ndarray] = []   # кольцевой буфер "до начала речи"
        self._speech_buffer: list[np.ndarray] = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_frame_count = 0
        # хвост сэмплов, не вместившийся в целое окно 512 — дорежем его следующим кадром
        self._tail = np.zeros((0,), dtype=np.int16)
        self._last_probability = 0.0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def last_probability(self) -> float:
        """Вероятность речи на последнем окне (0..1) — удобно для индикатора в UI."""
        return self._last_probability

    @property
    def is_speaking(self) -> bool:
        """Идёт ли сейчас фраза (речь началась, но пауза ещё не наступила)."""
        return self._in_speech

    def ensure_model(self):
        """Загружает модель, если она ещё не загружена (можно звать заранее)."""
        if self.model is None:
            self.model = load_silero_model()
        return self.model

    def start(self, frame_queue: "queue.Queue[np.ndarray]"):
        self._stop_event.clear()
        self._reset_state()
        self._thread = threading.Thread(target=self._run, args=(frame_queue,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._reset_model_states()

    def _run(self, frame_queue: "queue.Queue[np.ndarray]"):
        try:
            self.ensure_model()
        except Exception:
            logger.exception(
                "Не удалось загрузить модель Silero VAD. Установите пакет: pip install silero-vad torch"
            )
            return

        while not self._stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process_frame(frame)

    # ------------------------------------------------------------------
    # Обработка аудио
    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray):
        """Режет входящий кадр на окна ровно по 512 сэмплов и скармливает их модели."""
        if frame is None or len(frame) == 0:
            return

        if frame.dtype != np.int16:
            frame = frame.astype(np.int16)

        buf = np.concatenate([self._tail, frame]) if len(self._tail) else frame
        n_windows = len(buf) // self._frame_samples
        for i in range(n_windows):
            window = buf[i * self._frame_samples:(i + 1) * self._frame_samples]
            self._process_window(np.ascontiguousarray(window))
        self._tail = buf[n_windows * self._frame_samples:].copy()

    def _speech_probability(self, window: np.ndarray) -> float:
        """Прогон одного окна через Silero: int16 -> float32-тензор -> вероятность речи."""
        audio_f32 = window.astype(np.float32) / INT16_MAX
        tensor = torch.from_numpy(audio_f32)
        with torch.no_grad():
            output = self.model(tensor, SAMPLE_RATE)
        return float(output.item() if hasattr(output, "item") else output)

    def _process_window(self, window: np.ndarray):
        try:
            probability = self._speech_probability(window)
        except Exception:
            logger.exception("Ошибка Silero VAD на окне, пропускаем")
            return

        self._last_probability = probability
        # Гистерезис: войти в речь сложнее, чем остаться в ней —
        # так фраза не рвётся на тихих словах и коротких вдохах.
        threshold = VAD_NEG_THRESHOLD if self._in_speech else VAD_THRESHOLD
        is_speech = probability >= threshold

        if not self._in_speech:
            # копим небольшой "запас" тишины перед речью, чтобы не отрезать её начало
            self._ring_buffer.append(window)
            if len(self._ring_buffer) > self._pre_pad_frames:
                self._ring_buffer.pop(0)

            if is_speech:
                self._in_speech = True
                self._speech_buffer = list(self._ring_buffer)
                self._speech_frame_count = len(self._speech_buffer)
                self._silence_run = 0
            return

        # мы внутри фразы
        self._speech_buffer.append(window)
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
        # Сколько кадров в буфере — реальная речь (предзапись + сама речь),
        # без хвоста тишины, который копился только для ДЕТЕКЦИИ конца фразы.
        speech_frames = max(0, len(self._speech_buffer) - self._silence_run)

        if speech_frames >= self._min_speech_frames:
            # Хвост тишины отрезаем, но оставляем POST_SPEECH_PADDING_MS:
            # тихие окончания последних слов VAD помечает как тишину, и если
            # отрезать её целиком, Whisper теряет конец предложения.
            keep_tail = 0 if forced else min(self._silence_run, self._post_pad_frames)
            trimmed_buffer = self._speech_buffer[:speech_frames + keep_tail]

            total_frames = len(trimmed_buffer)
            pcm = np.concatenate(trimmed_buffer)
            duration_ms = int(total_frames * FRAME_MS)
            reason = "максимальная длина (речь без пауз)" if forced else "обнаружена пауза — конец фразы"
            logger.info("Фраза завершена (%s), длительность %d мс", reason, duration_ms)
            try:
                self.on_utterance(Utterance(pcm_int16=pcm, duration_ms=duration_ms))
            except Exception:
                logger.exception("Ошибка в обработчике готовой фразы")
        else:
            logger.debug("Отброшен короткий всплеск (%d мс) — не фраза", speech_frames * FRAME_MS)

        self._in_speech = False
        self._speech_buffer = []
        self._speech_frame_count = 0
        self._silence_run = 0
        self._ring_buffer = []
        # Silero рекуррентная: сбрасываем внутреннее состояние между фразами
        self._reset_model_states()

    def _reset_model_states(self):
        reset = getattr(self.model, "reset_states", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                logger.debug("Не удалось сбросить состояние Silero VAD", exc_info=True)

    def _reset_state(self):
        self._ring_buffer = []
        self._speech_buffer = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_frame_count = 0
        self._tail = np.zeros((0,), dtype=np.int16)
        self._last_probability = 0.0
        self._reset_model_states()
