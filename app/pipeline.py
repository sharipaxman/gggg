"""
Пайплайн: системный звук -> VAD (ждём конец фразы) -> Whisper (EN) -> перевод (RU)
         -> запись в локальную историю -> уведомление UI.

Распознавание/перевод выполняются в отдельном рабочем потоке, чтобы не блокировать
аудио-поток и UI.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from app.audio_capture import SystemAudioCapture
from app.vad_segmenter import VadSegmenter, Utterance
from app.speech_recognizer import SpeechRecognizer
from app.translator import Translator
from app.history_store import HistoryStore
from app.screenshot_manager import ScreenshotManager

logger = logging.getLogger(__name__)


@dataclass
class PhraseResult:
    source_text: str
    translated_text: str
    screenshot_path: Optional[str]
    entry_id: int


class LessonPipeline:
    def __init__(self, on_phrase: Callable[[PhraseResult], None], device_name: Optional[str] = None):
        self.on_phrase = on_phrase
        self.capture = SystemAudioCapture(device_name=device_name)
        self.segmenter = VadSegmenter(on_utterance=self._handle_utterance)
        self.recognizer = SpeechRecognizer()
        self.translator = Translator()
        self.history = HistoryStore()
        self.screenshots = ScreenshotManager()

        self._work_queue: "queue.Queue[Utterance]" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        self.capture.start()
        self.segmenter.start(self.capture.frame_queue)
        self._running = True
        logger.info("Пайплайн урока запущен")

    def stop(self):
        self.segmenter.stop()
        self.capture.stop()
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
        self._running = False
        logger.info("Пайплайн урока остановлен")

    def _handle_utterance(self, utterance: Utterance):
        # вызывается из потока VAD — просто кладём в очередь обработки
        self._work_queue.put(utterance)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                utterance = self._work_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process_utterance(utterance)

    def _process_utterance(self, utterance: Utterance):
        try:
            source_text = self.recognizer.transcribe_pcm16(utterance.pcm_int16, utterance.sample_rate)
            if not source_text.strip():
                logger.debug("Фраза распознана как пустая, пропускаем")
                return

            translated_text = self.translator.translate(source_text)
            screenshot_path = self.screenshots.pop_pending()

            entry_id = self.history.add_entry(
                source_text=source_text,
                translated_text=translated_text,
                screenshot_path=screenshot_path,
            )

            result = PhraseResult(
                source_text=source_text,
                translated_text=translated_text,
                screenshot_path=screenshot_path,
                entry_id=entry_id,
            )
            self.on_phrase(result)
        except Exception:
            logger.exception("Ошибка обработки фразы")

    def capture_screenshot_now(self) -> str:
        """Вызывается по хоткею: сделать снимок и привязать к следующей фразе."""
        return self.screenshots.capture()
