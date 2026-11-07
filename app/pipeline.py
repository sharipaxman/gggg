"""
Пайплайн: системный звук -> VAD (ждём конец фразы) -> Whisper (EN/RU) -> перевод (RU/EN)
         -> запись в локальную историю -> уведомление UI.

Распознавание/перевод выполняются в отдельном рабочем потоке, чтобы не блокировать
аудио-поток и UI.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, Optional, TYPE_CHECKING

from app.assistant import Assistance, ConversationAssistant, is_explicit_reply_request
from app.config import ASSISTANT_CONTEXT_ENTRIES
from app.assistant_workspace import AssistantWorkspace, ContextChat
from app.history_store import HistoryStore
from app.translator import Translator, target_language

if TYPE_CHECKING:
    from app.vad_segmenter import Utterance

logger = logging.getLogger(__name__)


@dataclass
class PhraseResult:
    source_text: str
    translated_text: str
    screenshot_path: Optional[str]
    entry_id: int
    source_language: str = "en"
    target_language: str = "ru"
    assistance: Assistance = field(default_factory=Assistance)
    assistance_status: str = "ИИ-подсказки выключены"
    memory_generation: int = 0
    memory_context: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class LessonPipeline:
    def __init__(self, on_phrase: Callable[[PhraseResult], None], device_name: Optional[str] = None,
                 on_error: Optional[Callable[[str], None]] = None):
        from app.audio_capture import SystemAudioCapture
        from app.vad_segmenter import VadSegmenter
        from app.speech_recognizer import SpeechRecognizer
        from app.screenshot_manager import ScreenshotManager

        self.on_error = on_error
        self.on_phrase = on_phrase
        self.capture = SystemAudioCapture(device_name=device_name)
        self.segmenter = VadSegmenter(on_utterance=self._handle_utterance)
        self.recognizer = SpeechRecognizer()
        self.translator = Translator()
        self.history = HistoryStore()
        self.assistant = ConversationAssistant(translator=self.translator)
        self.workspace = AssistantWorkspace()
        self.chat = ContextChat(self.assistant, self.workspace)
        self._assistant_enabled = threading.Event()
        self.screenshots = ScreenshotManager()

        self._work_queue: "queue.Queue[Utterance]" = queue.Queue()
        # Отдельная очередь ИИ: медленная LLM не задерживает распознавание/перевод.
        # Один ожидающий запрос; явные вопросы/просьбы приоритетнее утверждений.
        self._assistant_queue = queue.Queue(maxsize=1)
        self._assistant_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._prepared = False
        self._prepare_lock = threading.Lock()

    def set_assistant_enabled(self, enabled: bool):
        if enabled:
            self._assistant_enabled.set()
        else:
            self._assistant_enabled.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_prepared(self) -> bool:
        return self._prepared

    def prepare_models(self, on_progress: Optional[Callable[[str], None]] = None) -> str:
        """Вызывать из фонового потока UI. Звук пока не захватывается.

        Базовые модели обязательны. Недоступная Ollama не запрещает перевод;
        предупреждение возвращается отдельно. Повторный вызов может прогреть ИИ,
        включённый уже после подготовки переводчика.
        """
        def stage(label, action):
            if on_progress:
                on_progress(label)
            started = time.monotonic()
            action()
            logger.info("Подготовка %s: %.2fs", label, time.monotonic() - started)

        with self._prepare_lock:
            if not self._prepared:
                stage("Silero VAD", self.segmenter.ensure_model)
                stage("Whisper", self.recognizer.warm_up)
                stage("Переводчик EN ↔ RU", self.translator.warm_up)
                self._prepared = True
            warning = ""
            if self._assistant_enabled.is_set():
                try:
                    stage("Локальная Ollama", self.assistant.warm_up)
                except Exception as exc:
                    logger.warning("Перевод готов, но прогрев ИИ не удался: %s", exc)
                    warning = str(exc)
            return warning

    def start(self):
        if self._running:
            return
        if any(thread and thread.is_alive() for thread in (self._worker_thread, self._assistant_thread)):
            raise RuntimeError("Предыдущая обработка ещё завершается. Повторите запуск через несколько секунд.")
        # Защитный барьер для запусков вне UI: модели должны быть готовы до записи.
        if not self._prepared:
            self.prepare_models()
        self._stop_event.clear()
        try:
            self.capture.start()
            self.segmenter.start(self.capture.frame_queue)
        except Exception:
            self.segmenter.stop()
            self.capture.stop()
            self._stop_event.set()
            raise
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._assistant_thread = threading.Thread(target=self._assistant_loop, daemon=True)
        self._assistant_thread.start()
        self._worker_thread.start()

        self._running = True
        logger.info("Пайплайн урока запущен")

    def stop(self):
        self.segmenter.stop()
        self.capture.stop()
        self._stop_event.set()
        for thread in (self._worker_thread, self._assistant_thread):
            if thread:
                thread.join(timeout=2)
        self._running = False
        for pending in (self._work_queue, self._assistant_queue):
            while True:
                try:
                    pending.get_nowait()
                except queue.Empty:
                    break
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
            started = time.monotonic()
            recognition = self.recognizer.recognize_pcm16(utterance.pcm_int16, utterance.sample_rate)
            source_text = recognition.text
            if not source_text.strip():
                logger.debug("Фраза распознана как пустая, пропускаем")
                return

            recognized_at = time.monotonic()
            source_language = recognition.language
            target = target_language(source_language)
            translated_text = self.translator.translate(source_text, source_language, target)
            translated_at = time.monotonic()
            logger.info("Обработка фразы: ASR=%.2fs перевод=%.2fs всего=%.2fs (без очереди/ИИ)",
                        recognized_at - started, translated_at - recognized_at, translated_at - started)
            use_assistant = self._assistant_enabled.is_set()
            # Контекст до текущей фразы, в хронологическом порядке.
            context = list(reversed(self.history.get_recent(ASSISTANT_CONTEXT_ENTRIES))) if use_assistant else []
            generation, memory_context = self.workspace.snapshot()
            self.workspace.remember("heard", source_text)
            screenshot_path = self.screenshots.pop_pending()

            entry_id = self.history.add_entry(
                source_text=source_text,
                translated_text=translated_text,
                screenshot_path=screenshot_path,
                source_language=source_language,
                target_language=target,
            )

            result = PhraseResult(
                source_text=source_text,
                translated_text=translated_text,
                screenshot_path=screenshot_path,
                entry_id=entry_id,
                memory_generation=generation, memory_context=memory_context,
                source_language=source_language,
                target_language=target,
                assistance_status="ИИ анализирует контекст…" if use_assistant else "ИИ-подсказки выключены",
            )
            # Перевод отображается сразу, до медленного запроса к LLM.
            self.on_phrase(result)
            if use_assistant and not self._stop_event.is_set():
                try:
                    self._assistant_queue.put_nowait((result, context))
                except queue.Full:
                    try:
                        stale, stale_context = self._assistant_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        stale_needs_reply = (is_explicit_reply_request(stale.source_text)
                                             or "?" in stale.source_text)
                        new_needs_reply = (is_explicit_reply_request(result.source_text)
                                           or "?" in result.source_text)
                        if (not self.workspace.preferences().always_reply
                                and stale_needs_reply and not new_needs_reply):
                            self._assistant_queue.put_nowait((stale, stale_context))
                            self.on_phrase(replace(result, assistance_status="Пропущено: ожидается ответ на вопрос"))
                            return
                        self.on_phrase(replace(stale, assistance_status="Пропущено: появилась более новая фраза"))
                    self._assistant_queue.put_nowait((result, context))
        except Exception as exc:
            logger.exception("Ошибка обработки фразы")
            if self.on_error:
                self.on_error(str(exc))

    def _assistant_loop(self):
        while not self._stop_event.is_set():
            try:
                result, context = self._assistant_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process_assistance(result, context)

    def _process_assistance(self, result: PhraseResult, context: list):
        if self._stop_event.is_set():
            return
        if not self._assistant_enabled.is_set():
            self.on_phrase(replace(result, assistance_status="ИИ-подсказки выключены"))
            return
        try:
            if result.memory_generation != self.workspace.snapshot()[0]:
                self.on_phrase(replace(result, assistance_status="Пропущено: память очищена"))
                return
            preferences = self.workspace.preferences()
            if preferences.always_reply:
                reply = self.assistant.reply(
                    result.source_text, result.memory_context, preferences.prompt,
                )
                assistance = Assistance.from_reply(reply)
                self.workspace.remember("suggestion", reply.text, result.memory_generation, reply=reply)
                status = "Ответ по вашему промпту"
            else:
                assistance = self.assistant.suggest(
                    result.source_text, result.translated_text, result.source_language, context,
                    custom_prompt=preferences.prompt,
                )
                if assistance.suggestions:
                    self.workspace.remember("suggestion", "\n".join(
                        f"EN: {answer.en}\nRU: {answer.ru}" for answer in assistance.suggestions
                    ), result.memory_generation)
                status = ("Возможные ответы — примеры, проверьте перед использованием"
                          if assistance.is_question else "Вопрос не обнаружен")
            if result.memory_generation != self.workspace.snapshot()[0]:
                self.on_phrase(replace(result, assistance_status="Пропущено: память очищена"))
                return
            self.history.update_assistance(result.entry_id, assistance)
        except Exception as exc:
            logger.warning("ИИ-подсказки недоступны: %s", exc)
            assistance = Assistance()
            status = f"Подсказки недоступны: {exc}"
        # Новый объект: Qt мог ещё не обработать предыдущий сигнал.
        if not self._stop_event.is_set():
            self.on_phrase(replace(result, assistance=assistance, assistance_status=status))

    def close(self):
        self.chat.close()
        self.stop()

    def capture_screenshot_now(self) -> str:
        """Вызывается по хоткею: сделать снимок и привязать к следующей фразе."""
        return self.screenshots.capture()
