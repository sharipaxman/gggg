"""Проверка двух направлений, миграции истории и фоновых подсказок (без ML/аудио)."""
import queue
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.assistant import Assistance, AnswerSuggestion, AssistantReply, ReplyVariant
from app.history_store import HistoryStore
from app.assistant_workspace import AssistantWorkspace
from app.pipeline import LessonPipeline
from app.translator import Translator, target_language
from scripts.setup_models import setup_argos_translate


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.db"

    def test_migrate_existing_history_and_reopen(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE phrases (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, "
                         "source_text TEXT NOT NULL, translated_text TEXT NOT NULL, screenshot_path TEXT)")
            conn.execute("INSERT INTO phrases VALUES (1, '2026-01-01', 'Hello', 'Привет', 'screen.png')")
        store = HistoryStore(self.path)
        old = store.get_recent()[0]
        self.assertEqual((old.source_language, old.target_language), ("en", "ru"))
        self.assertEqual(old.screenshot_path, "screen.png")
        self.assertEqual(old.assistance, Assistance())
        entry_id = store.add_entry("Почему?", "Why?", source_language="ru", target_language="en")
        answer = Assistance(True, [AnswerSuggestion("Because…", "Потому что…")])
        store.update_assistance(entry_id, answer)
        entries = HistoryStore(self.path).get_recent()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].assistance, answer)
        self.assertEqual(entries[0].source_language, "ru")
        self.assertIn("RU: Почему?", store.get_context_text())
        self.assertIn("EN: Why?", store.get_context_text())

    def test_history_limit_preserved(self):
        store = HistoryStore(self.path, limit=2)
        for i in range(5):
            store.add_entry(str(i), str(i))
        self.assertEqual([entry.source_text for entry in store.get_recent()], ["4", "3"])


class TranslatorTests(unittest.TestCase):
    def test_language_direction(self):
        self.assertEqual(target_language("en"), "ru")
        self.assertEqual(target_language("ru"), "en")
        with self.assertRaises(ValueError):
            target_language("de")

    def test_direction_specific_cache(self):
        translator = Translator()
        en_ru, ru_en = MagicMock(), MagicMock()
        en_ru.translate.return_value = "Привет"
        ru_en.translate.return_value = "Hello"
        translator._translations = {("en", "ru"): en_ru, ("ru", "en"): ru_en}
        self.assertEqual(translator.translate("Hello"), "Привет")
        self.assertEqual(translator.translate("Привет", "ru", "en"), "Hello")
        self.assertEqual(translator.translate(""), "")
        en_ru.translate.assert_called_once_with("Hello")
        ru_en.translate.assert_called_once_with("Привет")
        with self.assertRaises(ValueError):
            translator.translate("Привет", "ru", "ru")

    def test_setup_installs_only_missing_direction(self):
        package = MagicMock()
        package.get_installed_packages.return_value = [SimpleNamespace(from_code="en", to_code="ru")]
        reverse = MagicMock(from_code="ru", to_code="en")
        package.get_available_packages.return_value = [reverse]
        with patch.dict("sys.modules", {"argostranslate": MagicMock(package=package),
                                         "argostranslate.package": package}):
            setup_argos_translate()
        reverse.download.assert_called_once()
        package.install_from_path.assert_called_once_with(reverse.download.return_value)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Не создаём реальные аудиоустройства/VAD/Whisper.
        self.pipeline = LessonPipeline.__new__(LessonPipeline)
        self.pipeline.history = HistoryStore(Path(self.temp.name) / "history.db")
        self.pipeline.workspace = AssistantWorkspace(Path(self.temp.name) / "history.db")
        self.pipeline.workspace.save_preferences("", False)  # legacy question-only tests
        self.pipeline.recognizer = MagicMock()
        self.pipeline.translator = MagicMock()
        self.pipeline.screenshots = MagicMock()
        self.pipeline.screenshots.pop_pending.return_value = "screen.png"
        self.pipeline.assistant = MagicMock()
        self.pipeline.assistant.suggest.return_value = Assistance(True, [AnswerSuggestion("Yes.", "Да.")])
        self.pipeline._assistant_enabled = threading.Event()
        self.pipeline._assistant_queue = queue.Queue(maxsize=1)
        self.pipeline._stop_event = threading.Event()
        self.pipeline._prepared = True
        self.pipeline._prepare_lock = threading.Lock()
        self.results = []
        self.pipeline.on_phrase = self.results.append
        self.pipeline.on_error = MagicMock()
        self.utterance = SimpleNamespace(pcm_int16=object(), sample_rate=16000)

    def recognize(self, language="en", text="Do you like it?"):
        self.pipeline.recognizer.recognize_pcm16.return_value = SimpleNamespace(text=text, language=language)
        self.pipeline.translator.translate.return_value = "Тебе нравится?" if language == "en" else "Why?"
        self.pipeline._process_utterance(self.utterance)

    def finish_assistance(self):
        self.pipeline._process_assistance(*self.pipeline._assistant_queue.get_nowait())

    def test_always_reply_processes_statement_with_custom_prompt(self):
        self.pipeline.workspace.save_preferences("Отвечай одним предложением на русском", True)
        self.pipeline.workspace.remember("heard", "We are discussing holidays.")
        self.pipeline.assistant.reply.return_value = AssistantReply("Речь о путешествиях.", "We are discussing travel.", "ru", "en")
        self.pipeline.set_assistant_enabled(True)
        self.recognize(text="I went to Paris.")
        self.finish_assistance()
        self.pipeline.assistant.reply.assert_called_once_with(
            "I went to Paris.", [{"role": "heard", "text": "We are discussing holidays."}],
            "Отвечай одним предложением на русском",
        )
        self.pipeline.assistant.suggest.assert_not_called()
        self.assertEqual(self.results[-1].assistance.response, "Речь о путешествиях.")
        self.assertEqual(self.results[-1].assistance.response_translation, "We are discussing travel.")
        self.assertEqual(self.pipeline.history.get_recent()[0].assistance.response_translation, "We are discussing travel.")
        self.assertEqual(self.pipeline.history.get_recent()[0].assistance.response, "Речь о путешествиях.")

    def test_auto_reply_keeps_both_variants_in_phrase_history(self):
        variants = [ReplyVariant("short", "Short", "Коротко", "en", "ru"),
                    ReplyVariant("long", "A longer answer", "Длинный ответ", "en", "ru")]
        self.pipeline.workspace.save_preferences("Мой уровень A2, дай два варианта", True)
        self.pipeline.assistant.reply.return_value = AssistantReply("Both", "Оба", "en", "ru", variants=variants)
        self.pipeline.set_assistant_enabled(True)
        self.recognize()
        self.finish_assistance()
        self.assertEqual(self.results[-1].assistance.variants, variants)
        self.assertEqual(self.pipeline.history.get_recent()[0].assistance.variants, variants)

    def test_clear_during_generation_discards_stale_answer(self):
        self.pipeline.workspace.save_preferences("Reply", True)
        self.pipeline.set_assistant_enabled(True)
        self.recognize(text="A statement.")

        def answer_and_clear(*args, **kwargs):
            self.pipeline.workspace.clear_memory()
            return AssistantReply("Stale answer")

        self.pipeline.assistant.reply.side_effect = answer_and_clear
        self.finish_assistance()
        self.assertIn("память очищена", self.results[-1].assistance_status)
        self.assertEqual(self.pipeline.workspace.recent(), [])
        self.assertEqual(self.pipeline.history.get_recent()[0].assistance.response, "")

    def test_context_accumulates_even_if_auto_ai_disabled(self):
        self.recognize(text="My name is Alex.")
        _, context = self.pipeline.workspace.snapshot()
        self.assertEqual(context, [{"role": "heard", "text": "My name is Alex."}])
        self.pipeline.assistant.reply.assert_not_called()

    def test_cleared_memory_does_not_send_old_pending_context(self):
        self.pipeline.set_assistant_enabled(True)
        self.recognize()
        self.pipeline.workspace.clear_memory()
        self.finish_assistance()
        self.pipeline.assistant.suggest.assert_not_called()
        self.assertIn("память очищена", self.results[-1].assistance_status)

    def test_both_directions_without_ai(self):
        for language, target in (("en", "ru"), ("ru", "en")):
            self.recognize(language)
            self.pipeline.translator.translate.assert_called_with("Do you like it?", language, target)
            self.assertEqual(self.results[-1].target_language, target)
        self.pipeline.assistant.suggest.assert_not_called()
        self.assertEqual(len(self.results), 2)

    def test_translation_emitted_before_ai_with_chronological_context(self):
        self.pipeline.history.add_entry("First", "Первое")
        self.pipeline.history.add_entry("Second", "Второе")
        self.pipeline.set_assistant_enabled(True)
        self.recognize()
        self.assertEqual(len(self.results), 1)
        self.pipeline.assistant.suggest.assert_not_called()
        self.assertIn("анализирует", self.results[0].assistance_status)
        self.finish_assistance()
        self.assertEqual(len(self.results), 2)
        args = self.pipeline.assistant.suggest.call_args.args
        self.assertEqual([e.source_text for e in args[3]], ["First", "Second"])
        self.assertEqual(self.results[0].entry_id, self.results[1].entry_id)
        self.assertEqual(self.results[1].assistance.suggestions[0].en, "Yes.")
        self.assertEqual(self.pipeline.history.get_recent()[0].assistance, self.results[1].assistance)
        self.assertEqual(len(self.pipeline.history.get_recent()), 3)
        self.assertEqual(self.pipeline.history.get_recent()[0].screenshot_path, "screen.png")

    def test_ai_failure_does_not_lose_translation(self):
        self.pipeline.set_assistant_enabled(True)
        self.pipeline.assistant.suggest.side_effect = RuntimeError("Ollama недоступна")
        self.recognize()
        with self.assertLogs("app.pipeline", level="WARNING"):
            self.finish_assistance()
        self.assertEqual(self.results[1].translated_text, "Тебе нравится?")
        self.assertIn("недоступна", self.results[1].assistance_status)
        self.assertEqual(len(self.pipeline.history.get_recent()), 1)
        self.pipeline.on_error.assert_not_called()

    def test_pending_queue_keeps_latest_utterance(self):
        self.pipeline.set_assistant_enabled(True)
        self.recognize(text="Old question?")
        self.recognize(text="New question?")
        self.assertEqual(self.pipeline._assistant_queue.qsize(), 1)
        pending, _ = self.pipeline._assistant_queue.get_nowait()
        self.assertEqual(pending.source_text, "New question?")
        self.assertIn("Пропущено", self.results[-1].assistance_status)

    def test_statement_cannot_evict_pending_reply_request(self):
        self.pipeline.set_assistant_enabled(True)
        self.recognize(text="Tell me the origin of your name.")
        self.recognize(text="We are talking about names.")
        pending, _ = self.pipeline._assistant_queue.get_nowait()
        self.assertEqual(pending.source_text, "Tell me the origin of your name.")
        self.assertIn("ожидается ответ", self.results[-1].assistance_status)
        self.assertEqual(len(self.pipeline.history.get_recent()), 2)

    def test_new_question_evicts_pending_statement(self):
        self.pipeline.set_assistant_enabled(True)
        self.recognize(text="We are talking about names.")
        self.recognize(text="Tell me the origin of your name.")
        pending, _ = self.pipeline._assistant_queue.get_nowait()
        self.assertEqual(pending.source_text, "Tell me the origin of your name.")

    def test_disable_skips_pending_ai_request(self):
        self.pipeline.set_assistant_enabled(True)
        self.recognize()
        self.pipeline.set_assistant_enabled(False)
        self.finish_assistance()
        self.pipeline.assistant.suggest.assert_not_called()
        self.assertIn("выключены", self.results[-1].assistance_status)

    def test_slow_ai_does_not_block_next_translation(self):
        pipeline = self.pipeline
        pipeline.capture = MagicMock()
        pipeline.segmenter = MagicMock()
        pipeline._work_queue = queue.Queue()
        pipeline._worker_thread = None
        pipeline._assistant_thread = None
        pipeline._running = False
        ai_started, release_ai, translated_second = (threading.Event() for _ in range(3))

        def slow_suggestion(*args, **kwargs):
            ai_started.set()
            release_ai.wait(timeout=3)
            return Assistance()

        def on_phrase(result):
            self.results.append(result)
            if result.source_text == "Second?":
                translated_second.set()

        pipeline.assistant.suggest.side_effect = slow_suggestion
        pipeline.translator.translate.return_value = "Перевод"
        pipeline.on_phrase = on_phrase
        pipeline.recognizer.recognize_pcm16.side_effect = [
            SimpleNamespace(text="First?", language="en"),
            SimpleNamespace(text="Second?", language="en"),
        ]
        pipeline.set_assistant_enabled(True)
        pipeline.start()
        try:
            pipeline._handle_utterance(self.utterance)
            self.assertTrue(ai_started.wait(timeout=2))
            pipeline._handle_utterance(self.utterance)
            self.assertTrue(translated_second.wait(timeout=2), "Перевод заблокирован запросом ИИ")
            self.assertFalse(release_ai.is_set())
        finally:
            release_ai.set()
            pipeline.stop()
        self.assertFalse(pipeline._worker_thread.is_alive())
        self.assertFalse(pipeline._assistant_thread.is_alive())

    def test_image_request_does_not_block_audio_translation(self):
        from app.assistant_workspace import ContextChat
        from image_fixture import png_image
        pipeline = self.pipeline
        pipeline.capture, pipeline.segmenter = MagicMock(), MagicMock()
        pipeline._running = False
        pipeline._worker_thread = pipeline._assistant_thread = None
        pipeline._work_queue = queue.Queue()
        started, release, translated = threading.Event(), threading.Event(), threading.Event()
        pipeline.chat = ContextChat(pipeline.assistant, pipeline.workspace, image_directory=Path(self.temp.name) / "shots")
        pipeline.recognizer.recognize_pcm16.return_value = SimpleNamespace(text="New audio", language="en")
        pipeline.translator.translate.return_value = "Новая речь"
        pipeline.on_phrase = lambda result: translated.set()

        def slow_image(*args):
            started.set()
            release.wait(timeout=3)
            return AssistantReply("Разбор", language="ru", explanation_ru="Разбор")

        pipeline.assistant.analyze_image.side_effect = slow_image
        pipeline.start()
        try:
            pipeline.chat.ask_image(png_image(), "Разбери", lambda _: None)
            self.assertTrue(started.wait(timeout=2))
            pipeline._handle_utterance(self.utterance)
            self.assertTrue(translated.wait(timeout=2), "Разбор картинки заблокировал перевод речи")
            self.assertFalse(release.is_set())
            pipeline.capture.stop.assert_not_called()
        finally:
            release.set()
            pipeline.chat._thread.join(timeout=2)
            pipeline.chat.close()
            pipeline.stop()

    def test_preparation_finishes_before_audio_start(self):
        pipeline = self.pipeline
        pipeline._prepared = False
        pipeline._running = False
        pipeline._worker_thread = pipeline._assistant_thread = None
        pipeline._work_queue = queue.Queue()
        pipeline.capture, pipeline.segmenter = MagicMock(), MagicMock()
        order = []
        pipeline.segmenter.ensure_model.side_effect = lambda: order.append("vad")
        pipeline.recognizer.warm_up.side_effect = lambda: order.append("asr")
        pipeline.translator.warm_up.side_effect = lambda: order.append("translate")
        pipeline.assistant.warm_up.side_effect = lambda: order.append("ai")
        pipeline.capture.start.side_effect = lambda: order.append("audio")
        pipeline.set_assistant_enabled(True)
        pipeline.start()
        try:
            self.assertEqual(order, ["vad", "asr", "translate", "ai", "audio"])
            self.assertTrue(pipeline.is_prepared)
        finally:
            pipeline.stop()

    def test_failed_preparation_does_not_start_audio(self):
        pipeline = self.pipeline
        pipeline._prepared = False
        pipeline._running = False
        pipeline._worker_thread = pipeline._assistant_thread = None
        pipeline.capture, pipeline.segmenter = MagicMock(), MagicMock()
        pipeline.recognizer.warm_up.side_effect = RuntimeError("local model missing")
        with self.assertRaisesRegex(RuntimeError, "local model missing"):
            pipeline.start()
        self.assertFalse(pipeline.is_prepared)
        pipeline.capture.start.assert_not_called()
        pipeline.segmenter.start.assert_not_called()

    def test_prepare_once_and_ai_failure_keeps_translation_available(self):
        pipeline = self.pipeline
        pipeline._prepared = False
        pipeline.segmenter = MagicMock()
        pipeline.set_assistant_enabled(True)
        pipeline.assistant.warm_up.side_effect = RuntimeError("Ollama unavailable")
        progress = []
        with self.assertLogs("app.pipeline", level="WARNING"):
            warning = pipeline.prepare_models(progress.append)
            pipeline.prepare_models()
        self.assertEqual(warning, "Ollama unavailable")
        self.assertTrue(pipeline.is_prepared)
        self.assertEqual(progress, ["Silero VAD", "Whisper", "Переводчик EN ↔ RU", "Локальная Ollama"])
        pipeline.recognizer.warm_up.assert_called_once()
        pipeline.translator.warm_up.assert_called_once()
        self.assertEqual(pipeline.assistant.warm_up.call_count, 2)
        self.assertEqual(self.results, [])  # прогрев не создаёт карточек/истории

    def test_processing_logs_stage_times_without_transcript(self):
        with self.assertLogs("app.pipeline", level="INFO") as logs:
            self.recognize(text="Private lesson phrase")
        joined = "\n".join(logs.output)
        self.assertIn("ASR=", joined)
        self.assertIn("перевод=", joined)
        self.assertNotIn("Private lesson phrase", joined)

    def test_unsupported_language_is_reported(self):
        with self.assertLogs("app.pipeline", level="ERROR"):
            self.recognize("de")
        self.pipeline.on_error.assert_called_once()
        self.pipeline.translator.translate.assert_not_called()
        self.assertEqual(self.results, [])

    def test_empty_recognition_is_ignored(self):
        self.recognize(text="   ")
        self.assertEqual(self.results, [])
        self.pipeline.translator.translate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
