"""Offscreen-проверки UI без реального аудио и глобальных хоткеев."""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication, QPushButton
    from app.ui.main_window import MainWindow
    from app.assistant import Assistance, AnswerSuggestion, ReplyVariant
    from app.history_store import HistoryEntry
    from app.pipeline import PhraseResult
    from app.window_privacy import CaptureProtectionResult
    from app.assistant_workspace import AssistantPreferences, ChatResult
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "Нужны PySide6 и системные библиотеки Qt")
class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.privacy_support_patch = patch("app.ui.main_window.capture_protection_support", return_value=(True, "Supported"))
        self.privacy_support_patch.start()
        self.addCleanup(self.privacy_support_patch.stop)
        self.pipeline_patch = patch("app.ui.main_window.LessonPipeline")
        self.hotkeys_patch = patch("app.ui.main_window.HotkeyManager")
        self.pipeline_factory = self.pipeline_patch.start()
        self.hotkeys_patch.start()
        self.addCleanup(self.pipeline_patch.stop)
        self.addCleanup(self.hotkeys_patch.stop)
        self.pipeline = self.pipeline_factory.return_value
        self.pipeline.is_running = False
        self.pipeline.is_prepared = False
        self.pipeline.history.get_recent.return_value = []
        self.pipeline.workspace.preferences.return_value = AssistantPreferences()
        self.pipeline.workspace.snapshot.return_value = (0, [])
        self.pipeline.workspace.recent.return_value = []
        self.window = MainWindow()
        self.addCleanup(self.window.close)

    def screenshot_dialogs(self, selected=True, confirmed=True):
        selector, composer = MagicMock(), MagicMock()
        selector.exec.return_value = 1 if selected else 0
        composer.exec.return_value = 1 if confirmed else 0
        composer.image_png = b"png"
        composer.question = "Объясни упражнение 2"
        return selector, composer

    def test_confirmed_screenshot_and_comment_do_not_stop_listening(self):
        self.pipeline.is_running = True
        selector, composer = self.screenshot_dialogs()
        with patch("app.ui.main_window.RegionSelectionDialog", return_value=selector), \
                patch("app.ui.main_window.ScreenshotQuestionDialog", return_value=composer):
            self.window._capture_image_question()
        self.pipeline.chat.ask_image.assert_called_once()
        self.assertEqual(self.pipeline.chat.ask_image.call_args.args[:2], (b"png", "Объясни упражнение 2"))
        self.pipeline.stop.assert_not_called()
        self.pipeline.start.assert_not_called()
        self.assertTrue(self.window._chat_busy)
        self.assertFalse(self.window.image_btn.isEnabled())
        self.window._on_chat_ready(ChatResult("Q", answer="Объяснение", explanation_ru="Объяснение"))
        self.assertTrue(self.window.image_btn.isEnabled())
        self.assertIn("Разбор снимка готов", self.window.chat_status.text())

    def test_cancelling_selection_or_preview_never_sends_image(self):
        for selected, confirmed in ((False, False), (True, False)):
            with self.subTest(selected=selected):
                selector, composer = self.screenshot_dialogs(selected, confirmed)
                with patch("app.ui.main_window.RegionSelectionDialog", return_value=selector), \
                        patch("app.ui.main_window.ScreenshotQuestionDialog", return_value=composer):
                    self.window._capture_image_question()
                self.pipeline.chat.ask_image.assert_not_called()
                self.assertFalse(self.window._image_capture_active)
                self.assertFalse(self.window._chat_busy)

    def test_busy_or_duplicate_hotkey_does_not_open_another_selector(self):
        for flag in ("_chat_busy", "_image_capture_active"):
            setattr(self.window, flag, True)
            with patch("app.ui.main_window.RegionSelectionDialog") as selector:
                self.window._capture_image_question()
                selector.assert_not_called()
            setattr(self.window, flag, False)

    def test_image_dispatch_failure_restores_controls(self):
        self.pipeline.chat.ask_image.side_effect = RuntimeError("Previous request is busy")
        selector, composer = self.screenshot_dialogs()
        with patch("app.ui.main_window.RegionSelectionDialog", return_value=selector), \
                patch("app.ui.main_window.ScreenshotQuestionDialog", return_value=composer):
            self.window._capture_image_question()
        self.assertFalse(self.window._chat_busy)
        self.assertTrue(self.window.image_btn.isEnabled())
        self.assertIn("Previous request is busy", self.window.chat_status.text())

    def test_privacy_failure_prevents_opening_screenshot_dialog(self):
        self.window._capture_exclusion_requested = True
        dialog = MagicMock()
        dialog.winId.return_value = 42
        failure = CaptureProtectionResult(False, False, "Denied")
        with patch("app.ui.main_window.set_capture_exclusion", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "Denied"):
                self.window._exec_image_dialog(dialog)
        dialog.exec.assert_not_called()
        dialog.deleteLater.assert_called_once()

    def test_image_explanation_is_visible_before_both_variants(self):
        variants = [{"kind": "short", "text": "It is a cat.", "translation": "Это кот.",
                     "language": "en", "target_language": "ru", "translation_error": ""},
                    {"kind": "long", "text": "It is a cat. It is small.", "translation": "Это маленький кот.",
                     "language": "en", "target_language": "ru", "translation_error": ""}]
        self.pipeline.workspace.recent.return_value = [
            {"role": "user", "text": "Опиши", "image_path": "/data/vision.png"},
            {"role": "assistant", "text": "Combined", "explanation_ru": "Нужно описать животное.", "variants": variants},
        ]
        self.window._load_chat()
        text = self.window.chat_view.toPlainText()
        self.assertIn("vision.png", text)
        self.assertIn("Нужно описать животное", text)
        self.assertIn("It is a cat", text)
        self.assertIn("Это кот", text)

    def test_first_listen_prepares_in_background_before_recording(self):
        with patch("app.ui.main_window.threading.Thread") as thread:
            self.window._toggle_listening()
            thread.assert_called_once()
            thread.return_value.start.assert_called_once()
        self.assertTrue(self.window._preparing)
        self.assertFalse(self.window.toggle_btn.isEnabled())
        self.assertIn("запись не идёт", self.window.status_label.text())
        self.pipeline.start.assert_not_called()
        self.pipeline.is_prepared = True
        self.window._on_preparation_finished(True, "")
        self.pipeline.start.assert_called_once()
        self.assertFalse(self.window._preparing)

    def test_preparation_failure_does_not_start_recording(self):
        with patch("app.ui.main_window.threading.Thread"):
            self.window._toggle_listening()
        self.window._on_preparation_finished(False, "missing local model")
        self.pipeline.start.assert_not_called()
        self.assertTrue(self.window.prepare_btn.isEnabled())
        self.assertIn("missing local model", self.window.statusBar().currentMessage())

    def test_prepare_button_does_not_start_listening_and_close_cancels_auto_start(self):
        with patch("app.ui.main_window.threading.Thread"):
            self.window._begin_preparation(False)
        self.window._on_preparation_finished(True, "Ollama unavailable")
        self.pipeline.start.assert_not_called()
        self.assertIn("Ollama unavailable", self.window.statusBar().currentMessage())
        with patch("app.ui.main_window.threading.Thread"):
            self.window._begin_preparation(True)
        self.window.close()
        self.window._on_preparation_finished(True, "")
        self.pipeline.start.assert_not_called()

    def test_default_disabled_checkbox_controls_pipeline(self):
        self.assertFalse(self.window.assistant_checkbox.isChecked())
        self.window.assistant_checkbox.setChecked(True)
        self.pipeline.set_assistant_enabled.assert_called_once_with(True)

    def test_capture_exclusion_is_opt_in(self):
        self.assertFalse(self.window.capture_exclusion_checkbox.isChecked())
        self.assertFalse(self.window._capture_exclusion_requested)

    def test_capture_checkbox_applies_and_removes_native_exclusion(self):
        def result(hwnd, enabled):
            return CaptureProtectionResult(True, enabled, "Accepted")

        with patch("app.ui.main_window.set_capture_exclusion", side_effect=result) as native:
            self.window.capture_exclusion_checkbox.setChecked(True)
            self.assertTrue(self.window._capture_exclusion_requested)
            self.assertEqual(native.call_args.args[1], True)
            self.assertGreater(native.call_args.args[0], 0)
            self.window.capture_exclusion_checkbox.setChecked(False)
            self.assertFalse(self.window._capture_exclusion_requested)
            self.assertEqual(native.call_args.args[1], False)

    def test_capture_failure_is_visible_and_not_claimed_as_enabled(self):
        with patch("app.ui.main_window.set_capture_exclusion", return_value=CaptureProtectionResult(False, False, "Denied")):
            self.window.capture_exclusion_checkbox.setChecked(True)
        self.assertFalse(self.window.capture_exclusion_checkbox.isChecked())
        self.assertEqual(self.window.capture_exclusion_status.text(), "Denied")

    def test_capture_exclusion_can_be_reapplied_after_native_window_change(self):
        with patch("app.ui.main_window.set_capture_exclusion", return_value=CaptureProtectionResult(True, True, "Accepted")) as native:
            self.window.capture_exclusion_checkbox.setChecked(True)
            self.window._reapply_capture_exclusion()
            self.assertGreaterEqual(native.call_count, 2)
            self.assertTrue(self.window.capture_exclusion_checkbox.isChecked())

    def test_ai_updates_card_without_duplicate_or_reverting_current_phrase(self):
        first = PhraseResult("Why?", "Почему?", None, 1)
        second = PhraseResult("Что дальше?", "What next?", None, 2, "ru", "en")
        self.window._on_phrase_ready(first)
        self.window._on_phrase_ready(second)
        first.assistance = Assistance(True, [AnswerSuggestion("Because…", "Потому что…")])
        self.window._on_phrase_ready(first)
        self.assertEqual(self.window.current_en_label.text(), "Что дальше?")
        self.assertIn("RU → EN", self.window.source_title.text())
        layout = self.window.history_container_layout
        self.assertEqual(layout.count(), 2)
        self.assertEqual([layout.itemAt(i).widget().entry.id for i in range(2)], [2, 1])
        self.assertTrue(layout.itemAt(1).widget().entry.assistance.is_question)

    def test_copy_answer_english_and_russian(self):
        result = PhraseResult("Why?", "Почему?", None, 1,
                              assistance=Assistance(True, [AnswerSuggestion("Because…", "Потому что…")]))
        self.window._on_phrase_ready(result)
        buttons = self.window.suggestions_widget.findChildren(QPushButton)
        buttons[0].click()
        self.assertEqual(QApplication.clipboard().text(), "Because…")
        buttons[1].click()
        self.assertEqual(QApplication.clipboard().text(), "Потому что…")

    def test_prompt_applies_and_manual_question_does_not_start_listening(self):
        self.window.prompt_edit.setPlainText("Summarize in Russian")
        self.window._save_preferences()
        self.pipeline.workspace.save_preferences.assert_called_with("Summarize in Russian", True)
        self.window.question_edit.setPlainText("What did we discuss?")
        self.window._ask_question()
        self.pipeline.chat.ask.assert_called_once()
        self.assertEqual(self.pipeline.chat.ask.call_args.args[0], "What did we discuss?")
        self.pipeline.start.assert_not_called()
        self.assertFalse(self.window.ask_btn.isEnabled())
        self.window._on_chat_ready(ChatResult("What did we discuss?", error="Offline"))
        self.assertTrue(self.window.ask_btn.isEnabled())
        self.assertIn("Offline", self.window.chat_status.text())

    def test_free_response_is_shown_and_can_be_copied(self):
        result = PhraseResult("A statement.", "Утверждение.", None, 1,
                              assistance=Assistance(response="Свободный ответ"))
        self.window._on_phrase_ready(result)
        buttons = self.window.suggestions_widget.findChildren(QPushButton)
        buttons[0].click()
        self.assertEqual(QApplication.clipboard().text(), "Свободный ответ")

    def test_free_suggestion_has_separate_translation_and_copy_button(self):
        result = PhraseResult("Tell me about your name.", "Расскажи о своём имени.", None, 1,
                              assistance=Assistance(response="My name is Arthur.",
                                                    response_translation="Меня зовут Артур.",
                                                    response_language="en", translation_language="ru"))
        self.window._on_phrase_ready(result)
        buttons = self.window.suggestions_widget.findChildren(QPushButton)
        buttons[0].click()
        self.assertEqual(QApplication.clipboard().text(), "My name is Arthur.")
        buttons[1].click()
        self.assertEqual(QApplication.clipboard().text(), "Меня зовут Артур.")

    def test_manual_translation_is_loaded_and_copied_after_restart(self):
        self.pipeline.workspace.recent.return_value = [
            {"role": "user", "text": "мое имя артур"},
            {"role": "assistant", "text": "My name is Arthur.", "translation": "Меня зовут Артур.",
             "translation_language": "ru"},
        ]
        self.window._load_chat()
        self.assertIn("Перевод (RU): Меня зовут Артур.", self.window.chat_view.toPlainText())
        self.window.copy_chat_translation_btn.click()
        self.assertEqual(QApplication.clipboard().text(), "Меня зовут Артур.")
        self.window.copy_chat_btn.click()
        self.assertEqual(QApplication.clipboard().text(), "My name is Arthur.")

    def test_short_and_long_cards_each_have_their_own_translation(self):
        variants = [ReplyVariant("short", "Short answer", "Короткий перевод", "en", "ru"),
                    ReplyVariant("long", "A longer answer", "Длинный перевод", "en", "ru")]
        result = PhraseResult("Q", "В", None, 1, assistance=Assistance(variants=variants))
        self.window._on_phrase_ready(result)
        buttons = self.window.suggestions_widget.findChildren(QPushButton)
        self.assertEqual(len(buttons), 4)
        for button, expected in zip(buttons, ("Short answer", "Короткий перевод", "A longer answer", "Длинный перевод")):
            button.click()
            self.assertEqual(QApplication.clipboard().text(), expected)

    def test_manual_chat_interleaves_each_variant_and_its_translation(self):
        self.pipeline.workspace.recent.return_value = [{
            "role": "assistant", "text": "Flat original", "translation": "Flat translation",
            "variants": [
                {"kind": "short", "text": "Short answer", "translation": "Короткий перевод", "language": "en", "target_language": "ru"},
                {"kind": "long", "text": "A longer answer", "translation": "Длинный перевод", "language": "en", "target_language": "ru"},
            ],
        }]
        self.window._load_chat()
        text = self.window.chat_view.toPlainText()
        self.assertIn("Короткий вариант (EN)", text)
        self.assertIn("Развёрнутый вариант (EN)", text)
        self.assertLess(text.index("Короткий перевод"), text.index("A longer answer"))
        self.assertNotIn("Flat original", text)
        self.assertNotIn("Flat translation", text)

    def test_history_load_newest_first(self):
        self.pipeline.history.get_recent.return_value = [
            HistoryEntry(2, "now", "new", "новое", None),
            HistoryEntry(1, "before", "old", "старое", None),
        ]
        self.window._load_history()
        layout = self.window.history_container_layout
        self.assertEqual([layout.itemAt(i).widget().entry.id for i in range(2)], [2, 1])


if __name__ == "__main__":
    unittest.main()
