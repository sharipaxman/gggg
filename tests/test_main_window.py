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
        self.pipeline_patch = patch("app.ui.main_window.LessonPipeline")
        self.hotkeys_patch = patch("app.ui.main_window.HotkeyManager")
        self.pipeline_factory = self.pipeline_patch.start()
        self.hotkeys_patch.start()
        self.addCleanup(self.pipeline_patch.stop)
        self.addCleanup(self.hotkeys_patch.stop)
        self.pipeline = self.pipeline_factory.return_value
        self.pipeline.history.get_recent.return_value = []
        self.pipeline.workspace.preferences.return_value = AssistantPreferences()
        self.pipeline.workspace.snapshot.return_value = (0, [])
        self.pipeline.workspace.recent.return_value = []
        self.window = MainWindow()
        self.addCleanup(self.window.close)

    def test_default_disabled_checkbox_controls_pipeline(self):
        self.assertFalse(self.window.assistant_checkbox.isChecked())
        self.window.assistant_checkbox.setChecked(True)
        self.pipeline.set_assistant_enabled.assert_called_once_with(True)

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
