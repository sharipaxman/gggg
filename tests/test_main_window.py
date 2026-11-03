"""Offscreen-проверки UI без реального аудио и глобальных хоткеев."""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication, QPushButton
    from app.ui.main_window import MainWindow
    from app.assistant import Assistance, AnswerSuggestion
    from app.history_store import HistoryEntry
    from app.pipeline import PhraseResult
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
