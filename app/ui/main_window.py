"""
Главное окно приложения.

Разделы интерфейса:
- Верхняя панель: статус (слушаем / остановлено), кнопка старт/стоп, индикатор "идёт речь...".
- Центр: текущая распознанная фраза (EN/RU), перевод и подсказки ИИ, кнопка "Скопировать" у перевода.
- Справа: контекст урока — список последних фраз с ответами и миниатюрами скриншотов.
- Низ: подсказка про горячие клавиши.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QObject, Signal, QSize
from PySide6.QtGui import QClipboard, QPixmap, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QListWidget, QListWidgetItem, QSplitter,
    QStatusBar, QApplication, QMessageBox, QCheckBox, QPlainTextEdit, QTabWidget,
)

from app.pipeline import LessonPipeline, PhraseResult
from app.assistant import Assistance
from app.assistant_workspace import PROMPT_MAX_CHARS, QUESTION_MAX_CHARS, MEMORY_LIMIT, CONTEXT_CHAR_BUDGET
from app.hotkey_manager import HotkeyManager
from app.history_store import HistoryEntry
from app.config import HOTKEY_ATTACH_SCREENSHOT, HOTKEY_TOGGLE_LISTEN, HISTORY_LIMIT, OLLAMA_MODEL, ASSISTANT_CONTEXT_ENTRIES

logger = logging.getLogger(__name__)


class PipelineBridge(QObject):
    """Мост сигналов: пайплайн работает в фоновых потоках, Qt-виджеты можно
    обновлять только из главного потока — поэтому прокидываем через сигнал."""

    phrase_ready = Signal(object)   # PhraseResult
    error_occurred = Signal(str)
    chat_ready = Signal(object)


class SuggestionsWidget(QWidget):
    """Одинаковые двуязычные подсказки для текущей фразы и истории."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = QVBoxLayout(self)
        self.rows.setContentsMargins(0, 0, 0, 0)

    def show_assistance(self, assistance: Assistance, status: str = ""):
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if status:
            label = QLabel(status)
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setStyleSheet("color: #aab8cf; font-size: 12px;")
            self.rows.addWidget(label)
        if assistance.response:
            prefix = f"Ответ ИИ ({assistance.response_language.upper()}):\n" if assistance.response_language else ""
            label = QLabel(prefix + assistance.response)
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setStyleSheet("color: #d1e8db; font-size: 14px;")
            self.rows.addWidget(label)
            copy = QPushButton("Копировать ответ ИИ")
            copy.clicked.connect(lambda: QApplication.clipboard().setText(assistance.response))
            self.rows.addWidget(copy)
        if assistance.response_translation:
            label = QLabel(f"Перевод ({assistance.translation_language.upper()}):\n{assistance.response_translation}")
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setStyleSheet("color: #b7c2d0; font-size: 14px;")
            self.rows.addWidget(label)
            copy = QPushButton("Копировать перевод подсказки")
            copy.clicked.connect(lambda: QApplication.clipboard().setText(assistance.response_translation))
            self.rows.addWidget(copy)
        if assistance.translation_error:
            warning = QLabel(f"Перевод подсказки недоступен: {assistance.translation_error}")
            warning.setTextFormat(Qt.PlainText)
            warning.setWordWrap(True)
            self.rows.addWidget(warning)
        for index, answer in enumerate(assistance.suggestions, 1):
            card = QWidget()
            layout = QVBoxLayout(card)
            for lang, text in (("EN", answer.en), ("RU", answer.ru)):
                label = QLabel(f"{index}. {lang}: {text}")
                label.setTextFormat(Qt.PlainText)
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                label.setStyleSheet("color: #d1e8db; font-size: 14px;")
                layout.addWidget(label)
                copy = QPushButton(f"Копировать ответ {lang}")
                copy.clicked.connect(lambda checked=False, value=text: QApplication.clipboard().setText(value))
                layout.addWidget(copy)
            self.rows.addWidget(card)


class HistoryItemWidget(QFrame):
    """Карточка истории: оригинал, перевод, подсказки и миниатюра скриншота."""

    def __init__(self, entry: HistoryEntry, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background-color: #2b2d31; border-radius: 8px; margin: 4px; }"
            "QLabel { color: #e3e5e8; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        time_label = QLabel(f"{entry.created_at}  •  {entry.source_language.upper()} → {entry.target_language.upper()}")
        time_label.setStyleSheet("color: #8a8d93; font-size: 11px;")
        layout.addWidget(time_label)

        en_label = QLabel(entry.source_text)
        en_label.setTextFormat(Qt.PlainText)
        en_label.setWordWrap(True)
        en_label.setStyleSheet("font-size: 13px; color: #b7c2d0;")
        layout.addWidget(en_label)

        row = QHBoxLayout()
        ru_label = QLabel(entry.translated_text)
        ru_label.setTextFormat(Qt.PlainText)
        ru_label.setWordWrap(True)
        ru_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #ffffff;")
        row.addWidget(ru_label, stretch=1)

        copy_btn = QPushButton("Копировать")
        copy_btn.setFixedWidth(90)
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(entry.translated_text))
        row.addWidget(copy_btn)
        layout.addLayout(row)

        if entry.assistance.suggestions or entry.assistance.response:
            answers = SuggestionsWidget()
            answers.show_assistance(entry.assistance, "Ответ ИИ" if entry.assistance.response else "Возможные ответы (примеры)")
            layout.addWidget(answers)

        if entry.screenshot_path and Path(entry.screenshot_path).exists():
            thumb = QLabel()
            pixmap = QPixmap(entry.screenshot_path)
            if not pixmap.isNull():
                thumb.setPixmap(pixmap.scaledToWidth(220, Qt.SmoothTransformation))
                thumb.setStyleSheet("margin-top: 6px;")
                layout.addWidget(thumb)

    def _copy_to_clipboard(self, text: str):
        QApplication.clipboard().setText(text, QClipboard.Clipboard)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EN Lesson Translator — локальный помощник на уроках английского")
        self.resize(1200, 800)
        self.setStyleSheet("QMainWindow { background-color: #1e1f22; }")

        self.bridge = PipelineBridge()
        self.bridge.phrase_ready.connect(self._on_phrase_ready)
        self.bridge.error_occurred.connect(self._on_error)
        self.bridge.chat_ready.connect(self._on_chat_ready)

        self._current_entry_id = 0
        self._last_chat_answer = ""
        self._last_chat_translation = ""
        self.pipeline = LessonPipeline(on_phrase=self._emit_phrase_ready,
                                       on_error=self.bridge.error_occurred.emit)
        self.hotkeys = HotkeyManager(
            on_screenshot=self._emit_screenshot_hotkey,
            on_toggle_listen=self._emit_toggle_hotkey,
        )

        self._build_ui()
        self._load_history()

        try:
            self.hotkeys.start()
        except Exception:
            logger.exception("Не удалось зарегистрировать глобальные хоткеи")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- левая колонка: текущая фраза + управление ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        top_bar = QHBoxLayout()
        self.status_label = QLabel("Остановлено")
        self.status_label.setStyleSheet("color: #f2a93c; font-weight: 600; font-size: 14px;")
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()

        self.toggle_btn = QPushButton("▶ Начать слушать")
        self.toggle_btn.setFixedWidth(180)
        self.toggle_btn.clicked.connect(self._toggle_listening)
        top_bar.addWidget(self.toggle_btn)
        left_layout.addLayout(top_bar)

        self.assistant_checkbox = QCheckBox(f"ИИ-подсказки по контексту (Ollama: {OLLAMA_MODEL})")
        self.assistant_checkbox.setStyleSheet("color: #d1e8db;")
        self.assistant_checkbox.setToolTip(
            "Нужны запущенная Ollama и загруженная модель. Промпт и ограниченный фрагмент памяти "
            "передаются только локальной модели. Скриншоты не анализируются."
        )
        self.assistant_checkbox.toggled.connect(self.pipeline.set_assistant_enabled)
        left_layout.addWidget(self.assistant_checkbox)

        current_frame = QFrame()
        current_frame.setStyleSheet(
            "QFrame { background-color: #2b2d31; border-radius: 10px; padding: 16px; }"
        )
        current_layout = QVBoxLayout(current_frame)

        self.source_title = self._section_label("Текущая фраза — автоопределение EN / RU")
        current_layout.addWidget(self.source_title)
        self.current_en_label = QLabel("—")
        self.current_en_label.setTextFormat(Qt.PlainText)
        self.current_en_label.setWordWrap(True)
        self.current_en_label.setStyleSheet("color: #b7c2d0; font-size: 16px;")
        current_layout.addWidget(self.current_en_label)

        current_layout.addWidget(self._section_label("Перевод"))
        self.current_ru_label = QLabel("—")
        self.current_ru_label.setTextFormat(Qt.PlainText)
        self.current_ru_label.setWordWrap(True)
        self.current_ru_label.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: 700;")
        current_layout.addWidget(self.current_ru_label)

        copy_row = QHBoxLayout()
        self.copy_current_btn = QPushButton("📋 Скопировать перевод")
        self.copy_current_btn.clicked.connect(self._copy_current_translation)
        copy_row.addWidget(self.copy_current_btn)
        copy_row.addStretch()
        current_layout.addLayout(copy_row)

        current_layout.addWidget(self._section_label("Ответ ИИ"))
        self.suggestions_widget = SuggestionsWidget()
        self.suggestions_widget.show_assistance(Assistance(), "Включите автоответы ИИ; промпт задаётся во вкладке справа")
        current_layout.addWidget(self.suggestions_widget)

        current_scroll = QScrollArea()
        current_scroll.setWidgetResizable(True)
        current_scroll.setWidget(current_frame)
        left_layout.addWidget(current_scroll, stretch=1)

        hint = QLabel(
            f"Хоткей скриншота: {HOTKEY_ATTACH_SCREENSHOT.upper()}  •  "
            f"Старт/стоп прослушивания: {HOTKEY_TOGGLE_LISTEN.upper()}\n"
            "Перевод появляется только после того, как преподаватель закончил фразу (пауза в речи), "
            "а не по отдельным словам."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a8d93; font-size: 11px; margin-top: 8px;")
        left_layout.addWidget(hint)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # --- правая колонка: контекст урока / история ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(self._section_label(f"История урока (последние {HISTORY_LIMIT} фраз)"))

        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_container = QWidget()
        self.history_container_layout = QVBoxLayout(self.history_container)
        self.history_container_layout.setAlignment(Qt.AlignTop)
        self.history_scroll.setWidget(self.history_container)
        right_layout.addWidget(self.history_scroll)

        right_tabs = QTabWidget()
        right_tabs.addTab(self._build_assistant_panel(), "Промпт и чат")
        right_tabs.addTab(right_panel, "История переводов")
        splitter.addWidget(right_tabs)
        splitter.setSizes([480, 500])

        self.setStatusBar(QStatusBar())

    def _build_assistant_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        preferences = self.pipeline.workspace.preferences()
        layout.addWidget(self._section_label(f"Инструкция ИИ (до {PROMPT_MAX_CHARS} символов)"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Например: объясняй новые слова; отвечай кратко по-русски…")
        self.prompt_edit.setPlainText(preferences.prompt)
        self.prompt_edit.setMaximumHeight(110)
        layout.addWidget(self.prompt_edit)
        self.always_reply_checkbox = QCheckBox("Обрабатывать каждую реплику по промпту, не только вопросы")
        self.always_reply_checkbox.setChecked(preferences.always_reply)
        layout.addWidget(self.always_reply_checkbox)
        apply_btn = QPushButton("Применить и сохранить промпт")
        apply_btn.clicked.connect(self._save_preferences)
        layout.addWidget(apply_btn)

        self.memory_info = self._section_label(
            f"Память: до {MEMORY_LIMIT} событий на диске; в запросе свежий фрагмент до {CONTEXT_CHAR_BUDGET} символов"
        )
        self.memory_info.setWordWrap(True)
        layout.addWidget(self.memory_info)
        self.memory_preview = QPlainTextEdit()
        self.memory_preview.setReadOnly(True)
        self.memory_preview.setMaximumHeight(90)
        layout.addWidget(self.memory_preview)
        self.clear_memory_btn = QPushButton("Очистить память и чат (переводы останутся)")
        self.clear_memory_btn.clicked.connect(self._clear_memory)
        layout.addWidget(self.clear_memory_btn)

        layout.addWidget(self._section_label("Вопрос ИИ по контексту — работает и без прослушивания"))
        self.chat_view = QPlainTextEdit()
        self.chat_view.setReadOnly(True)
        layout.addWidget(self.chat_view, stretch=1)
        self.question_edit = QPlainTextEdit()
        self.question_edit.setPlaceholderText(f"Свой вопрос по контексту (до {QUESTION_MAX_CHARS} символов)…")
        self.question_edit.setMaximumHeight(70)
        layout.addWidget(self.question_edit)
        buttons = QHBoxLayout()
        self.ask_btn = QPushButton("Спросить по контексту")
        self.ask_btn.clicked.connect(self._ask_question)
        buttons.addWidget(self.ask_btn)
        self.copy_chat_btn = QPushButton("Копировать ответ")
        self.copy_chat_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._last_chat_answer))
        buttons.addWidget(self.copy_chat_btn)
        self.copy_chat_translation_btn = QPushButton("Копировать перевод")
        self.copy_chat_translation_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._last_chat_translation))
        buttons.addWidget(self.copy_chat_translation_btn)
        layout.addLayout(buttons)
        self.chat_status = QLabel("Промпт и память сохраняются локально. Автоответ включается флажком слева.")
        self.chat_status.setTextFormat(Qt.PlainText)
        self.chat_status.setWordWrap(True)
        layout.addWidget(self.chat_status)
        self._refresh_memory()
        self._load_chat()
        return panel

    def _save_preferences(self):
        try:
            self.pipeline.workspace.save_preferences(
                self.prompt_edit.toPlainText(), self.always_reply_checkbox.isChecked(),
            )
        except Exception as exc:
            self.chat_status.setText(f"Не удалось сохранить промпт: {exc}")
            return False
        self.chat_status.setText("Промпт сохранён. Будет применяться к следующим запросам ИИ.")
        return True

    def _refresh_memory(self):
        _, context = self.pipeline.workspace.snapshot()
        roles = {"heard": "Услышано", "user": "Вы", "assistant": "ИИ", "suggestion": "Пример ИИ"}
        self.memory_preview.setPlainText("\n".join(f"{roles[event['role']]}: {event['text']}" for event in context))

    def _load_chat(self):
        events = self.pipeline.workspace.recent(chat_only=True)
        blocks = []
        for event in events:
            blocks.append(f"{'Вы' if event['role'] == 'user' else 'ИИ'}: {event['text']}")
            if event.get("translation"):
                blocks.append(f"Перевод ({event.get('translation_language', '').upper()}): {event['translation']}")
            if event.get("translation_error"):
                blocks.append(f"Перевод недоступен: {event['translation_error']}")
        self.chat_view.setPlainText("\n\n".join(blocks))
        answers = [event for event in events if event["role"] == "assistant"]
        self._last_chat_answer = answers[-1]["text"] if answers else ""
        self._last_chat_translation = answers[-1].get("translation", "") if answers else ""
        self.copy_chat_translation_btn.setEnabled(bool(self._last_chat_translation))
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_chat_busy(self, busy: bool):
        self.ask_btn.setEnabled(not busy)
        self.question_edit.setEnabled(not busy)
        self.clear_memory_btn.setEnabled(not busy)

    def _ask_question(self):
        if not self._save_preferences():
            return
        self._set_chat_busy(True)
        self.chat_status.setText("ИИ отвечает… Если идёт автоответ, ручной запрос дождётся модели.")
        try:
            self.pipeline.chat.ask(self.question_edit.toPlainText(), self.bridge.chat_ready.emit)
            self.question_edit.clear()
            self._load_chat()
            self._refresh_memory()
        except Exception as exc:
            self._set_chat_busy(False)
            self.chat_status.setText(str(exc))

    def _on_chat_ready(self, result):
        self._set_chat_busy(False)
        self._load_chat()
        self._refresh_memory()
        self.chat_status.setText(
            f"Ошибка: {result.error}" if result.error else
            (f"Ответ готов. Перевод недоступен: {result.translation_error}" if result.translation_error
             else "Ответ и перевод готовы и сохранены в чате.")
        )

    def _clear_memory(self):
        if QMessageBox.question(self, "Очистить память?",
                                "Удалить память ИИ и ручной чат? Промпт и история переводов останутся.") != QMessageBox.Yes:
            return
        self.pipeline.workspace.clear_memory()
        self._load_chat()
        self._refresh_memory()
        self.chat_status.setText("Память очищена. Начат новый контекст.")

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #8a8d93; font-size: 11px; font-weight: 600; margin-top: 10px;")
        return lbl

    # ------------------------------------------------------------------
    # Загрузка истории при старте
    # ------------------------------------------------------------------
    def _load_history(self):
        entries = self.pipeline.history.get_recent()
        for entry in reversed(entries):  # prepend сохраняет порядок «сначала новые»
            self._prepend_history_widget(entry)

    def _prepend_history_widget(self, entry: HistoryEntry):
        widget = HistoryItemWidget(entry)
        # Обновление ИИ для той же фразы не создаёт дубликат карточки.
        position = 0
        for index in range(self.history_container_layout.count()):
            old = self.history_container_layout.itemAt(index).widget()
            if old.entry.id == entry.id:
                position = index
                self.history_container_layout.takeAt(index)
                old.deleteLater()
                break
        self.history_container_layout.insertWidget(position, widget)
        # ограничиваем число видимых виджетов, чтобы не раздувать UI
        while self.history_container_layout.count() > HISTORY_LIMIT:
            item = self.history_container_layout.takeAt(self.history_container_layout.count() - 1)
            if item.widget():
                item.widget().deleteLater()

    # ------------------------------------------------------------------
    # Управление прослушиванием
    # ------------------------------------------------------------------
    def _toggle_listening(self):
        if self.pipeline.is_running:
            self.pipeline.stop()
            self.status_label.setText("Остановлено")
            self.status_label.setStyleSheet("color: #f2a93c; font-weight: 600; font-size: 14px;")
            self.toggle_btn.setText("▶ Начать слушать")
        else:
            try:
                self.pipeline.start()
            except Exception as exc:
                logger.exception("Не удалось запустить прослушивание")
                QMessageBox.critical(self, "Ошибка", f"Не удалось начать захват звука:\n{exc}")
                return
            self.status_label.setText("Слушаю системный звук...")
            self.status_label.setStyleSheet("color: #4caf7d; font-weight: 600; font-size: 14px;")
            self.toggle_btn.setText("⏸ Остановить")

    # ------------------------------------------------------------------
    # Хоткеи (вызываются из потока библиотеки `keyboard` — пробрасываем в UI-поток через сигналы)
    # ------------------------------------------------------------------
    def _emit_screenshot_hotkey(self):
        try:
            self.pipeline.capture_screenshot_now()
            self.statusBar().showMessage("Снимок экрана сделан — будет прикреплён к следующей фразе", 4000)
        except Exception:
            logger.exception("Не удалось сделать снимок экрана")

    def _emit_toggle_hotkey(self):
        self.bridge.error_occurred.emit("__toggle__")

    def _on_error(self, msg: str):
        if msg == "__toggle__":
            self._toggle_listening()
        else:
            self.statusBar().showMessage(f"Ошибка: {msg}", 15000)

    # ------------------------------------------------------------------
    # Новая распознанная + переведённая фраза
    # ------------------------------------------------------------------
    def _emit_phrase_ready(self, result: PhraseResult):
        # вызывается из рабочего потока пайплайна — пробрасываем в UI-поток через сигнал
        self.bridge.phrase_ready.emit(result)

    def _on_phrase_ready(self, result: PhraseResult):
        self._refresh_memory()
        if result.entry_id >= self._current_entry_id:
            self._current_entry_id = result.entry_id
            self.source_title.setText(
                f"Текущая фраза ({result.source_language.upper()} → {result.target_language.upper()})"
            )
            self.current_en_label.setText(result.source_text)
            self.current_ru_label.setText(result.translated_text)
            self.suggestions_widget.show_assistance(result.assistance, result.assistance_status)

        self._prepend_history_widget(HistoryEntry(
            id=result.entry_id, created_at=result.created_at, source_text=result.source_text,
            translated_text=result.translated_text, screenshot_path=result.screenshot_path,
            source_language=result.source_language, target_language=result.target_language,
            assistance=result.assistance,
        ))

    def _copy_current_translation(self):
        text = self.current_ru_label.text()
        if text and text != "—":
            QApplication.clipboard().setText(text, QClipboard.Clipboard)
            self.statusBar().showMessage("Перевод скопирован в буфер обмена", 2000)

    def closeEvent(self, event):
        try:
            self._save_preferences()
            self.pipeline.close()
            self.hotkeys.stop()
        except Exception:
            logger.exception("Ошибка при закрытии приложения")
        super().closeEvent(event)
