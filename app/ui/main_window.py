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
    QStatusBar, QApplication, QMessageBox, QCheckBox,
)

from app.pipeline import LessonPipeline, PhraseResult
from app.assistant import Assistance
from app.hotkey_manager import HotkeyManager
from app.history_store import HistoryEntry
from app.config import HOTKEY_ATTACH_SCREENSHOT, HOTKEY_TOGGLE_LISTEN, HISTORY_LIMIT, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class PipelineBridge(QObject):
    """Мост сигналов: пайплайн работает в фоновых потоках, Qt-виджеты можно
    обновлять только из главного потока — поэтому прокидываем через сигнал."""

    phrase_ready = Signal(object)   # PhraseResult
    error_occurred = Signal(str)


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

        if entry.assistance.suggestions:
            answers = SuggestionsWidget()
            answers.show_assistance(entry.assistance, "Возможные ответы (примеры)")
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
        self.resize(980, 640)
        self.setStyleSheet("QMainWindow { background-color: #1e1f22; }")

        self.bridge = PipelineBridge()
        self.bridge.phrase_ready.connect(self._on_phrase_ready)
        self.bridge.error_occurred.connect(self._on_error)

        self._current_entry_id = 0
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
            "Нужны запущенная Ollama и загруженная модель. Последние 6 реплик передаются "
            "только локальной модели. Скриншоты не анализируются."
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

        current_layout.addWidget(self._section_label("Как можно ответить"))
        self.suggestions_widget = SuggestionsWidget()
        self.suggestions_widget.show_assistance(Assistance(), "Включите ИИ-подсказки для анализа вопросов")
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

        splitter.addWidget(right_panel)
        splitter.setSizes([480, 500])

        self.setStatusBar(QStatusBar())

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
            self.pipeline.stop()
            self.hotkeys.stop()
        except Exception:
            logger.exception("Ошибка при закрытии приложения")
        super().closeEvent(event)
