"""Выбор области на мониторе под курсором и явное подтверждение отправки.

Снимок берётся ДО показа оверлея, в PNG попадают только выделенные пиксели.
До подтверждения изображение существует лишь в памяти, без файлов и HTTP.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QPoint, QRect, QBuffer, QIODevice
from PySide6.QtGui import QGuiApplication, QCursor, QPainter, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit,
    QScrollArea, QDialogButtonBox,
)

from app.image_attachment import MAX_IMAGE_SIDE, validate_png
from app.assistant_workspace import QUESTION_MAX_CHARS


def physical_crop(rect: QRect, logical_width: int, logical_height: int, pixel_width: int, pixel_height: int) -> QRect:
    """Локальные logical координаты -> реальные пиксели, в том числе при 150% DPI."""
    left = min(pixel_width, max(0, math.floor(rect.x() * pixel_width / logical_width)))
    top = min(pixel_height, max(0, math.floor(rect.y() * pixel_height / logical_height)))
    right = min(pixel_width, math.ceil((rect.x() + rect.width()) * pixel_width / logical_width))
    bottom = min(pixel_height, math.ceil((rect.y() + rect.height()) * pixel_height / logical_height))
    return QRect(left, top, max(0, right - left), max(0, bottom - top))


class RegionSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("Не найден монитор для снимка")
        self._snapshot = screen.grabWindow(0)
        if self._snapshot.isNull():
            raise RuntimeError("Не удалось снять экран. Проверьте разрешения захвата экрана.")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self._origin = None
        self._selection = QRect()
        self.selected_image = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self._snapshot)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if not self._selection.isEmpty():
            painter.save()
            painter.setClipRect(self._selection)
            painter.drawPixmap(self.rect(), self._snapshot)
            painter.restore()
            painter.setPen(QPen(QColor("#66d9ff"), 2))
            painter.drawRect(self._selection)
        painter.fillRect(QRect(12, 12, min(self.width() - 24, 780), 48), QColor(25, 25, 25, 220))
        painter.setPen(Qt.white)
        painter.drawText(QRect(24, 12, self.width() - 48, 48), Qt.AlignVCenter,
                         "Выделите область мышью • Esc / правая кнопка — отмена • Только этот монитор")

    def _point(self, event):
        point = event.position().toPoint()
        return QPoint(max(0, min(point.x(), self.width() - 1)), max(0, min(point.y(), self.height() - 1)))

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.reject()
        elif event.button() == Qt.LeftButton:
            self._origin = self._point(event)
            self._selection = QRect()

    def mouseMoveEvent(self, event):
        if self._origin is not None:
            self._selection = QRect(self._origin, self._point(event)).normalized().intersected(self.rect())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._origin is None:
            return
        self.mouseMoveEvent(event)
        self._origin = None
        if self._selection.width() < 10 or self._selection.height() < 10:
            return
        rect = physical_crop(self._selection, self.width(), self.height(),
                             self._snapshot.width(), self._snapshot.height())
        self.selected_image = self._snapshot.toImage().copy(rect)
        self.selected_image.setDevicePixelRatio(1.0)
        self.accept()


class ScreenshotQuestionDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        if image is None or image.isNull():
            raise ValueError("Выбранная область пуста")
        self.setWindowTitle("Снимок → локальная Qwen • проверьте перед отправкой")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.resize(800, 650)
        resized = max(image.width(), image.height()) > MAX_IMAGE_SIDE
        if resized:
            image = image.scaled(MAX_IMAGE_SIDE, MAX_IMAGE_SIDE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image.setDevicePixelRatio(1.0)
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        if not image.save(buffer, "PNG"):
            raise RuntimeError("Не удалось подготовить PNG")
        self.image_png = bytes(buffer.data())
        validate_png(self.image_png)

        layout = QVBoxLayout(self)
        notice = QLabel(
            "В локальную модель уйдут выбранная область, ваш текст и свежий контекст урока. "
            "Распознавание речи и перевод не останавливаются."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        if resized:
            warning = QLabel(f"Большая область уменьшена до {image.width()}×{image.height()}. "
                             "Если текст мелкий, отмените и выделите меньшую область.")
            warning.setWordWrap(True)
            layout.addWidget(warning)
        # 1:1, со скроллом, чтобы можно было проверить читаемость реальных пикселей.
        preview = QLabel()
        preview.setPixmap(QPixmap.fromImage(image))
        preview.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(preview)
        scroll.setAlignment(Qt.AlignCenter)
        layout.addWidget(scroll, 1)
        layout.addWidget(QLabel(f"Что сделать? Необязательно, до {QUESTION_MAX_CHARS} символов:"))
        self.question_edit = QPlainTextEdit()
        self.question_edit.setPlaceholderText("Например: реши упражнение 3; объясни картинку; помоги ответить…")
        self.question_edit.setMaximumHeight(95)
        layout.addWidget(self.question_edit)
        self.counter = QLabel()
        self.counter.setWordWrap(True)
        layout.addWidget(self.counter)
        buttons = QDialogButtonBox()
        self.send_btn = buttons.addButton("Отправить в локальную модель", QDialogButtonBox.AcceptRole)
        buttons.addButton("Отмена", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.question_edit.textChanged.connect(self._update_length)
        self._update_length()

    def _update_length(self):
        length = len(self.question_edit.toPlainText().strip())
        self.counter.setText(f"{length} / {QUESTION_MAX_CHARS} • пустое поле: разобрать изображение и объяснить по-русски")
        self.send_btn.setEnabled(length <= QUESTION_MAX_CHARS)

    @property
    def question(self):
        return self.question_edit.toPlainText().strip()
