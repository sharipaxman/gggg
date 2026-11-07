"""Qt-снимки: геометрия/DPI и предпросмотр, без реального захвата рабочего стола."""
import os
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import Qt, QRect, QPoint
    from PySide6.QtGui import QImage, QPixmap, QColor
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from app.ui.screenshot_dialog import physical_crop, RegionSelectionDialog, ScreenshotQuestionDialog
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "Нужны PySide6 и системные библиотеки Qt")
class ScreenshotDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_crop_scales_logical_coordinates_to_fractional_dpi_pixels(self):
        self.assertEqual(physical_crop(QRect(10, 20, 100, 50), 200, 100, 300, 150), QRect(15, 30, 150, 75))
        self.assertEqual(physical_crop(QRect(0, 0, 200, 100), 200, 100, 400, 200), QRect(0, 0, 400, 200))

    def test_preview_prepares_bounded_png_and_checks_comment_length(self):
        image = QImage(4000, 1000, QImage.Format_RGB32)
        image.fill(QColor("white"))
        dialog = ScreenshotQuestionDialog(image)
        self.addCleanup(dialog.close)
        self.assertEqual(struct.unpack(">II", dialog.image_png[16:24]), (2048, 512))
        self.assertEqual(dialog.question, "")
        self.assertTrue(dialog.send_btn.isEnabled())
        dialog.question_edit.setPlainText("x" * 1001)
        self.assertFalse(dialog.send_btn.isEnabled())
        dialog.question_edit.setPlainText("  Объясни картинку  ")
        self.assertEqual(dialog.question, "Объясни картинку")
        self.assertTrue(dialog.send_btn.isEnabled())

    def test_selection_on_negative_origin_monitor_crops_source_not_overlay(self):
        image = QImage(400, 200, QImage.Format_RGB32)
        image.fill(QColor("red"))
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(2.0)
        screen = SimpleNamespace(grabWindow=lambda _: pixmap, geometry=lambda: QRect(-200, 0, 200, 100))
        with patch("app.ui.screenshot_dialog.QGuiApplication") as gui:
            gui.screenAt.return_value = screen
            dialog = RegionSelectionDialog()
        self.addCleanup(dialog.close)
        QTest.mousePress(dialog, Qt.LeftButton, pos=QPoint(10, 20))
        QTest.mouseRelease(dialog, Qt.LeftButton, pos=QPoint(110, 70))
        self.assertEqual(dialog.result(), dialog.Accepted)
        self.assertEqual((dialog.selected_image.width(), dialog.selected_image.height()), (202, 102))
        self.assertEqual(dialog.selected_image.pixelColor(0, 0), QColor("red"))


if __name__ == "__main__":
    unittest.main()
