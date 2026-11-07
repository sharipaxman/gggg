"""
Точка входа.

Запуск:  python main.py
Перед первым запуском выполните разовую настройку: python scripts/setup_models.py
"""
import logging
import os
import sys

# Только запуск приложения: скрипт setup_models.py остаётся отдельным онлайн-шагом.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ARGOS_MODEL_PROVIDER"] = "OPENNMT"

from PySide6.QtWidgets import QApplication

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("EN Lesson Translator")

    from app.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
