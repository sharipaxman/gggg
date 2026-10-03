"""
Централизованная конфигурация приложения.
Все пути — относительно папки проекта, база работает полностью офлайн.
"""
import os
from pathlib import Path

APP_NAME = "EN Lesson Translator"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "history.db"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
MODELS_DIR = DATA_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Аудио / VAD
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Гц, обязателен для webrtcvad и whisper
FRAME_MS = 30                # длина кадра VAD: 10, 20 или 30 мс
VAD_AGGRESSIVENESS = 2       # 0 (мягкий) .. 3 (жёсткий фильтр не-речи)

# Сколько тишины подряд считать концом фразы ("человек закончил говорить")
END_OF_SPEECH_SILENCE_MS = 700
# Минимальная длительность речи, чтобы не реагировать на короткие шумы/вдохи
MIN_UTTERANCE_MS = 400
# Защита от зависания на непрерывной речи без пауз — форсируем разбиение
MAX_UTTERANCE_MS = 18000
# Небольшой запас аудио до начала речи (чтобы не отрезать первый звук фразы)
PRE_SPEECH_PADDING_MS = 300

# ---------------------------------------------------------------------------
# Распознавание речи (faster-whisper, полностью локально)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = "small"       # tiny/base/small/medium — баланс скорости/точности
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"      # быстрее на CPU
WHISPER_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Перевод (argostranslate, офлайн-пакет языков EN->RU)
# ---------------------------------------------------------------------------
TRANSLATE_FROM = "en"
TRANSLATE_TO = "ru"

# ---------------------------------------------------------------------------
# История
# ---------------------------------------------------------------------------
HISTORY_LIMIT = 50

# ---------------------------------------------------------------------------
# Горячие клавиши
# ---------------------------------------------------------------------------
HOTKEY_ATTACH_SCREENSHOT = "ctrl+shift+s"
HOTKEY_TOGGLE_LISTEN = "ctrl+shift+l"
