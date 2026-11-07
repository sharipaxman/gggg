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
SAMPLE_RATE = 16000          # Гц, обязателен для Silero VAD и whisper
# Silero VAD принимает окна строго фиксированного размера:
# 512 сэмплов при 16 кГц (= 32 мс) и 256 при 8 кГц. Отсюда FRAME_MS = 32.
FRAME_MS = 32
VAD_WINDOW_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 512 сэмплов

# Silero возвращает вероятность речи 0.0..1.0 (вместо бинарного ответа webrtcvad).
VAD_THRESHOLD = 0.5          # выше этого значения окно считается речью (старт фразы)
# Порог выхода из речи (гистерезис): пока мы внутри фразы, тишиной считается
# только падение вероятности ниже этого значения — так фраза не рвётся
# на тихих словах и согласных. Рекомендация авторов Silero: threshold - 0.15.
VAD_NEG_THRESHOLD = 0.35
VAD_USE_ONNX = False         # True — использовать onnx-версию (нужен пакет onnxruntime)

# Кеш torch.hub на случай, если модель грузится не из пакета silero-vad, а с GitHub
TORCH_HUB_DIR = MODELS_DIR / "torch_hub"
# Модель VAD крошечная (~1 МБ): многопоточность torch только мешает и ест CPU
TORCH_NUM_THREADS = 1

# Сколько тишины подряд считать концом фразы ("человек закончил говорить")
END_OF_SPEECH_SILENCE_MS = 1500
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
WHISPER_LANGUAGE = None       # auto: определять EN/RU для каждой фразы

# ---------------------------------------------------------------------------
# Перевод (argostranslate, офлайн-пакеты EN <-> RU)
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
HOTKEY_ANALYZE_SCREENSHOT = "ctrl+alt+s"

# ---------------------------------------------------------------------------
# Локальный ИИ-помощник (Ollama). Включается отдельно в интерфейсе.
# Разрешён только loopback: текст не должен незаметно уйти в облако.
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("LESSON_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("LESSON_OLLAMA_MODEL", "qwen3-vl:4b")
ASSISTANT_TIMEOUT_SECONDS = 60
ASSISTANT_VISION_TIMEOUT_SECONDS = 180
ASSISTANT_CONTEXT_ENTRIES = 4
ASSISTANT_MAX_TEXT_CHARS = 800

# Одна локальная VL-модель для текста и снимков. Общее окно не требует
# смены размера KV-кеша между текстовым и визуальным запросами.
ASSISTANT_NUM_CTX = 8192
ASSISTANT_NUM_PREDICT = 384
ASSISTANT_KEEP_ALIVE = "30m"  # не выгружать модель между репликами (занимает RAM/VRAM)
