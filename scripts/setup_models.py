"""
Разовая настройка: готовит модель Silero VAD, модель Whisper и языковой пакет
Argos Translate (en->ru) на локальном диске, чтобы дальше приложение работало
полностью офлайн.

Запуск:  python scripts/setup_models.py
Интернет нужен только для этого разового запуска.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def setup_silero_vad():
    """Готовит детектор речи Silero VAD и прогревает его одним прогоном.

    Если установлен pip-пакет `silero-vad`, модель (~1 МБ) уже лежит внутри
    пакета и интернет не нужен совсем. Иначе модель один раз скачается через
    torch.hub в data/models/torch_hub.
    """
    logger.info("Подготовка Silero VAD...")
    import numpy as np
    import torch

    from app.config import SAMPLE_RATE, VAD_WINDOW_SAMPLES
    from app.vad_segmenter import load_silero_model

    model = load_silero_model()

    # прогрев: один прогон тишины, чтобы убедиться, что модель реально работает
    silence = torch.zeros(VAD_WINDOW_SAMPLES, dtype=torch.float32)
    with torch.no_grad():
        probability = float(model(silence, SAMPLE_RATE).item())
    if hasattr(model, "reset_states"):
        model.reset_states()

    logger.info(
        "Silero VAD готов (окно %d сэмплов, вероятность речи на тишине %.4f)",
        VAD_WINDOW_SAMPLES,
        probability,
    )


def setup_whisper():
    logger.info("Скачивание модели Whisper '%s'...", WHISPER_MODEL_SIZE)
    from faster_whisper import WhisperModel

    WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=str(MODELS_DIR),
    )
    logger.info("Модель Whisper готова (сохранена в %s)", MODELS_DIR)


def setup_argos_translate():
    logger.info("Установка офлайн-пакета перевода Argos Translate en->ru...")
    import argostranslate.package as argos_package
    import argostranslate.translate as argos_translate

    installed_languages = argos_translate.get_installed_languages()
    has_en_ru = any(
        l.code == "en" and any(t.to_lang.code == "ru" for t in getattr(l, "translations_from", []))
        for l in installed_languages
    )
    if has_en_ru:
        logger.info("Пакет en->ru уже установлен")
        return

    argos_package.update_package_index()
    available = argos_package.get_available_packages()
    pkg = next((p for p in available if p.from_code == "en" and p.to_code == "ru"), None)
    if not pkg:
        raise RuntimeError("Пакет перевода en->ru не найден в индексе Argos Translate")

    path = pkg.download()
    argos_package.install_from_path(path)
    logger.info("Пакет перевода en->ru установлен")


if __name__ == "__main__":
    setup_silero_vad()
    setup_whisper()
    setup_argos_translate()
    logger.info("Готово. Теперь приложение может работать полностью офлайн.")
