"""
Разовая настройка: готовит модель Silero VAD, модель Whisper и языковой пакет
Argos Translate (en↔ru) на локальном диске, чтобы дальше приложение работало
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

    model = load_silero_model(allow_download=True)

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
        local_files_only=False,  # загрузка разрешена только на этапе настройки
    )
    logger.info("Модель Whisper готова (сохранена в %s)", MODELS_DIR)


def setup_argos_translate():
    logger.info("Установка офлайн-пакетов Argos Translate EN ↔ RU...")
    import argostranslate.package as argos_package

    installed = {(p.from_code, p.to_code) for p in argos_package.get_installed_packages()}
    required = [("en", "ru"), ("ru", "en")]
    missing = [pair for pair in required if pair not in installed]
    if not missing:
        logger.info("Оба языковых пакета уже установлены")
        return
    argos_package.update_package_index()
    available = argos_package.get_available_packages()
    for source, target in missing:
        pkg = next((p for p in available if p.from_code == source and p.to_code == target), None)
        if not pkg:
            raise RuntimeError(f"Пакет перевода {source}→{target} не найден в индексе Argos Translate")
        argos_package.install_from_path(pkg.download())
        logger.info("Пакет перевода %s→%s установлен", source, target)


if __name__ == "__main__":
    setup_silero_vad()
    setup_whisper()
    setup_argos_translate()
    logger.info("Готово. Теперь приложение может работать полностью офлайн.")
