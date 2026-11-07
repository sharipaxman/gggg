"""Ограниченное PNG-вложение для локальной Ollama, без зависимости от Qt."""
from __future__ import annotations

import base64
import struct
import uuid
from pathlib import Path

from app.config import SCREENSHOTS_DIR

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 2048
DEFAULT_IMAGE_QUESTION = "Объясни, что видно на изображении. Если есть задание, помоги его выполнить. Ответь по-русски."


def validate_png(data: bytes):
    if not isinstance(data, bytes) or not 33 <= len(data) <= MAX_IMAGE_BYTES:
        raise ValueError("Нужен PNG-снимок размером не более 8 МБ")
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[8:16] != b"\x00\x00\x00\rIHDR":
        raise ValueError("Вложение должно быть PNG-изображением")
    width, height = struct.unpack(">II", data[16:24])
    if not (1 <= width <= MAX_IMAGE_SIDE and 1 <= height <= MAX_IMAGE_SIDE):
        raise ValueError(f"Сторона снимка должна быть от 1 до {MAX_IMAGE_SIDE} пикселей")


def encode_png(data: bytes) -> str:
    validate_png(data)
    return base64.b64encode(data).decode("ascii")


def save_png(data: bytes, directory: Path = SCREENSHOTS_DIR) -> Path:
    validate_png(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"vision_{uuid.uuid4().hex}.png"
    path.write_bytes(data)
    return path
