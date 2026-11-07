"""Исключение собственного HWND из системного захвата Windows.

Аналог идеи Electron setContentProtection(true) для Qt. Это не гарантия
невидимости во всех рекордерах: проверяется только состояние Windows API.
Не скрывает процесс, панель задач или отдельные окна/диалоги приложения.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011
MIN_CAPTURE_EXCLUSION_BUILD = 19041  # Windows 10 version 2004


@dataclass(frozen=True)
class CaptureProtectionResult:
    applied: bool
    enabled: bool | None
    message: str


def capture_protection_support() -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "Исключение окна из захвата доступно только в Windows 10 2004+ / Windows 11."
    version = sys.getwindowsversion()
    if version.major < 10 or (version.major == 10 and version.build < MIN_CAPTURE_EXCLUSION_BUILD):
        return False, "Нужна Windows 10 2004 (сборка 19041) или новее. Старый режим с чёрным окном не используется."
    return True, "Windows поддерживает исключение собственного окна из системного захвата."


def _load_user32():
    # HWND — pointer-sized: c_int обрезал бы 64-битный дескриптор.
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
    return user32


def _last_error() -> int:
    return getattr(ctypes, "get_last_error", lambda: 0)()


def set_capture_exclusion(hwnd: int, enabled: bool) -> CaptureProtectionResult:
    supported, reason = capture_protection_support()
    if not supported:
        return CaptureProtectionResult(False, False, reason)
    if not hwnd or hwnd < 0:
        return CaptureProtectionResult(False, None, "Окно ещё не имеет действительного HWND.")
    try:
        user32 = _load_user32()
        affinity = WDA_EXCLUDEFROMCAPTURE if enabled else WDA_NONE
        written = bool(user32.SetWindowDisplayAffinity(hwnd, affinity))
        write_error = _last_error() if not written else 0
        actual = wintypes.DWORD()
        if not user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(actual)):
            return CaptureProtectionResult(
                False, None, "Не удалось подтвердить режим захвата окна "
                f"(ошибка Windows {_last_error()}). Не полагайтесь на исключение из демонстрации.",
            )
        protected = actual.value == WDA_EXCLUDEFROMCAPTURE
        if not written:
            return CaptureProtectionResult(
                False, protected, f"Windows не смогла изменить режим захвата (ошибка {write_error}). "
                "Проверьте результат в программе демонстрации.",
            )
        if actual.value != affinity:
            return CaptureProtectionResult(
                False, protected, "Windows не подтвердила запрошенный режим. "
                "Не полагайтесь на исключение из демонстрации.",
            )
        message = ("Windows подтвердила исключение основного окна из системного захвата. "
                   "Проверьте в своей программе демонстрации; отдельные диалоги не защищены."
                   if enabled else "Исключение из захвата выключено: основное окно можно захватывать.")
        return CaptureProtectionResult(True, protected, message)
    except (OSError, AttributeError, TypeError, ValueError) as exc:
        return CaptureProtectionResult(False, None, f"Не удалось настроить исключение из захвата: {exc}")
