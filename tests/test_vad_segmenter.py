"""
Тесты сегментатора фраз на Silero VAD.

Настоящая нейросеть в тестах не нужна: вместо неё подставляется заглушка,
повторяющая её интерфейс (`model(tensor, sample_rate) -> вероятность речи`
и `model.reset_states()`). Так тесты проверяют именно логику "дождаться
конца фразы", работают за доли секунды и не лезут в интернет.

Запуск:  python -m unittest discover -s tests
         (или просто: pytest)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (  # noqa: E402
    FRAME_MS,
    VAD_WINDOW_SAMPLES,
    END_OF_SPEECH_SILENCE_MS,
    MIN_UTTERANCE_MS,
    MAX_UTTERANCE_MS,
)
from app.vad_segmenter import VadSegmenter, Utterance  # noqa: E402

SILENCE_FRAMES = max(1, int(END_OF_SPEECH_SILENCE_MS / FRAME_MS))
MIN_SPEECH_FRAMES = max(1, int(MIN_UTTERANCE_MS / FRAME_MS))
MAX_SPEECH_FRAMES = max(1, int(MAX_UTTERANCE_MS / FRAME_MS))


class LoudnessModel:
    """Заглушка Silero: громкое окно = речь, тихое = тишина."""

    def __init__(self):
        self.calls = 0
        self.reset_calls = 0
        self.window_sizes: list[int] = []

    def __call__(self, tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self.calls += 1
        self.window_sizes.append(int(tensor.shape[-1]))
        rms = float(torch.sqrt(torch.mean(tensor ** 2)))
        return torch.tensor(0.95 if rms > 0.05 else 0.01)

    def reset_states(self):
        self.reset_calls += 1


class ScriptedModel:
    """Заглушка Silero, возвращающая заранее заданную последовательность вероятностей."""

    def __init__(self, probabilities: list[float]):
        self.probabilities = list(probabilities)
        self.index = 0
        self.reset_calls = 0

    def __call__(self, tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        value = self.probabilities[min(self.index, len(self.probabilities) - 1)]
        self.index += 1
        return torch.tensor(value)

    def reset_states(self):
        self.reset_calls += 1


def loud_frame(samples: int = VAD_WINDOW_SAMPLES) -> np.ndarray:
    """Окно «речи»: достаточно громкий сигнал."""
    t = np.arange(samples)
    return (8000 * np.sin(2 * np.pi * 150 * t / 16000)).astype(np.int16)


def quiet_frame(samples: int = VAD_WINDOW_SAMPLES) -> np.ndarray:
    """Окно «тишины»: почти нули."""
    return np.zeros(samples, dtype=np.int16)


class VadSegmenterTestCase(unittest.TestCase):
    def setUp(self):
        self.utterances: list[Utterance] = []

    def make_segmenter(self, model) -> VadSegmenter:
        return VadSegmenter(on_utterance=self.utterances.append, model=model)

    def feed(self, segmenter: VadSegmenter, frames: list[np.ndarray]):
        for frame in frames:
            segmenter._process_frame(frame)

    # ------------------------------------------------------------------
    def test_phrase_is_emitted_only_after_pause(self):
        """Фраза отдаётся на распознавание только когда наступила пауза."""
        model = LoudnessModel()
        segmenter = self.make_segmenter(model)

        speech_frames = 40  # ~1.3 с речи
        self.feed(segmenter, [loud_frame() for _ in range(speech_frames)])
        self.assertEqual(self.utterances, [], "фраза не должна отдаваться, пока человек говорит")

        # пауза короче порога — фраза ещё не закончена
        self.feed(segmenter, [quiet_frame() for _ in range(SILENCE_FRAMES - 1)])
        self.assertEqual(self.utterances, [], "короткая пауза (вдох) не должна завершать фразу")

        # добираем паузу до порога — фраза завершается
        self.feed(segmenter, [quiet_frame()])
        self.assertEqual(len(self.utterances), 1)

        utterance = self.utterances[0]
        self.assertEqual(utterance.duration_ms, speech_frames * FRAME_MS)
        self.assertEqual(len(utterance.pcm_int16), speech_frames * VAD_WINDOW_SAMPLES)
        self.assertEqual(utterance.pcm_int16.dtype, np.int16)

    def test_short_blip_is_discarded(self):
        """Короткий щелчок/вдох короче MIN_UTTERANCE_MS фразой не считается."""
        model = LoudnessModel()
        segmenter = self.make_segmenter(model)

        self.feed(segmenter, [loud_frame() for _ in range(MIN_SPEECH_FRAMES - 2)])
        self.feed(segmenter, [quiet_frame() for _ in range(SILENCE_FRAMES)])

        self.assertEqual(self.utterances, [])

    def test_two_phrases_are_separated(self):
        """Две фразы с паузой между ними дают ровно две фразы."""
        model = LoudnessModel()
        segmenter = self.make_segmenter(model)

        for _ in range(2):
            self.feed(segmenter, [loud_frame() for _ in range(30)])
            self.feed(segmenter, [quiet_frame() for _ in range(SILENCE_FRAMES)])

        self.assertEqual(len(self.utterances), 2)
        for utterance in self.utterances:
            self.assertEqual(utterance.duration_ms, 30 * FRAME_MS)

    def test_long_speech_is_force_split(self):
        """Речь без пауз форсированно режется по MAX_UTTERANCE_MS."""
        model = LoudnessModel()
        segmenter = self.make_segmenter(model)

        self.feed(segmenter, [loud_frame() for _ in range(MAX_SPEECH_FRAMES + 5)])

        self.assertEqual(len(self.utterances), 1)
        self.assertEqual(self.utterances[0].duration_ms, MAX_SPEECH_FRAMES * FRAME_MS)

    def test_model_state_is_reset_between_phrases(self):
        """Silero рекуррентная: после каждой фразы состояние сбрасывается."""
        model = LoudnessModel()
        segmenter = self.make_segmenter(model)

        self.feed(segmenter, [loud_frame() for _ in range(30)])
        self.feed(segmenter, [quiet_frame() for _ in range(SILENCE_FRAMES)])

        self.assertEqual(len(self.utterances), 1)
        self.assertGreaterEqual(model.reset_calls, 1)

    def test_frames_are_rechunked_to_silero_window(self):
        """Кадры любого размера нарезаются ровно по 512 сэмплов — Silero других не принимает."""
        model = LoudnessModel()
        segmenter = self.make_segmenter(model)

        # «неудобные» размеры кадров: 480 (как было у webrtcvad), 1000, 37
        for samples in (480, 1000, 37):
            self.feed(segmenter, [loud_frame(samples) for _ in range(20)])
            self.feed(segmenter, [quiet_frame(samples) for _ in range(SILENCE_FRAMES * 3)])

        self.assertTrue(model.window_sizes, "модель должна была получить хотя бы одно окно")
        self.assertEqual(set(model.window_sizes), {VAD_WINDOW_SAMPLES})

    def test_hysteresis_keeps_quiet_words_inside_phrase(self):
        """Вероятность между порогами не рвёт фразу: тишина — только ниже VAD_NEG_THRESHOLD."""
        # 10 окон уверенной речи, 10 окон «серой зоны» (0.4), снова речь, затем пауза
        probabilities = (
            [0.9] * 10
            + [0.4] * 10
            + [0.9] * 10
            + [0.01] * (SILENCE_FRAMES + 5)
        )
        model = ScriptedModel(probabilities)
        segmenter = self.make_segmenter(model)

        self.feed(segmenter, [quiet_frame() for _ in range(len(probabilities))])

        self.assertEqual(len(self.utterances), 1, "серая зона не должна разрывать фразу на две")
        self.assertEqual(self.utterances[0].duration_ms, 30 * FRAME_MS)

    def test_pre_speech_padding_is_included(self):
        """В начало фразы добавляется запас аудио, чтобы не отрезать первый звук."""
        model = ScriptedModel([0.01] * 20 + [0.9] * 30 + [0.01] * (SILENCE_FRAMES + 2))
        segmenter = self.make_segmenter(model)

        self.feed(segmenter, [quiet_frame() for _ in range(20 + 30 + SILENCE_FRAMES + 2)])

        self.assertEqual(len(self.utterances), 1)
        # 30 окон речи + несколько окон предзаписи
        self.assertGreater(self.utterances[0].duration_ms, 30 * FRAME_MS)


class SileroModelIntegrationTestCase(unittest.TestCase):
    """Проверка настоящей модели — пропускается, если пакет silero-vad не установлен."""

    def test_real_model_scores_silence_as_non_speech(self):
        try:
            from silero_vad import load_silero_vad
        except ImportError:
            self.skipTest("пакет silero-vad не установлен")

        from app.config import SAMPLE_RATE, VAD_THRESHOLD

        model = load_silero_vad()
        silence = torch.zeros(VAD_WINDOW_SAMPLES, dtype=torch.float32)
        with torch.no_grad():
            probability = float(model(silence, SAMPLE_RATE).item())

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertLess(probability, VAD_THRESHOLD, "тишина не должна считаться речью")


if __name__ == "__main__":
    unittest.main()
