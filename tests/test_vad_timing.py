"""Порог паузы на реальной логике сегментатора, без PyTorch/Silero.

Только вычисление вероятности речи подменяется на детерминированное.
Изолированная загрузка не меняет sys.modules для других тестов VAD.
"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import numpy as np
except ImportError:
    np = None

from app.config import END_OF_SPEECH_SILENCE_MS, FRAME_MS, VAD_WINDOW_SAMPLES


@unittest.skipIf(np is None, "Нужен numpy")
class VadTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = "vad_timing_test_module"
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "app/vad_segmenter.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: module, "torch": MagicMock()}):
            spec.loader.exec_module(module)
        cls.segmenter_class = module.VadSegmenter

    def setUp(self):
        self.results = []
        self.segmenter = self.segmenter_class(self.results.append, model=SimpleNamespace(reset_states=MagicMock()))
        self.segmenter._speech_probability = lambda window: 0.95 if np.any(window) else 0.01
        self.speech = np.ones(VAD_WINDOW_SAMPLES, dtype=np.int16)
        self.silence = np.zeros(VAD_WINDOW_SAMPLES, dtype=np.int16)

    def feed(self, frame, count):
        for _ in range(count):
            self.segmenter._process_frame(frame)

    def test_threshold_is_never_rounded_below_1500ms(self):
        self.assertEqual(END_OF_SPEECH_SILENCE_MS, 1500)
        frames = self.segmenter._silence_frames_needed
        self.assertEqual(frames, math.ceil(1500 / FRAME_MS))
        self.assertGreaterEqual(frames * FRAME_MS, 1500)
        self.assertLess((frames - 1) * FRAME_MS, 1500)
        self.assertEqual(frames * FRAME_MS, 1504)

    def test_phrase_waits_1504ms_instead_of_ending_after_one_second(self):
        self.feed(self.speech, 30)
        self.feed(self.silence, 31)  # 992 мс
        self.assertEqual(self.results, [])
        self.feed(self.silence, 15)  # 1472 мс суммарно
        self.assertEqual(self.results, [])
        self.feed(self.silence, 1)   # 1504 мс
        self.assertEqual(len(self.results), 1)
        self.assertEqual(self.results[0].duration_ms, 30 * FRAME_MS)  # хвост тишины отрезан

    def test_short_pause_then_resumed_speech_remains_one_phrase(self):
        self.feed(self.speech, 30)
        self.feed(self.silence, 31)
        self.feed(self.speech, 30)
        self.assertEqual(self.results, [])
        self.feed(self.silence, 47)
        self.assertEqual(len(self.results), 1)
        self.assertEqual(self.results[0].duration_ms, (30 + 31 + 30) * FRAME_MS)


if __name__ == "__main__":
    unittest.main()
