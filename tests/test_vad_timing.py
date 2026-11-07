"""Порог паузы на реальной логике сегментатора, без PyTorch/Silero.

Только вычисление вероятности речи подменяется на детерминированное.
Изолированная загрузка не меняет sys.modules для других тестов VAD.
"""
import importlib.util
import math
import sys
import unittest
import tempfile
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
        cls.module = module

    def setUp(self):
        self.results = []
        self.segmenter = self.segmenter_class(self.results.append, model=SimpleNamespace(reset_states=MagicMock()))
        self.segmenter._speech_probability = lambda window: 0.95 if np.any(window) else 0.01
        self.speech = np.ones(VAD_WINDOW_SAMPLES, dtype=np.int16)
        self.silence = np.zeros(VAD_WINDOW_SAMPLES, dtype=np.int16)

    def feed(self, frame, count):
        for _ in range(count):
            self.segmenter._process_frame(frame)

    def test_runtime_does_not_download_missing_silero(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(self.module, "TORCH_HUB_DIR", Path(temp)), \
                patch.dict(sys.modules, {"silero_vad": None}):
            self.module.torch.hub.load.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "setup_models.py"):
                self.module.load_silero_model()
            self.module.torch.hub.load.assert_not_called()

    def test_torch_hub_cache_uses_correct_root_without_network(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(self.module, "TORCH_HUB_DIR", Path(temp)), \
                patch.dict(sys.modules, {"silero_vad": None}):
            cached = Path(temp) / "snakers4_silero-vad_master"
            cached.mkdir()
            model = object()
            self.module.torch.hub.load.reset_mock()
            self.module.torch.hub.load.return_value = model, []
            self.assertIs(self.module.load_silero_model(), model)
            self.module.torch.hub.load.assert_called_once_with(
                repo_or_dir=str(cached), model="silero_vad", source="local", onnx=False)

    def test_only_setup_can_explicitly_allow_download(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(self.module, "TORCH_HUB_DIR", Path(temp)), \
                patch.dict(sys.modules, {"silero_vad": None}):
            self.module.torch.hub.load.reset_mock()
            self.module.torch.hub.load.return_value = object(), []
            self.module.load_silero_model(allow_download=True)
            self.assertEqual(self.module.torch.hub.load.call_args.kwargs["repo_or_dir"],
                             self.module.SILERO_HUB_REPO)

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
