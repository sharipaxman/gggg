"""Проверка автоязыка Whisper без загрузки модели."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

try:
    import numpy as np
    from app.speech_recognizer import SpeechRecognizer
except ImportError:
    np = None


@unittest.skipIf(np is None, "Нужен numpy")
class SpeechRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = SpeechRecognizer()
        self.recognizer._model = MagicMock()
        self.audio = np.array([0, 16384, -16384], dtype=np.int16)

    def test_auto_language_preserves_original_transcript(self):
        for language, text in (("ru", "Почему?"), ("en", "Why?")):
            with self.subTest(language=language):
                self.recognizer._model.transcribe.return_value = (
                    iter([SimpleNamespace(text=f" {text} ")]), SimpleNamespace(language=language),
                )
                result = self.recognizer.recognize_pcm16(self.audio, 16000)
                self.assertEqual((result.text, result.language), (text, language))
                call = self.recognizer._model.transcribe.call_args
                self.assertIsNone(call.kwargs["language"])
                self.assertEqual(call.kwargs["task"], "transcribe")
                np.testing.assert_array_equal(call.args[0], np.array([0, .5, -.5], dtype=np.float32))

    def test_invalid_sample_rate_rejected(self):
        with self.assertRaises(ValueError):
            self.recognizer.recognize_pcm16(self.audio, 48000)
        self.recognizer._model.transcribe.assert_not_called()

    def test_original_text_api_still_works(self):
        self.recognizer._model.transcribe.return_value = (
            [SimpleNamespace(text="Hello")], SimpleNamespace(language="en"),
        )
        self.assertEqual(self.recognizer.transcribe_pcm16(self.audio, 16000), "Hello")


if __name__ == "__main__":
    unittest.main()
