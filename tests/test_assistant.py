"""Тесты локального ИИ-клиента без Ollama, моделей и сети."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

from app.assistant import Assistance, AnswerSuggestion, ConversationAssistant
from app.config import ASSISTANT_CONTEXT_ENTRIES, ASSISTANT_MAX_TEXT_CHARS


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.client = ConversationAssistant()
        self.client._opener = MagicMock()
        self.answer = Assistance(True, [AnswerSuggestion("I like reading.", "Я люблю читать.")])

    def respond(self, content):
        response = self.client._opener.open.return_value.__enter__.return_value
        response.read.return_value = json.dumps({"message": {"content": content}}).encode()

    def test_structured_local_request_and_bounded_context(self):
        self.respond(self.answer.to_json())
        context = [SimpleNamespace(source_language="ru", source_text=str(i), translated_text="x" * 3000)
                   for i in range(12)]
        result = self.client.suggest("What do you like?", "Что тебе нравится?", "en", context)
        self.assertEqual(result, self.answer)
        request = self.client._opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        body = json.loads(request.data)
        self.assertFalse(body["stream"])
        self.assertEqual(body["format"]["type"], "object")
        transcript = json.loads(body["messages"][1]["content"])
        self.assertEqual(len(transcript["previous_turns"]), ASSISTANT_CONTEXT_ENTRIES)
        self.assertEqual(transcript["previous_turns"][0]["source"], "6")
        self.assertEqual(len(transcript["previous_turns"][0]["translation"]), ASSISTANT_MAX_TEXT_CHARS)
        self.assertEqual(transcript["current"]["source_language"], "en")
        self.assertNotIn("screenshot", request.data.decode())
        self.assertEqual(self.client._opener.open.call_args.kwargs["timeout"], 30)

    def test_statement_has_no_answers(self):
        self.respond('{"is_question":false,"suggestions":[]}')
        self.assertEqual(self.client.suggest("Hello", "Привет", "en", []), Assistance())

    def test_invalid_response(self):
        for content in ('not json', '[]', '{"is_question":"yes","suggestions":[]}',
                        '{"is_question":true,"suggestions":[]}',
                        '{"is_question":true,"suggestions":[{"en":"OK"}]}'):
            with self.subTest(content=content):
                self.respond(content)
                with self.assertRaisesRegex(RuntimeError, "Некорректный ответ"):
                    self.client.suggest("Почему?", "Why?", "ru", [])

    def test_unavailable_or_timed_out(self):
        for error in (URLError("refused"), TimeoutError()):
            with self.subTest(error=error):
                self.client._opener.open.side_effect = error
                with self.assertRaisesRegex(RuntimeError, "Перевод продолжает работать"):
                    self.client.suggest("Question?", "Вопрос?", "en", [])

    def test_missing_model_has_actionable_message(self):
        self.client._opener.open.side_effect = HTTPError(self.client.url, 404, "not found", {}, None)
        with self.assertRaisesRegex(RuntimeError, "ollama pull"):
            self.client.suggest("Question?", "Вопрос?", "en", [])

    def test_response_size_limit(self):
        self.client._opener.open.return_value.__enter__.return_value.read.return_value = b"x" * (128 * 1024 + 1)
        with self.assertRaises(RuntimeError):
            self.client.suggest("Question?", "Вопрос?", "en", [])

    def test_disallow_cloud_proxy_and_credentials(self):
        for url in ("https://api.example.com", "http://192.168.1.2:11434", "http://localhost.evil.com",
                    "http://user:pass@localhost:11434", "http://localhost/api", "http://localhost?x=y"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ConversationAssistant(url=url)
        for url in ("http://localhost:11434/", "http://127.0.0.1:11434", "http://[::1]:11434"):
            ConversationAssistant(url=url)


if __name__ == "__main__":
    unittest.main()
