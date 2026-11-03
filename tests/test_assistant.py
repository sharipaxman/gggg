"""Тесты локального ИИ-клиента без Ollama, моделей и сети."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

from app.assistant import Assistance, AnswerSuggestion, ConversationAssistant, is_explicit_reply_request, RESPONSE_SCHEMA
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
        context = [SimpleNamespace(source_language="ru", source_text=f"{i}:" + "x" * 3000, translated_text="private translation")
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
        self.assertTrue(transcript["previous_turns"][0]["source"].startswith(f"{12 - ASSISTANT_CONTEXT_ENTRIES}:"))
        self.assertEqual(len(transcript["previous_turns"][0]["source"]), ASSISTANT_MAX_TEXT_CHARS)
        self.assertEqual(transcript["current"]["source_language"], "en")
        self.assertNotIn("private translation", request.data.decode())
        self.assertNotIn("screenshot", request.data.decode())
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(body["options"]["num_ctx"], 4096)
        self.assertEqual(body["options"]["num_predict"], 384)
        self.assertEqual(self.client._opener.open.call_args.kwargs["timeout"], 30)

    def test_explicit_requests_without_question_mark(self):
        for text in ("Tell me the origin of your name.", "Please tell me about yourself.", "Okay. Tell me about your name.",
                     "Could you please explain your answer.", "Describe your family.",
                     "Расскажи о происхождении своего имени.", "Объясните свой выбор.",
                     "Можешь рассказать о себе."):
            with self.subTest(text=text):
                self.assertTrue(is_explicit_reply_request(text))
        for text in ("My name is Alex.", "I can explain it later.", "The teacher asked me to describe it.",
                     "He said tell me your name.", "Я расскажу о себе.", "Он попросил: расскажи о себе."):
            with self.subTest(text=text):
                self.assertFalse(is_explicit_reply_request(text))

    def test_name_origin_request_forces_answers_in_schema(self):
        answers = Assistance(True, [
            AnswerSuggestion("My name is [name], and it comes from [origin].", "Меня зовут [имя], это имя происходит из [источник]."),
            AnswerSuggestion("I'm not sure about its origin.", "Я не уверен в его происхождении."),
        ])
        self.respond(answers.to_json())
        result = self.client.suggest("Tell me the origin of your name.", "Расскажи о своём имени.", "en", [])
        self.assertEqual(result, answers)
        body = json.loads(self.client._opener.open.call_args.args[0].data)
        self.assertTrue(json.loads(body["messages"][1]["content"])["current"]["reply_required"])
        self.assertTrue(body["format"]["properties"]["is_question"]["const"])
        self.assertEqual(body["format"]["properties"]["suggestions"]["minItems"], 2)
        self.assertNotIn("const", RESPONSE_SCHEMA["properties"]["is_question"])
        self.assertNotIn("minItems", RESPONSE_SCHEMA["properties"]["suggestions"])
        self.client._opener.open.assert_called_once()  # без второго медленного запроса

    def test_request_cannot_be_silently_classified_as_statement(self):
        self.respond('{"is_question":false,"suggestions":[]}')
        with self.assertRaisesRegex(RuntimeError, "Некорректный ответ"):
            self.client.suggest("Tell me the origin of your name.", "Расскажи о своём имени.", "en", [])

    def test_custom_answer_uses_prompt_and_memory_without_question_gate(self):
        self.respond('{"answer":"Сегодня обсуждали поездку."}')
        answer = self.client.answer("I went to Paris.", [{"role": "heard", "text": "We discuss travel."}],
                                    "Отвечай кратко по-русски")
        self.assertEqual(answer, "Сегодня обсуждали поездку.")
        body = json.loads(self.client._opener.open.call_args.args[0].data)
        self.assertIn("Отвечай кратко по-русски", body["messages"][0]["content"])
        data = json.loads(body["messages"][1]["content"])
        self.assertEqual(data["task"], "heard_utterance")
        self.assertEqual(data["memory"][0]["role"], "heard")
        self.assertNotIn("is_question", body["format"]["properties"])
        self.assertNotIn("suggestions", body["format"]["properties"])

    def test_manual_question_is_distinguished_from_heard_speech(self):
        self.respond('{"answer":"Paris"}')
        self.client.answer("Which city?", [], "Be concise", manual=True)
        body = json.loads(self.client._opener.open.call_args.args[0].data)
        self.assertEqual(json.loads(body["messages"][1]["content"])["task"], "manual_question")

    def test_free_answer_validation(self):
        for value in ('{}', '{"answer":" "}', '{"answer":false}'):
            with self.subTest(value=value):
                self.respond(value)
                with self.assertRaisesRegex(RuntimeError, "некорректный"):
                    self.client.answer("Q", [], "")

    def test_free_response_history_round_trip_and_legacy_json(self):
        answer = Assistance(response="Произвольный ответ")
        self.assertEqual(Assistance.from_json(answer.to_json()), answer)
        self.assertEqual(Assistance.from_json('{"is_question":false,"suggestions":[]}'), Assistance())

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

    def test_truncated_generation_reports_token_limit(self):
        response = self.client._opener.open.return_value.__enter__.return_value
        response.read.return_value = json.dumps({
            "done_reason": "length", "message": {"content": "{"},
        }).encode()
        with self.assertRaisesRegex(RuntimeError, "ASSISTANT_NUM_PREDICT"):
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
