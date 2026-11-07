"""Изображения в локальной Ollama и независимый чат: без ML, экрана и сети."""
import base64
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError

from image_fixture import png_image

from app.assistant import ConversationAssistant, AssistantReply
from app.assistant_workspace import AssistantWorkspace, ContextChat, QUESTION_MAX_CHARS
from app.config import ASSISTANT_NUM_CTX, ASSISTANT_VISION_TIMEOUT_SECONDS
from app.image_attachment import validate_png, encode_png, MAX_IMAGE_BYTES, DEFAULT_IMAGE_QUESTION




class VisionTests(unittest.TestCase):
    def setUp(self):
        self.png = png_image()
        self.client = ConversationAssistant()
        self.client._opener = MagicMock()
        self.client._translator = MagicMock()
        self.client._translator.translate.side_effect = ["Мне нравится картинка.", "Мне нравится картинка. На ней кот."]
        self.data = {"explanation_ru": "На картинке кот. Задание просит его описать.", "english_needed": True,
                     "short_answer": "I like this picture.",
                     "long_answer": "I like this picture. It shows a small cat. The cat is on a chair."}

    def respond(self, data):
        envelope = {"message": {"content": json.dumps(data)}}
        self.client._opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(envelope).encode()

    def test_png_limits_and_encoding(self):
        self.assertEqual(base64.b64decode(encode_png(self.png)), self.png)
        for bad in (b"not an image", b"x" * (MAX_IMAGE_BYTES + 1), png_image(2049, 1), png_image(0, 1)):
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                validate_png(bad)

    def test_current_image_only_local_payload_and_two_translated_answers(self):
        self.respond(self.data)
        context = [{"role": "heard", "text": "Describe this picture.", "image_path": "/private/old.png"}]
        reply = self.client.analyze_image(self.png, "Объясни рисунок", context, "Мой уровень A2")
        request = self.client._opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        body = json.loads(request.data)
        self.assertEqual(body["messages"][-1]["content"], "Объясни рисунок")
        self.assertEqual(body["messages"][-1]["images"], [base64.b64encode(self.png).decode()])
        self.assertNotIn("/private/old.png", request.data.decode())
        self.assertTrue(all("images" not in message for message in body["messages"][:-1]))
        self.assertEqual(body["options"]["num_ctx"], ASSISTANT_NUM_CTX)
        self.assertEqual(self.client._opener.open.call_args.kwargs["timeout"], ASSISTANT_VISION_TIMEOUT_SECONDS)
        self.assertEqual(reply.explanation_ru, self.data["explanation_ru"])
        self.assertEqual([v.kind for v in reply.variants], ["short", "long"])
        self.assertTrue(all(v.translation for v in reply.variants))
        self.assertIn("кот", reply.text)
        for call in self.client._translator.translate.call_args_list:
            self.assertEqual(call.args[1:], ("en", "ru"))

    def test_russian_explanation_without_unnecessary_english(self):
        self.respond({"explanation_ru": "Ответ: 4. Складываем два и два.", "english_needed": False,
                      "short_answer": "", "long_answer": ""})
        reply = self.client.analyze_image(self.png, "Реши пример", [], "")
        self.assertEqual(reply.language, "ru")
        self.assertEqual(reply.variants, [])
        self.client._translator.translate.assert_not_called()

    def test_partial_translation_failure_does_not_lose_image_explanation(self):
        self.respond(self.data)
        self.client._translator.translate.side_effect = [RuntimeError("missing package"), "Длинный перевод"]
        reply = self.client.analyze_image(self.png, "Опиши", [], "")
        self.assertTrue(reply.explanation_ru)
        self.assertTrue(reply.variants[0].text)
        self.assertIn("missing package", reply.variants[0].translation_error)
        self.assertEqual(reply.variants[1].translation, "Длинный перевод")

    def test_invalid_output_is_not_shown_as_valid_solution(self):
        invalid = [{}, {**self.data, "english_needed": "yes"}, {**self.data, "explanation_ru": ""},
                   {**self.data, "long_answer": "Too short."},
                   {**self.data, "english_needed": False}, {**self.data, "short_answer": None}]
        for data in invalid:
            with self.subTest(data=data):
                self.respond(data)
                with self.assertRaisesRegex(RuntimeError, "разбор снимка"):
                    self.client.analyze_image(self.png, "Разбери", [], "")

    def test_truncated_vision_response_is_reported_instead_of_partial_answer(self):
        response = self.client._opener.open.return_value.__enter__.return_value
        response.read.return_value = json.dumps({"done_reason": "length", "message": {"content": json.dumps(self.data)}}).encode()
        with self.assertRaisesRegex(RuntimeError, "Разбор снимка обрезан"):
            self.client.analyze_image(self.png, "Реши", [], "")
        self.client._translator.translate.assert_not_called()

    def test_unsupported_model_error_explains_vision_requirement(self):
        self.client._opener.open.side_effect = HTTPError(self.client.url, 400, "no images", {}, None)
        with self.assertRaisesRegex(RuntimeError, "зрительная модель"):
            self.client.analyze_image(self.png, "Разбери", [], "")

    def test_regular_text_does_not_resend_image(self):
        self.respond({"short_answer": "It is a cat.", "long_answer": "It is a cat. The cat is small.", "answer_language": "en"})
        self.client.reply("What is it?", [], "")
        body = json.loads(self.client._opener.open.call_args.args[0].data)
        self.assertTrue(all("images" not in message for message in body["messages"]))

    def test_text_and_vision_share_request_lock(self):
        started, release, text_started = threading.Event(), threading.Event(), threading.Event()
        errors = []
        self.addCleanup(release.set)

        def open_request(request, timeout):
            body = json.loads(request.data)
            is_image = bool(body["messages"][-1].get("images"))
            if is_image:
                started.set()
                release.wait(timeout=3)
                content = {"explanation_ru": "Пример", "english_needed": False, "short_answer": "", "long_answer": ""}
            else:
                content = {"answer_language": "en", "short_answer": "Yes, I do.", "long_answer": "Yes, I do. I like it a lot."}
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({"message": {"content": json.dumps(content)}}).encode()
            return response

        self.client._opener.open.side_effect = open_request

        def run(image):
            try:
                if image:
                    self.client.analyze_image(self.png, "Разбери", [], "")
                else:
                    text_started.set()
                    self.client.reply("Do you like it?", [], "")
            except Exception as exc:
                errors.append(exc)

        vision, text = threading.Thread(target=run, args=(True,)), threading.Thread(target=run, args=(False,))
        vision.start()
        self.assertTrue(started.wait(timeout=2))
        text.start()
        self.assertTrue(text_started.wait(timeout=2))
        try:
            self.assertEqual(self.client._opener.open.call_count, 1)
        finally:
            release.set()
            vision.join(timeout=3)
            text.join(timeout=3)
        self.assertFalse(vision.is_alive() or text.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.client._opener.open.call_count, 2)


class ImageChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = AssistantWorkspace(Path(self.temp.name) / "chat.db")
        self.assistant = MagicMock()
        self.assistant.analyze_image.return_value = AssistantReply("Разбор", language="ru", explanation_ru="Разбор")
        self.directory = Path(self.temp.name) / "shots"
        self.chat = ContextChat(self.assistant, self.workspace, image_directory=self.directory)
        self.addCleanup(self.chat.close)
        self.png = png_image()

    def ask(self, question=""):
        done, results = threading.Event(), []
        self.chat.ask_image(self.png, question, lambda result: (results.append(result), done.set()))
        self.assertTrue(done.wait(timeout=2))
        self.chat._thread.join(timeout=2)
        return results[0]

    def test_image_comment_and_explanation_persist_without_base64_in_memory(self):
        self.workspace.remember("heard", "We are talking about animals.")
        result = self.ask("Объясни упражнение 2")
        self.assertFalse(result.error)
        self.assistant.reply.assert_not_called()
        args = self.assistant.analyze_image.call_args.args
        self.assertEqual(args[:2], (self.png, "Объясни упражнение 2"))
        self.assertIn("animals", args[2][0]["text"])
        events = AssistantWorkspace(self.workspace.db_path).recent(chat_only=True)
        self.assertEqual(Path(events[0]["image_path"]).read_bytes(), self.png)
        self.assertEqual(events[-1]["explanation_ru"], "Разбор")
        self.assertEqual(result.image_path, events[0]["image_path"])
        self.assertNotIn("image_path", self.workspace.snapshot()[1][-1])
        self.assertNotIn(base64.b64encode(self.png).decode(), json.dumps(events))

    def test_empty_comment_gets_default_question(self):
        self.ask()
        self.assertEqual(self.assistant.analyze_image.call_args.args[1], DEFAULT_IMAGE_QUESTION)

    def test_invalid_or_overlong_input_does_not_write_or_send(self):
        with self.assertRaises(ValueError):
            self.chat.ask_image(b"bad", "", lambda _: None)
        with self.assertRaises(ValueError):
            self.chat.ask_image(self.png, "x" * (QUESTION_MAX_CHARS + 1), lambda _: None)
        self.assertFalse(self.directory.exists())
        self.assertEqual(self.workspace.recent(), [])
        self.assistant.analyze_image.assert_not_called()

    def test_busy_does_not_save_second_image_and_audio_memory_can_continue(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def analyze(*args):
            started.set()
            release.wait(timeout=3)
            return AssistantReply("Разбор", language="ru", explanation_ru="Разбор")

        self.assistant.analyze_image.side_effect = analyze
        self.chat.ask_image(self.png, "Первый", lambda _: None)
        self.assertTrue(started.wait(timeout=2))
        try:
            with self.assertRaises(RuntimeError):
                self.chat.ask_image(self.png, "Второй", lambda _: None)
            self.workspace.remember("heard", "New audio is still processed.")
            self.assertIn("New audio", self.workspace.recent()[-1]["text"])
            self.assertEqual(len(list(self.directory.glob("*.png"))), 1)
        finally:
            release.set()
            self.chat._thread.join(timeout=2)

    def test_clear_and_close_suppress_stale_results(self):
        def clear(*args):
            self.workspace.clear_memory()
            return AssistantReply("Old result")
        self.assistant.analyze_image.side_effect = clear
        result = self.ask("Первый")
        self.assertIn("очищена", result.error)
        self.assertEqual(self.workspace.recent(), [])
        self.chat.close()
        with self.assertRaises(RuntimeError):
            self.chat.ask_image(self.png, "Закрыто", lambda _: None)


if __name__ == "__main__":
    unittest.main()
