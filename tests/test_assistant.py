"""Тесты локального ИИ-клиента без Ollama, моделей и сети."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call
from urllib.error import HTTPError, URLError

from app.assistant import Assistance, AnswerSuggestion, ConversationAssistant, is_explicit_reply_request, RESPONSE_SCHEMA, AssistantReply, resolve_task, ReplyVariant, parse_variants, learner_name_fact
from app.config import ASSISTANT_CONTEXT_ENTRIES, ASSISTANT_MAX_TEXT_CHARS


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.client = ConversationAssistant()
        self.client._opener = MagicMock()
        self.answer = Assistance(True, [AnswerSuggestion("I like reading.", "Я люблю читать."),
                                        AnswerSuggestion("I like reading books. I often read at home.", "Я люблю читать книги. Я часто читаю дома.")])

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
        data = json.loads(body["messages"][-1]["content"])
        self.assertEqual(data["task"], "heard_utterance")
        self.assertIn("TRANSCRIPT", body["messages"][1]["content"])
        self.assertNotIn("is_question", body["format"]["properties"])
        self.assertNotIn("suggestions", body["format"]["properties"])

    def test_manual_question_is_distinguished_from_heard_speech(self):
        self.respond('{"answer":"Paris"}')
        self.client.answer("Which city?", [], "Be concise", manual=True)
        body = json.loads(self.client._opener.open.call_args.args[0].data)
        self.assertEqual(json.loads(body["messages"][-1]["content"])["task"], "manual_question")

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

    def test_arthur_clarification_completes_teacher_task_not_generic_name_discussion(self):
        task = "Tell me about the origin of your name."
        generic = "Your name origins can vary widely. If you could provide more details about your name..."
        context = [{"role": "heard", "text": task}, {"role": "suggestion", "text": generic},
                   {"role": "user", "text": "мое имя артур"}, {"role": "assistant", "text": generic}]
        self.client._translator = MagicMock()
        self.client._translator.translate.return_value = "Меня зовут Артур. Точное происхождение имени спорно."
        short = "My name is Arthur. People know this name from stories about King Arthur."
        long = "My name is Arthur. It is an old name. Many people know it from stories about King Arthur. People are not sure where the name first came from."
        self.respond(json.dumps({"short_answer": short, "long_answer": long, "answer_language": "en"}))
        reply = self.client.reply("мое имя артур", context, "Помоги мне ответить преподавателю", manual=True)
        body = json.loads(self.client._opener.open.call_args.args[0].data)
        request = json.loads(body["messages"][-1]["content"])
        self.assertEqual(request["task"], "complete_previous_task")
        self.assertEqual(request["task_to_answer"], task)
        self.assertEqual(request["learner_name"], "артур")
        self.assertEqual(request["learner_facts"], ["мое имя артур"])
        self.assertEqual([m["role"] for m in body["messages"]],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertIn("first person", body["messages"][0]["content"])
        self.assertIn("Do not ask for the name again", body["messages"][0]["content"])
        self.assertIn("uncertain", body["messages"][0]["content"])
        self.assertIn("answer_language", body["format"]["required"])
        self.client._translator.translate.assert_has_calls([call(short, "en", "ru"), call(long, "en", "ru")])
        self.assertEqual(reply.variants[0].translation, "Меня зовут Артур. Точное происхождение имени спорно.")
        self.assertEqual([v.kind for v in reply.variants], ["short", "long"])
        self.assertEqual(reply.language, "en")
        self.assertEqual(Assistance.from_json(Assistance.from_reply(reply).to_json()), Assistance.from_reply(reply))

    def test_clarification_resolution_is_not_hardcoded_to_arthur(self):
        context = [{"role": "heard", "text": "Tell me about your name."}]
        for text, name in (("моё имя Анна", "Анна"), ("меня зовут Илья", "Илья"),
                           ("My name is Alice.", "Alice")):
            with self.subTest(text=text):
                resolved = resolve_task(text, context, True)
                self.assertEqual(resolved["learner_name"], name)
                self.assertEqual(resolved["task"], "complete_previous_task")
        city = resolve_task("я живу в Казани", [{"role": "heard", "text": "Where do you live?"}], True)
        self.assertEqual(city["task_to_answer"], "Where do you live?")
        self.assertEqual(city["learner_facts"], ["я живу в Казани"])

    def test_latest_name_correction_wins_even_if_it_repeats_an_older_name(self):
        context = [{"role": "heard", "text": "Tell me about your name."},
                   {"role": "user", "text": "мое имя Артур"}, {"role": "user", "text": "мое имя Анна"}]
        resolved = resolve_task("мое имя Артур", context, True)
        self.assertEqual(resolved["learner_name"], "Артур")
        self.assertEqual(resolved["learner_facts"][-1], "мое имя Артур")

    def test_new_question_is_not_replaced_with_previous_task(self):
        context = [{"role": "heard", "text": "Tell me about your name."}]
        for question in ("Объясни Present Perfect", "Сколько будет два плюс два?"):
            resolved = resolve_task(question, context, True)
            self.assertEqual(resolved["task"], "manual_question")
            self.assertEqual(resolved["task_to_answer"], question)

    def test_ai_drafts_are_not_learner_facts(self):
        context = [{"role": "assistant", "text": "My name is Invented."},
                   {"role": "suggestion", "text": "мое имя Придуманное"}]
        resolved = resolve_task("Что мы обсуждали?", context, True)
        self.assertEqual(resolved["learner_facts"], [])
        self.assertEqual(resolved["learner_name"], "")

    def test_translation_direction_ru_to_en_and_separate_output(self):
        self.client._translator = MagicMock()
        self.client._translator.translate.return_value = "My name is Arthur."
        self.respond('{"short_answer":"Меня зовут Артур.","long_answer":"Меня зовут Артур. Это старое имя, известное по историям о короле Артуре.","answer_language":"ru"}')
        reply = self.client.reply("Q", [], "Отвечай на русском")
        self.client._translator.translate.assert_any_call("Меня зовут Артур.", "ru", "en")
        self.assertEqual(self.client._translator.translate.call_count, 2)
        self.assertEqual(reply.variants[0].translation, "My name is Arthur.")

    def test_translation_failure_preserves_original_answer(self):
        self.client._translator = MagicMock()
        self.client._translator.translate.side_effect = RuntimeError("Пакет en→ru не установлен")
        self.respond('{"short_answer":"My name is Arthur.","long_answer":"My name is Arthur. It is an old name.","answer_language":"en"}')
        with self.assertLogs("app.assistant", level="WARNING"):
            reply = self.client.reply("Q", [], "")
        self.assertEqual(reply.variants[0].text, "My name is Arthur.")
        self.assertEqual(len(reply.variants), 2)
        self.assertEqual(reply.translation, "")
        self.assertIn("Пакет", reply.translation_error)

    def test_invalid_reply_language_is_not_sent_to_translator(self):
        self.client._translator = MagicMock()
        for language in (None, "de"):
            self.respond(json.dumps({"short_answer": "Answer", "long_answer": "A longer answer with details.", "answer_language": language}))
            with self.assertRaises(RuntimeError):
                self.client.reply("Q", [], "")
        self.client._translator.translate.assert_not_called()

    def test_two_variants_and_a2_contract_apply_even_to_existing_custom_prompts(self):
        self.client._translator = MagicMock()
        self.client._translator.translate.side_effect = ["Короткий перевод", "Длинный перевод"]
        self.respond(json.dumps({
            "short_answer": "My name is Arthur.",
            "long_answer": "My name is Arthur. People know this name from old stories about King Arthur.",
            "answer_language": "en",
        }))
        reply = self.client.reply("артур имя", [{"role": "heard", "text": "Tell me about the origin of your name."}],
                                  "Дай два варианта, по длиннее и по короче", manual=True)
        body = json.loads(self.client._opener.open.call_args.args[0].data)
        self.assertEqual(set(body["format"]["required"]), {"short_answer", "long_answer", "answer_language"})
        self.assertFalse(body["format"]["additionalProperties"])
        self.assertIn("MAXIMUM English level is CEFR A2", body["messages"][0]["content"])
        self.assertIn("short sentences", body["messages"][0]["content"])
        self.assertIn("concrete relevant information", body["messages"][0]["content"])
        self.assertIn("uncertainty in SIMPLE words", body["messages"][0]["content"])
        task = json.loads(body["messages"][-1]["content"])
        self.assertEqual(task["task"], "complete_previous_task")
        self.assertEqual(task["learner_name"], "артур")
        self.assertEqual([v.translation for v in reply.variants], ["Короткий перевод", "Длинный перевод"])
        self.assertGreater(len(reply.variants[1].text.split()), len(reply.variants[0].text.split()))
        self.client._opener.open.assert_called_once()  # два варианта за один запрос модели

    def test_single_duplicate_or_non_longer_variant_is_rejected(self):
        self.client._translator = MagicMock()
        bad = [
            {"answer": "Only one answer", "answer_language": "en"},
            {"short_answer": "Short answer", "long_answer": "", "answer_language": "en"},
            {"short_answer": "Same answer", "long_answer": "Same answer", "answer_language": "en"},
            {"short_answer": "This is the short answer", "long_answer": "Shorter", "answer_language": "en"},
        ]
        for response in bad:
            with self.subTest(response=response):
                self.respond(json.dumps(response))
                with self.assertRaisesRegex(RuntimeError, "короткий и развёрнутый"):
                    self.client.reply("Q", [], "")
        self.client._translator.translate.assert_not_called()

    def test_one_translation_error_does_not_hide_the_other_variant(self):
        self.client._translator = MagicMock()
        self.client._translator.translate.side_effect = [RuntimeError("missing model"), "Длинный перевод"]
        self.respond('{"short_answer":"Short answer.","long_answer":"A longer answer with useful details.","answer_language":"en"}')
        with self.assertLogs("app.assistant", level="WARNING"):
            reply = self.client.reply("Q", [], "")
        self.assertIn("missing model", reply.variants[0].translation_error)
        self.assertEqual(reply.variants[1].translation, "Длинный перевод")
        self.assertEqual(len(reply.variants), 2)

    def test_name_shorthand_and_questions_are_not_confused(self):
        for text in ("артур имя", "Артур моё имя", "имя Артур", "имя: Артур"):
            with self.subTest(text=text):
                self.assertEqual(learner_name_fact(text).casefold(), "артур")
        for text in ("какое имя", "что значит имя", "твое имя?", "что имя значит"):
            with self.subTest(text=text):
                self.assertEqual(learner_name_fact(text), "")

    def test_variant_json_is_strict_but_old_records_remain_readable(self):
        self.assertEqual(parse_variants([]), [])
        for invalid in (None, {}, [{"kind": "short", "text": "Only one"}],
                        [{"kind": "long", "text": "L"}, {"kind": "short", "text": "S"}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_variants(invalid)

    def test_legacy_question_mode_also_requires_two_variants(self):
        self.respond('{"is_question":true,"suggestions":[{"en":"Only one","ru":"Только один"}]}')
        with self.assertRaises(RuntimeError):
            self.client.suggest("What do you like?", "Что тебе нравится?", "en", [])

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
