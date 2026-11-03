"""Промпт, ограниченная память и ручной контекстный чат без сети, ML и Qt."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.assistant_workspace import (
    AssistantPreferences, AssistantWorkspace, ContextChat, CONTEXT_CHAR_BUDGET,
    PROMPT_MAX_CHARS, QUESTION_MAX_CHARS,
)
from app.history_store import HistoryStore


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.db"
        self.workspace = AssistantWorkspace(self.path)

    def test_preferences_persist_across_restarts(self):
        self.assertEqual(self.workspace.preferences(), AssistantPreferences())
        self.workspace.save_preferences("  Respond in English only  ", False)
        reopened = AssistantWorkspace(self.path)
        self.assertEqual(reopened.preferences(), AssistantPreferences("Respond in English only", False))

    def test_overlong_prompt_does_not_replace_settings(self):
        with self.assertRaises(ValueError):
            self.workspace.save_preferences("x" * (PROMPT_MAX_CHARS + 1), True)
        self.assertEqual(self.workspace.preferences(), AssistantPreferences())

    def test_bootstrap_old_transcripts_only_once_and_clear_preserves_history(self):
        other = Path(self.temp.name) / "old.db"
        history = HistoryStore(other)
        history.add_entry("First", "Первое")
        history.add_entry("Second", "Второе")
        workspace = AssistantWorkspace(other)
        self.assertEqual([event["text"] for event in workspace.recent()], ["First", "Second"])
        workspace.save_preferences("Keep this prompt", True)
        workspace.clear_memory()
        reopened = AssistantWorkspace(other)
        self.assertEqual(reopened.recent(), [])
        self.assertEqual(len(history.get_recent()), 2)
        self.assertEqual(reopened.preferences().prompt, "Keep this prompt")

    def test_memory_is_bounded_and_roles_are_preserved(self):
        with patch("app.assistant_workspace.MEMORY_LIMIT", 3):
            for i in range(7):
                self.workspace.remember("heard", str(i))
        self.assertEqual([event["text"] for event in self.workspace.recent()], ["4", "5", "6"])
        reopened = AssistantWorkspace(self.path)
        self.assertEqual(len(reopened.recent()), 3)
        self.assertEqual(reopened.recent()[0]["role"], "heard")

    def test_snapshot_is_chronological_and_bounded(self):
        for i in range(20):
            self.workspace.remember("heard", f"{i}:" + "x" * 1000)
        _, context = self.workspace.snapshot()
        self.assertLessEqual(sum(len(event["text"]) for event in context), CONTEXT_CHAR_BUDGET)
        self.assertEqual(context[-1]["text"][:3], "19:")
        self.assertEqual(sorted(int(event["text"].split(":")[0]) for event in context),
                         [int(event["text"].split(":")[0]) for event in context])

    def test_clear_invalidates_inflight_write_without_erasing_prompt(self):
        generation, _ = self.workspace.snapshot()
        self.workspace.clear_memory()
        self.workspace.remember("suggestion", "stale answer", generation)
        self.assertEqual(self.workspace.recent(), [])
        self.workspace.remember("heard", "new topic")
        self.assertEqual(self.workspace.recent()[0]["text"], "new topic")

    def test_chat_only_excludes_heard_and_generated_examples(self):
        for role in ("heard", "user", "assistant", "suggestion"):
            self.workspace.remember(role, role)
        self.assertEqual([e["role"] for e in self.workspace.recent(chat_only=True)], ["user", "assistant"])


class ContextChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "chat.db"
        self.workspace = AssistantWorkspace(self.path)
        self.assistant = MagicMock()
        self.assistant.answer.return_value = "You discussed Paris."
        self.chat = ContextChat(self.assistant, self.workspace)
        self.addCleanup(self.chat.close)

    def run_question(self, question):
        done, results = threading.Event(), []
        self.chat.ask(question, lambda result: (results.append(result), done.set()))
        self.assertTrue(done.wait(timeout=2))
        self.chat._thread.join(timeout=2)
        return results[0]

    def test_manual_question_uses_persisted_prompt_and_heard_context(self):
        self.workspace.save_preferences("Answer concisely", True)
        self.workspace.remember("heard", "I visited Paris.")
        result = self.run_question("Which city did I visit?")
        self.assertEqual(result.answer, "You discussed Paris.")
        self.assistant.answer.assert_called_once_with(
            "Which city did I visit?", [{"role": "heard", "text": "I visited Paris."}],
            "Answer concisely", manual=True,
        )
        reopened = AssistantWorkspace(self.path)
        self.assertEqual([e["role"] for e in reopened.recent()], ["heard", "user", "assistant"])

    def test_followup_contains_previous_user_question_and_answer(self):
        self.run_question("Which city?")
        self.run_question("Can you repeat that?")
        context = self.assistant.answer.call_args.args[1]
        self.assertEqual([e["role"] for e in context], ["user", "assistant"])
        self.assertEqual(context[1]["text"], "You discussed Paris.")

    def test_busy_request_is_rejected_without_duplicate_question(self):
        started, release = threading.Event(), threading.Event()

        def slow_answer(*args, **kwargs):
            started.set()
            release.wait(timeout=3)
            return "Answer"

        self.assistant.answer.side_effect = slow_answer
        self.chat.ask("First?", lambda result: None)
        try:
            self.assertTrue(started.wait(timeout=2))
            with self.assertRaisesRegex(RuntimeError, "выполняется"):
                self.chat.ask("Second?", lambda result: None)
            self.assertEqual([e["text"] for e in self.workspace.recent()], ["First?"])
        finally:
            release.set()
            self.chat._thread.join(timeout=2)

    def test_failure_is_visible_and_next_question_can_be_asked(self):
        self.assistant.answer.side_effect = RuntimeError("Ollama offline")
        first = self.run_question("Question?")
        self.assertEqual(first.error, "Ollama offline")
        self.assertEqual(first.answer, "")
        self.assistant.answer.side_effect = None
        second = self.run_question("Retry?")
        self.assertEqual(second.answer, "You discussed Paris.")
        self.assertEqual([e["role"] for e in self.workspace.recent()], ["user", "user", "assistant"])

    def test_invalid_question_is_not_stored_or_sent(self):
        for question in (" ", "x" * (QUESTION_MAX_CHARS + 1)):
            with self.assertRaises(ValueError):
                self.chat.ask(question, lambda result: None)
        self.assistant.answer.assert_not_called()
        self.assertEqual(self.workspace.recent(), [])

    def test_clear_during_generation_does_not_restore_old_answer(self):
        def answer_and_clear(*args, **kwargs):
            self.workspace.clear_memory()
            return "Old answer"

        self.assistant.answer.side_effect = answer_and_clear
        result = self.run_question("Question?")
        self.assertEqual(result.answer, "")
        self.assertIn("очищена", result.error)
        self.assertEqual(self.workspace.recent(), [])

    def test_close_suppresses_callback_to_destroyed_ui(self):
        started, release = threading.Event(), threading.Event()
        callback = MagicMock()

        def slow_answer(*args, **kwargs):
            started.set()
            release.wait(timeout=3)
            return "Answer"

        self.assistant.answer.side_effect = slow_answer
        self.chat.ask("Question?", callback)
        self.assertTrue(started.wait(timeout=2))
        self.chat.close()
        release.set()
        self.chat._thread.join(timeout=2)
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
