"""Сохранённые инструкции, память урока и ручной чат (независимо от прослушивания)."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.config import DB_PATH
from app.assistant import AssistantReply, active_heard_task, is_learner_fact

DEFAULT_PROMPT = (
    "Ты мой помощник на уроке английского. Учитывай контекст разговора. "
    "На каждую новую реплику дай короткий полезный ответ: предложи, что можно сказать "
    "на английском, и рядом русский перевод. Если я задаю вопрос вручную, ответь на него "
    "по контексту. Не выдумывай мои личные данные; если сведений нет, прямо скажи об этом."
)
PROMPT_MAX_CHARS = 1500
QUESTION_MAX_CHARS = 1000
MEMORY_LIMIT = 200
CONTEXT_CHAR_BUDGET = 3000


@dataclass(frozen=True)
class AssistantPreferences:
    prompt: str = DEFAULT_PROMPT
    always_reply: bool = True


@dataclass(frozen=True)
class ChatResult:
    question: str
    answer: str = ""
    error: str = ""
    translation: str = ""
    answer_language: str = ""
    translation_language: str = ""
    translation_error: str = ""


class AssistantWorkspace:
    """Память отделена от ленты 50 переводов; хранится максимум 200 событий.

    В запрос входит ограниченный фрагмент последних событий, не вся БД.
    Роли различают услышанную речь, реальные вопросы пользователя и примеры ИИ.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._generation = 0
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='assistant_memory'").fetchone()
            conn.execute("CREATE TABLE IF NOT EXISTS assistant_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("""CREATE TABLE IF NOT EXISTS assistant_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                role TEXT NOT NULL, text TEXT NOT NULL)""")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(assistant_memory)")}
            for name in ("translation", "answer_language", "translation_language", "translation_error"):
                if name not in columns:
                    conn.execute(f"ALTER TABLE assistant_memory ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
            # Подхватываем старую историю один раз, но не после очистки памяти.
            if not exists and conn.execute("SELECT 1 FROM sqlite_master WHERE name='phrases'").fetchone():
                conn.execute("""INSERT INTO assistant_memory (created_at, role, text)
                    SELECT created_at, 'heard', source_text FROM
                    (SELECT * FROM phrases ORDER BY id DESC LIMIT 50) ORDER BY id""")

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def preferences(self) -> AssistantPreferences:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT value FROM assistant_settings WHERE key='preferences'").fetchone()
        if not row:
            return AssistantPreferences()
        data = json.loads(row["value"])
        return AssistantPreferences(prompt=data["prompt"], always_reply=data["always_reply"])

    def save_preferences(self, prompt: str, always_reply: bool):
        if len(prompt) > PROMPT_MAX_CHARS:
            raise ValueError(f"Промпт не должен превышать {PROMPT_MAX_CHARS} символов")
        value = json.dumps(asdict(AssistantPreferences(prompt.strip(), bool(always_reply))), ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO assistant_settings VALUES ('preferences', ?)", (value,))

    def remember(self, role: str, text: str, generation: int | None = None, reply: AssistantReply | None = None):
        if role not in ("heard", "user", "assistant", "suggestion") or not text.strip():
            raise ValueError("Некорректное событие памяти")
        with self._lock, self._connect() as conn:
            if generation is not None and generation != self._generation:
                return
            conn.execute("INSERT INTO assistant_memory "
                         "(created_at, role, text, translation, answer_language, translation_language, translation_error) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (datetime.now().isoformat(timespec="seconds"), role, text,
                          reply.translation if reply else "", reply.language if reply else "",
                          reply.target_language if reply else "", reply.translation_error if reply else ""))
            conn.execute("DELETE FROM assistant_memory WHERE id NOT IN "
                         "(SELECT id FROM assistant_memory ORDER BY id DESC LIMIT ?)", (MEMORY_LIMIT,))

    def recent(self, chat_only: bool = False) -> list[dict]:
        where = "WHERE role IN ('user', 'assistant')" if chat_only else ""
        with self._lock, self._connect() as conn:
            rows = conn.execute(f"SELECT role, text, created_at, translation, answer_language, translation_language, translation_error FROM assistant_memory {where} ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def snapshot(self) -> tuple[int, list[dict]]:
        """Свежие события в хронологическом порядке, с ограниченным размером."""
        with self._lock:
            events = self.recent()
            active_task = active_heard_task(events)
            task_index = next((i for i in range(len(events) - 1, -1, -1)
                               if events[i]["role"] == "heard" and events[i]["text"] == active_task), None)
            fact_index = next((i for i in range(len(events) - 1, -1, -1)
                               if events[i]["role"] == "user" and is_learner_fact(events[i]["text"])), None)
            # Речь и уточнения не должны исчезать под повторяющимися ответами ИИ.
            priority = [(i, 800) for i in (task_index, fact_index) if i is not None]
            drafts = [i for i in range(len(events) - 1, -1, -1)
                      if events[i]["role"] in ("assistant", "suggestion")][:2]
            priority += [(i, 240) for i in drafts]
            priority += [(i, 800) for i in range(len(events) - 1, -1, -1)
                         if events[i]["role"] in ("heard", "user")]
            selected, remaining = {}, CONTEXT_CHAR_BUDGET
            for index, limit in priority:
                if index in selected or remaining <= 0:
                    continue
                event = events[index]
                text = event["text"][:min(limit, remaining)]
                selected[index] = {"role": event["role"], "text": text}
                remaining -= len(text)
            return self._generation, [selected[index] for index in sorted(selected)]

    def clear_memory(self):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM assistant_memory")
            self._generation += 1


class ContextChat:
    """Один ручной запрос за раз. Не запускает захват звука, не трогает Qt из потока."""

    def __init__(self, assistant, workspace: AssistantWorkspace):
        self.assistant = assistant
        self.workspace = workspace
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._thread = None

    def ask(self, question: str, on_result: Callable[[ChatResult], None]):
        question = question.strip()
        if not question or len(question) > QUESTION_MAX_CHARS:
            raise ValueError(f"Введите вопрос длиной от 1 до {QUESTION_MAX_CHARS} символов")
        if self._closed.is_set() or not self._lock.acquire(blocking=False):
            raise RuntimeError("Предыдущий запрос ещё выполняется")
        try:
            preferences = self.workspace.preferences()
            generation, context = self.workspace.snapshot()
            self.workspace.remember("user", question)
            self._thread = threading.Thread(
                target=self._answer, args=(question, preferences.prompt, generation, context, on_result),
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self._lock.release()
            raise

    def _answer(self, question, prompt, generation, context, on_result):
        try:
            reply = self.assistant.reply(question, context, prompt, manual=True)
            if generation != self.workspace.snapshot()[0]:
                result = ChatResult(question, error="Память была очищена во время запроса. Задайте вопрос заново.")
            else:
                self.workspace.remember("assistant", reply.text, generation, reply=reply)
                result = ChatResult(question, answer=reply.text, translation=reply.translation,
                                    answer_language=reply.language, translation_language=reply.target_language,
                                    translation_error=reply.translation_error)
        except Exception as exc:
            result = ChatResult(question, error=str(exc))
        finally:
            self._lock.release()
        if not self._closed.is_set():
            on_result(result)

    def close(self):
        self._closed.set()
