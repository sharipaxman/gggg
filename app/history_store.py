"""
Локальное хранилище истории на SQLite.

Храним каждую фразу: оригинал, перевод, языки и подсказки ИИ, время, путь к привязанному
скриншоту (если есть). Храним только последние HISTORY_LIMIT записей —
старые автоматически удаляются, чтобы история не росла бесконечно.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.config import DB_PATH, HISTORY_LIMIT
from app.assistant import Assistance


@dataclass
class HistoryEntry:
    id: int
    created_at: str
    source_text: str
    translated_text: str
    screenshot_path: Optional[str]
    source_language: str = "en"
    target_language: str = "ru"
    assistance: Assistance = field(default_factory=Assistance)


class HistoryStore:
    def __init__(self, db_path: Path = DB_PATH, limit: int = HISTORY_LIMIT):
        self.db_path = db_path
        self.limit = limit
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS phrases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    screenshot_path TEXT
                )
                """
            )
            # Миграция существующей истории без удаления старых фраз.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(phrases)")}
            for name, declaration in (
                ("source_language", "TEXT NOT NULL DEFAULT 'en'"),
                ("target_language", "TEXT NOT NULL DEFAULT 'ru'"),
                ("assistance_json", "TEXT NOT NULL DEFAULT '{\"is_question\":false,\"suggestions\":[]}'"),
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE phrases ADD COLUMN {name} {declaration}")
            conn.commit()

    def add_entry(self, source_text: str, translated_text: str, screenshot_path: Optional[str] = None,
                  source_language: str = "en", target_language: str = "ru") -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO phrases (created_at, source_text, translated_text, screenshot_path, "
                "source_language, target_language) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), source_text, translated_text, screenshot_path,
                 source_language, target_language),
            )
            new_id = cur.lastrowid
            self._trim_to_limit(conn)
            conn.commit()
            return new_id

    def update_assistance(self, entry_id: int, assistance: Assistance):
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE phrases SET assistance_json = ? WHERE id = ?",
                         (assistance.to_json(), entry_id))

    def _trim_to_limit(self, conn: sqlite3.Connection):
        conn.execute(
            """
            DELETE FROM phrases WHERE id NOT IN (
                SELECT id FROM phrases ORDER BY id DESC LIMIT ?
            )
            """,
            (self.limit,),
        )

    def attach_screenshot_to_latest(self, screenshot_path: str) -> Optional[int]:
        """Прикрепляет снимок экрана к последней сохранённой фразе (хоткей)."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT id FROM phrases ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                return None
            conn.execute("UPDATE phrases SET screenshot_path = ? WHERE id = ?", (screenshot_path, row["id"]))
            conn.commit()
            return row["id"]

    def get_recent(self, limit: Optional[int] = None) -> List[HistoryEntry]:
        limit = limit or self.limit
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM phrases ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            HistoryEntry(
                id=r["id"],
                created_at=r["created_at"],
                source_text=r["source_text"],
                translated_text=r["translated_text"],
                screenshot_path=r["screenshot_path"],
                source_language=r["source_language"],
                target_language=r["target_language"],
                assistance=Assistance.from_json(r["assistance_json"]),
            )
            for r in rows
        ]

    def get_context_text(self, max_entries: int = 5) -> str:
        """Формирует текстовый контекст урока из последних N фраз — для показа рядом с новой фразой."""
        entries = self.get_recent(limit=max_entries)
        entries.reverse()
        lines = []
        for e in entries:
            lines.append(f"[{e.created_at}] {e.source_language.upper()}: {e.source_text}\n     {e.target_language.upper()}: {e.translated_text}")
        return "\n".join(lines)
