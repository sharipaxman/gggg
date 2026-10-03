"""
Локальное хранилище истории на SQLite.

Храним каждую фразу: оригинал (EN), перевод (RU), время, путь к привязанному
скриншоту (если есть). Храним только последние HISTORY_LIMIT записей —
старые автоматически удаляются, чтобы история не росла бесконечно.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.config import DB_PATH, HISTORY_LIMIT


@dataclass
class HistoryEntry:
    id: int
    created_at: str
    source_text: str
    translated_text: str
    screenshot_path: Optional[str]


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
            conn.commit()

    def add_entry(self, source_text: str, translated_text: str, screenshot_path: Optional[str] = None) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO phrases (created_at, source_text, translated_text, screenshot_path) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), source_text, translated_text, screenshot_path),
            )
            new_id = cur.lastrowid
            self._trim_to_limit(conn)
            conn.commit()
            return new_id

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
            )
            for r in rows
        ]

    def get_context_text(self, max_entries: int = 5) -> str:
        """Формирует текстовый контекст урока из последних N фраз — для показа рядом с новой фразой."""
        entries = self.get_recent(limit=max_entries)
        entries.reverse()
        lines = []
        for e in entries:
            lines.append(f"[{e.created_at}] EN: {e.source_text}\n     RU: {e.translated_text}")
        return "\n".join(lines)
