"""SQLite persistence: sessions, messages and learner profiles (M1).

The same database will host the structured learner memory (M3: vocabulary,
recurring errors...) so everything about a learner lives in one place.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    user        TEXT NOT NULL,
    title       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content     TEXT NOT NULL,
    tokens      INTEGER,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS profile (
    user        TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (user, key)
);
-- M2: one rolling summary per session, covering every message up to covered_until.
CREATE TABLE IF NOT EXISTS summaries (
    session_id     INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    content        TEXT NOT NULL,
    covered_until  INTEGER NOT NULL,
    tokens         INTEGER NOT NULL,
    updated_at     REAL NOT NULL
);
-- M2 metrics: one row per compression pass.
CREATE TABLE IF NOT EXISTS compressions (
    id                 INTEGER PRIMARY KEY,
    session_id         INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    messages_in        INTEGER NOT NULL,
    tokens_before      INTEGER NOT NULL,
    tokens_after       INTEGER NOT NULL,
    duration_ms        REAL NOT NULL,
    ok                 INTEGER NOT NULL,
    error              TEXT,
    created_at         REAL NOT NULL
);
"""


@dataclass
class Message:
    role: str
    content: str
    tokens: int | None = None
    id: int | None = None

    def to_ollama(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class SessionInfo:
    id: int
    title: str | None
    updated_at: float
    message_count: int


class MemoryStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- sessions -------------------------------------------------------

    def create_session(self, user: str) -> int:
        now = time.time()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO sessions (user, created_at, updated_at) VALUES (?, ?, ?)", (user, now, now)
            )
        return cur.lastrowid

    def list_sessions(self, user: str, limit: int = 20) -> list[SessionInfo]:
        rows = self.conn.execute(
            """SELECT s.id, s.title, s.updated_at, COUNT(m.id) AS n
               FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
               WHERE s.user = ? GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?""",
            (user, limit),
        ).fetchall()
        return [SessionInfo(r["id"], r["title"], r["updated_at"], r["n"]) for r in rows]

    def session_exists(self, user: str, session_id: int) -> bool:
        row = self.conn.execute("SELECT 1 FROM sessions WHERE id = ? AND user = ?", (session_id, user)).fetchone()
        return row is not None

    def latest_session(self, user: str) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM sessions WHERE user = ? ORDER BY updated_at DESC LIMIT 1", (user,)
        ).fetchone()
        return row["id"] if row else None

    # --- messages -------------------------------------------------------

    def add_message(self, session_id: int, msg: Message) -> None:
        """Insert `msg` and set its id."""
        now = time.time()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO messages (session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, msg.role, msg.content, msg.tokens, now),
            )
            msg.id = cur.lastrowid
            self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            if msg.role == "user":
                # First user message becomes the session title.
                self.conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ? AND title IS NULL",
                    (msg.content[:60], session_id),
                )

    def load_messages(self, session_id: int, after_id: int = 0) -> list[Message]:
        rows = self.conn.execute(
            "SELECT id, role, content, tokens FROM messages WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, after_id),
        ).fetchall()
        return [Message(r["role"], r["content"], r["tokens"], r["id"]) for r in rows]

    def count_messages(self, session_id: int) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)).fetchone()
        return row["n"]

    # --- rolling summary (M2) -------------------------------------------

    def get_summary(self, session_id: int) -> tuple[str, int] | None:
        """Return (content, covered_until) or None."""
        row = self.conn.execute(
            "SELECT content, covered_until FROM summaries WHERE session_id = ?", (session_id,)
        ).fetchone()
        return (row["content"], row["covered_until"]) if row else None

    def set_summary(self, session_id: int, content: str, covered_until: int, tokens: int) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO summaries (session_id, content, covered_until, tokens, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (session_id) DO UPDATE SET content = excluded.content,
                   covered_until = excluded.covered_until, tokens = excluded.tokens,
                   updated_at = excluded.updated_at""",
                (session_id, content, covered_until, tokens, time.time()),
            )

    def log_compression(
        self,
        session_id: int,
        messages_in: int,
        tokens_before: int,
        tokens_after: int,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO compressions (session_id, messages_in, tokens_before, tokens_after,
                   duration_ms, ok, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, messages_in, tokens_before, tokens_after, duration_ms, error is None, error, time.time()),
            )

    def compression_stats(self, session_id: int) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS passes, COALESCE(SUM(ok), 0) AS ok,
                      COALESCE(SUM(CASE WHEN ok THEN messages_in END), 0) AS messages,
                      COALESCE(SUM(CASE WHEN ok THEN tokens_before - tokens_after END), 0) AS saved,
                      COALESCE(AVG(CASE WHEN ok THEN duration_ms END), 0) AS avg_ms
               FROM compressions WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        return dict(row)

    # --- learner profile ------------------------------------------------

    def get_profile(self, user: str) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM profile WHERE user = ? ORDER BY key", (user,)).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_profile(self, user: str, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO profile (user, key, value, updated_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT (user, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (user, key, value, time.time()),
            )

    def delete_profile_key(self, user: str, key: str) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM profile WHERE user = ? AND key = ?", (user, key))
        return cur.rowcount > 0
