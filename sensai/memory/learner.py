"""Structured learner memory (M3): vocabulary with spaced repetition,
recurring mistakes, and an audit log of every operation.

The model decides *what* to remember (see extractor.py) as a list of
operations; this module validates and applies them. Nothing the model
outputs touches the database without going through `apply()`.

Spaced repetition uses a Leitner system: each word sits in a box 0..5.
A correct answer moves it up one box, a wrong answer sends it back to box 0,
and the box decides when the word is due for review again.
"""

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field

from .tokens import CJK, estimate_message

SCHEMA = """
CREATE TABLE IF NOT EXISTS vocabulary (
    id           INTEGER PRIMARY KEY,
    user         TEXT NOT NULL,
    language     TEXT NOT NULL,
    word         TEXT NOT NULL,
    translation  TEXT NOT NULL,
    box          INTEGER NOT NULL DEFAULT 0,
    correct      INTEGER NOT NULL DEFAULT 0,
    wrong        INTEGER NOT NULL DEFAULT 0,
    next_review  REAL NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    UNIQUE (user, language, word)
);
CREATE INDEX IF NOT EXISTS idx_vocab_due ON vocabulary(user, language, next_review);
CREATE TABLE IF NOT EXISTS mistakes (
    id           INTEGER PRIMARY KEY,
    user         TEXT NOT NULL,
    language     TEXT NOT NULL,
    pattern      TEXT NOT NULL,
    correction   TEXT NOT NULL,
    example      TEXT,
    count        INTEGER NOT NULL DEFAULT 1,
    last_seen    REAL NOT NULL,
    UNIQUE (user, language, pattern)
);
CREATE TABLE IF NOT EXISTS memory_ops (
    id           INTEGER PRIMARY KEY,
    user         TEXT NOT NULL,
    session_id   INTEGER,
    op           TEXT NOT NULL,
    payload      TEXT NOT NULL,
    applied      INTEGER NOT NULL,
    error        TEXT,
    created_at   REAL NOT NULL
);
"""

DAY = 86400
LEITNER_DAYS = [0, 1, 3, 7, 14, 30]  # review interval per box
MASTERED_BOX = 4
MAX_FIELD = 200
MAX_OPS_PER_TURN = 12

OPS = ["add_word", "review_word", "delete_word", "add_mistake", "resolve_mistake", "set_profile"]
REQUIRED = {
    "add_word": ["word", "translation"],
    "review_word": ["word", "correct"],
    "delete_word": ["word"],
    "add_mistake": ["pattern", "correction"],
    "resolve_mistake": ["pattern"],
    "set_profile": ["key", "value"],
}


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def _mentions(text: str, word: str) -> bool:
    """Whole-word match, so « inu » is not found in « inutile ».
    Scripts written without spaces (Japanese, Chinese) fall back to substring."""
    if CJK.search(word):
        return word in text
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


@dataclass
class Word:
    word: str
    translation: str
    box: int
    correct: int
    wrong: int
    next_review: float

    def line(self) -> str:
        return f"{self.word} = {self.translation} ({self.correct} ✔ / {self.wrong} ✘)"


@dataclass
class Mistake:
    pattern: str
    correction: str
    example: str | None
    count: int

    def line(self) -> str:
        ex = f" — ex. « {self.example} »" if self.example else ""
        return f"{self.pattern} → {self.correction} (vue {self.count} fois){ex}"


@dataclass
class ApplyReport:
    applied: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def short(self) -> str:
        counts: dict[str, int] = {}
        for op in self.applied:
            counts[op] = counts.get(op, 0) + 1
        parts = [f"{n}× {op}" for op, n in counts.items()]
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejetée(s)")
        return ", ".join(parts)


class LearnerMemory:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.executescript(SCHEMA)

    # --- vocabulary -----------------------------------------------------

    def upsert_word(self, user: str, language: str, word: str, translation: str) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                """INSERT INTO vocabulary (user, language, word, translation, next_review, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (user, language, word)
                   DO UPDATE SET translation = excluded.translation, updated_at = excluded.updated_at""",
                (user, language, norm(word), translation.strip(), now, now, now),
            )

    def review_word(self, user: str, language: str, word: str, correct: bool, translation: str = "") -> None:
        """Leitner step; unknown words are created first if a translation is given."""
        key = norm(word)
        row = self.conn.execute(
            "SELECT box FROM vocabulary WHERE user = ? AND language = ? AND word = ?", (user, language, key)
        ).fetchone()
        if row is None:
            if not translation:
                raise ValueError(f"mot inconnu « {word} » et pas de traduction pour le créer")
            self.upsert_word(user, language, key, translation)
            box = 0
        else:
            box = row["box"]
        box = min(box + 1, len(LEITNER_DAYS) - 1) if correct else 0
        now = time.time()
        with self.conn:
            self.conn.execute(
                f"""UPDATE vocabulary SET box = ?, next_review = ?, updated_at = ?,
                    {'correct = correct + 1' if correct else 'wrong = wrong + 1'}
                    WHERE user = ? AND language = ? AND word = ?""",
                (box, now + LEITNER_DAYS[box] * DAY, now, user, language, key),
            )

    def delete_word(self, user: str, language: str, word: str) -> bool:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM vocabulary WHERE user = ? AND language = ? AND word = ?", (user, language, norm(word))
            )
        return cur.rowcount > 0

    def words(self, user: str, language: str, limit: int = 500) -> list[Word]:
        rows = self.conn.execute(
            """SELECT word, translation, box, correct, wrong, next_review FROM vocabulary
               WHERE user = ? AND language = ? ORDER BY box, next_review LIMIT ?""",
            (user, language, limit),
        ).fetchall()
        return [Word(*r) for r in rows]

    def due_words(self, user: str, language: str, limit: int = 10) -> list[Word]:
        rows = self.conn.execute(
            """SELECT word, translation, box, correct, wrong, next_review FROM vocabulary
               WHERE user = ? AND language = ? AND next_review <= ? ORDER BY box, next_review LIMIT ?""",
            (user, language, time.time(), limit),
        ).fetchall()
        return [Word(*r) for r in rows]

    def mentioned_words(self, user: str, language: str, text: str, limit: int = 10) -> list[Word]:
        low = norm(text)
        return [w for w in self.words(user, language) if _mentions(low, w.word)][:limit]

    def vocab_stats(self, user: str, language: str) -> tuple[int, int]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(box >= ?), 0) AS m FROM vocabulary WHERE user = ? AND language = ?",
            (MASTERED_BOX, user, language),
        ).fetchone()
        return row["n"], row["m"]

    # --- mistakes -------------------------------------------------------

    def add_mistake(self, user: str, language: str, pattern: str, correction: str, example: str = "") -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO mistakes (user, language, pattern, correction, example, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (user, language, pattern) DO UPDATE SET count = count + 1,
                   correction = excluded.correction, example = COALESCE(excluded.example, example),
                   last_seen = excluded.last_seen""",
                (user, language, norm(pattern), correction.strip(), example.strip() or None, time.time()),
            )

    def resolve_mistake(self, user: str, language: str, pattern: str) -> bool:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM mistakes WHERE user = ? AND language = ? AND pattern = ?", (user, language, norm(pattern))
            )
        return cur.rowcount > 0

    def mistakes(self, user: str, language: str, limit: int = 10) -> list[Mistake]:
        rows = self.conn.execute(
            """SELECT pattern, correction, example, count FROM mistakes WHERE user = ? AND language = ?
               ORDER BY count DESC, last_seen DESC LIMIT ?""",
            (user, language, limit),
        ).fetchall()
        return [Mistake(*r) for r in rows]

    # --- operations from the model --------------------------------------

    def apply(
        self, user: str, language: str, ops: list[dict], set_profile, session_id: int | None = None
    ) -> ApplyReport:
        """Validate and apply model operations; every op is logged in memory_ops."""
        report = ApplyReport()
        for op in ops[:MAX_OPS_PER_TURN]:
            name = op.get("op", "?")
            try:
                self._apply_one(user, language, op, set_profile)
            except ValueError as exc:
                report.rejected.append(f"{name}: {exc}")
                self._log(user, session_id, name, op, error=str(exc))
            else:
                report.applied.append(name)
                self._log(user, session_id, name, op)
        for op in ops[MAX_OPS_PER_TURN:]:
            report.rejected.append(f"{op.get('op', '?')}: trop d'opérations dans un seul tour")
            self._log(user, session_id, op.get("op", "?"), op, error="too many operations")
        return report

    def _apply_one(self, user: str, language: str, op: dict, set_profile) -> None:
        name = op.get("op")
        if name not in OPS:
            raise ValueError("opération inconnue")
        for key in REQUIRED[name]:
            value = op.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"champ « {key} » manquant")
            if isinstance(value, str) and len(value) > MAX_FIELD:
                raise ValueError(f"champ « {key} » trop long")
        if name == "add_word":
            self.upsert_word(user, language, op["word"], op["translation"])
        elif name == "review_word":
            if not isinstance(op["correct"], bool):
                raise ValueError("« correct » doit être un booléen")
            self.review_word(user, language, op["word"], op["correct"], op.get("translation") or "")
        elif name == "delete_word":
            if not self.delete_word(user, language, op["word"]):
                raise ValueError(f"mot « {op['word']} » absent")
        elif name == "add_mistake":
            self.add_mistake(user, language, op["pattern"], op["correction"], op.get("example") or "")
        elif name == "resolve_mistake":
            if not self.resolve_mistake(user, language, op["pattern"]):
                raise ValueError(f"erreur « {op['pattern']} » absente")
        elif name == "set_profile":
            set_profile(user, norm(op["key"]).replace(" ", "_"), op["value"].strip())

    def _log(self, user: str, session_id: int | None, op: str, payload: dict, error: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO memory_ops (user, session_id, op, payload, applied, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user, session_id, op, json.dumps(payload, ensure_ascii=False), error is None, error, time.time()),
            )

    def log_failure(self, user: str, session_id: int | None, error: str) -> None:
        self._log(user, session_id, "extract", {}, error=error)

    def op_stats(self, user: str) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS total, COALESCE(SUM(applied), 0) AS applied,
                      COALESCE(SUM(op = 'extract'), 0) AS failed_extractions
               FROM memory_ops WHERE user = ?""",
            (user,),
        ).fetchone()
        return dict(row)

    # --- context injection ----------------------------------------------

    def render(self, user: str, language: str, current_message: str, max_tokens: int) -> str | None:
        """Markdown block for the system prompt, filled by priority until max_tokens."""
        total, mastered = self.vocab_stats(user, language)
        mistakes = self.mistakes(user, language, limit=5)
        if not total and not mistakes:
            return None

        header = f"## Mémoire de l'apprenant ({language or 'langue non précisée'})\n{total} mots vus, {mastered} maîtrisés."
        footer = "Réutilise les mots à réviser dans tes exercices et surveille les erreurs fréquentes."
        sections: list[tuple[str, list[str]]] = [
            ("Mots présents dans le message", [w.line() for w in self.mentioned_words(user, language, current_message)]),
            ("Mots à réviser", [w.line() for w in self.due_words(user, language)]),
            ("Erreurs fréquentes", [m.line() for m in mistakes]),
        ]

        text = f"{header}\n\n{footer}"
        if estimate_message(text) > max_tokens:
            return None
        blocks: list[str] = []
        for title, lines in sections:
            seen = [line for line in lines if not any(line in b for b in blocks)]
            kept: list[str] = []
            for line in seen:
                section = f"### {title}\n" + "\n".join(f"- {x}" for x in kept + [line])
                candidate = "\n\n".join([header, *blocks, section, footer])
                if estimate_message(candidate) > max_tokens:
                    break
                kept.append(line)
            if kept:
                blocks.append(f"### {title}\n" + "\n".join(f"- {line}" for line in kept))
        return "\n\n".join([header, *blocks, footer])
