"""Structured logging with secret masking (EV3).

Every component (chat turns, ReAct loop, tools, file guard, sandbox,
confirmations) reports events through one EventLogger. Events are JSON
objects, one per line, appended to a JSONL file: easy to `tail -f`, to grep
and to aggregate (see monitoring.py), with no extra dependency.

Privacy rules, applied to every event before it is written:
- values whose key looks sensitive (password, token, api_key...) are replaced;
- secrets recognized in free text (bearer tokens, API keys, private keys,
  credentials in URLs, emails, card numbers...) are masked;
- long strings are truncated to `content_chars`: we log what happened and how
  big it was, not the full prompt, answer or file content.
"""

import json
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MASK = "***"

SENSITIVE_KEYS = re.compile(
    r"(pass(word|wd)?|pwd|secret|token|api[_-]?key|authorization|auth|cookie|session[_-]?key|private[_-]?key|credential)",
    re.IGNORECASE,
)

# (pattern, replacement), applied in order.
SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)", re.DOTALL), "<private-key>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer " + MASK),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"), "<jwt>"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"), "<github-token>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "<api-key>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<aws-key>"),
    (re.compile(r"(?i)\b(xox[abposr]-[A-Za-z0-9-]{10,})\b"), "<slack-token>"),
    (re.compile(r"(?i)(://[^/\s:@]+:)[^/\s@]+@"), r"\1" + MASK + "@"),
    (
        re.compile(
            r"(?i)\b(pass(word|wd)?|pwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret|mot[_ -]de[_ -]passe)"
            r"(\s*[:=]\s*)(['\"]?)[^\s'\",;]+"
        ),
        r"\1\3\4" + MASK,
    ),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<email>"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<card-number>"),
    (re.compile(r"(?<!\d)(?:\+33\s?|0)[1-9](?:[ .-]?\d{2}){4}(?!\d)"), "<phone>"),
]


def redact(text: str) -> str:
    """Mask secrets and personal data found in free text."""
    for pattern, repl in SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return f"<{len(text)} chars>" if text else ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [+{len(text) - limit} chars]"


def sanitize(value, limit: int, key: str = ""):
    """Return a JSON-safe copy of `value` with secrets masked and strings truncated."""
    # Counters such as `prompt_tokens` are not secrets: only text values are masked by key.
    if key and isinstance(value, str) and value and SENSITIVE_KEYS.search(key):
        return MASK
    if isinstance(value, str):
        return truncate(redact(value), limit)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): sanitize(v, limit, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [sanitize(v, limit) for v in items[:20]]
        if len(items) > 20:
            out.append(f"<+{len(items) - 20} items>")
        return out
    return truncate(redact(str(value)), limit)


def new_interaction_id() -> str:
    return uuid.uuid4().hex[:12]


class EventLogger:
    """Append-only JSONL event log. Thread-safe; never raises on I/O errors."""

    def __init__(self, path: str | None, content_chars: int = 120, max_bytes: int = 5_000_000):
        self.path = Path(path) if path else None
        self.content_chars = content_chars
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.write_errors = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, interaction_id: str | None = None, **fields) -> dict:
        record = {"ts": round(time.time(), 3), "event": event}
        if interaction_id:
            record["interaction_id"] = interaction_id
        record.update(sanitize(fields, self.content_chars))
        if self.path:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self.lock:
                try:
                    self._rotate()
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except OSError:
                    # Logging must never break the chat.
                    self.write_errors += 1
        return record

    @contextmanager
    def timed(self, event: str, interaction_id: str | None = None, **fields) -> Iterator[dict]:
        """Log `event` with duration_ms and status; the caller may add fields to the yielded dict."""
        extra: dict = {}
        start = time.perf_counter()
        try:
            yield extra
        except BaseException as exc:
            extra.setdefault("status", "error")
            extra.setdefault("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            extra.setdefault("status", "ok")
            duration = round((time.perf_counter() - start) * 1000, 1)
            self.log(event, interaction_id, **fields, **extra, duration_ms=duration)

    def _rotate(self) -> None:
        if self.max_bytes and self.path.exists() and self.path.stat().st_size > self.max_bytes:
            self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))


class NullLogger(EventLogger):
    """Logger that keeps nothing (tests, logging disabled)."""

    def __init__(self):
        super().__init__(None)
