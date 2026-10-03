"""Semantic compression of old turns into a rolling summary (M2).

When the history approaches its share of the token budget, the oldest turns
are summarized and replaced by a compact structured summary.

Design choices:
- The model only summarizes the *new* chunk of conversation, as JSON
  constrained by a schema (Ollama `format`). Merging with the previous
  summary is done by code, so a small model cannot silently drop facts that
  were already summarized by rewriting everything.
- Fields are tailored to language learning: what matters to keep is what
  the learner saw, got wrong and said about themselves, not the chit-chat.
- The summary has a hard token cap. When it is exceeded, items are dropped
  deterministically, least useful first (oldest exercises, then vocabulary,
  then mistakes, then learner facts).
"""

import json
import time
from dataclasses import dataclass, field

from ..ollama_client import ChatStats, OllamaClient, OllamaError
from .store import Message
from .tokens import estimate_message

# Appended lists, merged by code. Order = display order.
LIST_FIELDS = {
    "apprenant": "Faits sur l'apprenant : préférences, objectifs, informations personnelles qu'il a données",
    "vocabulaire": "Mots ou expressions de la langue cible vus, au format « mot = traduction »",
    "erreurs": "Erreurs commises par l'apprenant, au format « erreur → correction (règle) »",
    "exercices": "Exercices proposés et résultat (réussi / raté)",
}
# Replaced at each pass: only the latest value is relevant.
TEXT_FIELDS = {
    "resume": "Résumé en 2 à 3 phrases de ce qui s'est passé dans l'extrait",
    "en_cours": "Ce qui était en cours à la fin de l'extrait (exercice non terminé, prochaine étape prévue), ou chaîne vide",
}
DROP_ORDER = ["exercices", "vocabulaire", "erreurs", "apprenant"]

TITLES = {
    "apprenant": "Apprenant",
    "vocabulaire": "Vocabulaire vu",
    "erreurs": "Erreurs à surveiller",
    "exercices": "Exercices faits",
}

SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "array", "items": {"type": "string"}} for k in LIST_FIELDS},
        **{k: {"type": "string"} for k in TEXT_FIELDS},
    },
    "required": [*LIST_FIELDS, *TEXT_FIELDS],
}

SYSTEM_PROMPT = (
    "Tu résumes un extrait de cours de langue entre un tuteur et un apprenant. "
    "Tu réponds uniquement avec du JSON. "
    "N'invente rien : ne note que ce qui apparaît dans l'extrait. "
    "Chaque élément de liste tient sur une ligne courte. Une liste vide est acceptable."
)


class CompressionError(Exception):
    pass


@dataclass
class Summary:
    lists: dict[str, list[str]] = field(default_factory=lambda: {k: [] for k in LIST_FIELDS})
    resume: str = ""
    en_cours: str = ""

    # --- serialization --------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({**self.lists, "resume": self.resume, "en_cours": self.en_cours}, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "Summary":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CompressionError(f"invalid JSON from model: {exc}") from exc
        if not isinstance(data, dict):
            raise CompressionError("model output is not a JSON object")
        summary = cls()
        for key in LIST_FIELDS:
            items = data.get(key) or []
            if not isinstance(items, list):
                items = [items]
            summary.lists[key] = [str(i).strip() for i in items if str(i).strip()]
        summary.resume = str(data.get("resume") or "").strip()
        summary.en_cours = str(data.get("en_cours") or "").strip()
        return summary

    # --- merge / trim / render ------------------------------------------

    def merged_with(self, newer: "Summary") -> "Summary":
        """Lists are appended (deduplicated); text fields come from `newer`."""
        out = Summary(resume=newer.resume or self.resume, en_cours=newer.en_cours)
        for key in LIST_FIELDS:
            seen: set[str] = set()
            merged = []
            for item in self.lists[key] + newer.lists[key]:
                norm = " ".join(item.lower().split())
                if norm not in seen:
                    seen.add(norm)
                    merged.append(item)
            out.lists[key] = merged
        return out

    def trim_to(self, max_tokens: int) -> int:
        """Drop oldest items until the rendered summary fits; return the number dropped."""
        dropped = 0
        while self.tokens() > max_tokens:
            key = next((k for k in DROP_ORDER if self.lists[k]), None)
            if key is None:
                break
            self.lists[key].pop(0)
            dropped += 1
        if self.tokens() > max_tokens:
            # Only the free text is left: cut it.
            excess_chars = (self.tokens() - max_tokens) * 4
            self.resume = self.resume[: max(0, len(self.resume) - excess_chars)].rstrip() + "…"
        return dropped

    def render(self) -> str:
        parts = ["## Résumé des échanges précédents"]
        if self.resume:
            parts.append(self.resume)
        for key, title in TITLES.items():
            if self.lists[key]:
                parts.append(f"### {title}\n" + "\n".join(f"- {i}" for i in self.lists[key]))
        if self.en_cours:
            parts.append(f"### En cours\n{self.en_cours}")
        return "\n\n".join(parts)

    def tokens(self) -> int:
        return estimate_message(self.render())

    def is_empty(self) -> bool:
        return not (self.resume or self.en_cours or any(self.lists.values()))


def message_cost(msg: Message) -> int:
    return msg.tokens or estimate_message(msg.content)


class Compressor:
    def __init__(self, client: OllamaClient, trigger: float, keep: float, chunk: float = 0.5):
        """
        trigger: compress when history exceeds this share of its token budget.
        keep:    share of that budget left untouched (most recent turns).
        chunk:   max share summarized in one model call (long reloaded sessions
                 are compressed in several passes).
        """
        self.client = client
        self.trigger = trigger
        self.keep = keep
        self.chunk = chunk

    def select(self, history: list[Message], available: int, force: bool = False) -> list[Message]:
        """Return the oldest messages to compress (empty list = nothing to do)."""
        total = sum(message_cost(m) for m in history)
        if not force and total <= self.trigger * available:
            return []

        # Keep the most recent turns verbatim; always keep the last message.
        kept_tokens = 0
        split = len(history)
        while split > 1:
            cost = message_cost(history[split - 1])
            if split < len(history) and kept_tokens + cost > self.keep * available:
                break
            kept_tokens += cost
            split -= 1

        # Cap one pass to `chunk` of the budget.
        selected, tokens = [], 0
        for msg in history[:split]:
            cost = message_cost(msg)
            if selected and tokens + cost > self.chunk * available:
                break
            selected.append(msg)
            tokens += cost
        return selected if len(selected) >= 2 else []

    def summarize(self, messages: list[Message], previous: Summary | None) -> Summary:
        """Ask the model to summarize `messages` only; raises CompressionError."""
        transcript = "\n".join(
            f"{'Apprenant' if m.role == 'user' else 'Tuteur'} : {m.content}" for m in messages if m.role != "system"
        )
        fields_doc = "\n".join(f"- {k} : {v}" for k, v in {**LIST_FIELDS, **TEXT_FIELDS}.items())
        context = f"Contexte (résumé précédent, ne pas le recopier) : {previous.resume}\n\n" if previous and previous.resume else ""
        prompt = f"{context}Extrait à résumer :\n{transcript}\n\nChamps JSON attendus :\n{fields_doc}"
        try:
            raw = self.client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                ChatStats(),
                schema=SCHEMA,
                options={"temperature": 0},
                purpose="compression",
            )
        except OllamaError as exc:
            raise CompressionError(str(exc)) from exc
        return Summary.from_json(raw)


@dataclass
class CompressionResult:
    summary: Summary
    covered_until: int
    messages_in: int
    tokens_before: int
    tokens_after: int
    duration_ms: float
    dropped_items: int


def compress(
    compressor: Compressor, messages: list[Message], previous: Summary | None, max_summary_tokens: int
) -> CompressionResult:
    start = time.perf_counter()
    newer = compressor.summarize(messages, previous)
    summary = (previous or Summary()).merged_with(newer)
    dropped = summary.trim_to(max_summary_tokens)
    before = sum(message_cost(m) for m in messages) + (previous.tokens() if previous else 0)
    return CompressionResult(
        summary=summary,
        covered_until=messages[-1].id,
        messages_in=len(messages),
        tokens_before=before,
        tokens_after=summary.tokens(),
        duration_ms=(time.perf_counter() - start) * 1000,
        dropped_items=dropped,
    )
