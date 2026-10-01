"""Builds the message list sent to the model, within the token budget.

Every source of context (system prompt, learner profile, RAG chunks, chat
history...) competes for the same num_ctx window. If we overflow it,
Ollama silently drops the oldest tokens, so this module owns the budget
and decides what goes in.

Priority order:
  1. system prompt + learner profile      (always included)
  2. retrieved documents (RAG hook)       (capped share of the budget)
  3. recent history, newest first         (whatever is left)
A share of num_ctx is kept free for the model's answer.

TODO(M2): summarize the turns that no longer fit instead of dropping them.
TODO(M3): inject relevant learner facts (vocabulary, recurring errors).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from .memory.store import Message
from .memory.tokens import estimate_message

# retriever(query, max_tokens) -> list of text chunks that fit in max_tokens
Retriever = Callable[[str, int], list[str]]


@dataclass
class ContextReport:
    """What went into the prompt; shown by /context."""

    budget: int
    used: int = 0
    system_tokens: int = 0
    rag_tokens: int = 0
    history_tokens: int = 0
    history_kept: int = 0
    history_dropped: int = 0
    sections: list[str] = field(default_factory=list)


class ContextBuilder:
    def __init__(
        self,
        system_prompt: str,
        num_ctx: int,
        response_reserve: float,
        retriever: Retriever | None = None,
        rag_share: float = 0.25,
    ):
        self.system_prompt = system_prompt
        self.budget = int(num_ctx * (1 - response_reserve))
        self.retriever = retriever
        self.rag_share = rag_share

    def build(self, profile: dict[str, str], history: list[Message]) -> tuple[list[dict], ContextReport]:
        """`history` ends with the current user message."""
        report = ContextReport(budget=self.budget)

        system = self._system_message(profile)
        report.system_tokens = estimate_message(system)
        remaining = self.budget - report.system_tokens

        if self.retriever and history:
            chunks = self.retriever(history[-1].content, int(self.budget * self.rag_share))
            if chunks:
                docs = "\n\n".join(chunks)
                system += f"\n\n## Documents de référence\n{docs}"
                report.rag_tokens = estimate_message(docs)
                remaining -= report.rag_tokens
                report.sections.append("rag")

        kept: list[Message] = []
        for msg in reversed(history):
            cost = msg.tokens or estimate_message(msg.content)
            # The current user message is always kept, even if over budget.
            if kept and cost > remaining:
                break
            kept.append(msg)
            remaining -= cost
            report.history_tokens += cost
        kept.reverse()
        report.history_kept = len(kept)
        report.history_dropped = len(history) - len(kept)
        report.used = self.budget - remaining

        messages = [{"role": "system", "content": system}] + [m.to_ollama() for m in kept]
        return messages, report

    def _system_message(self, profile: dict[str, str]) -> str:
        if not profile:
            return self.system_prompt
        lines = "\n".join(f"- {k}: {v}" for k, v in profile.items())
        return f"{self.system_prompt}\n\n## Profil de l'apprenant\n{lines}"
