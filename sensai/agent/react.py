"""Reasoning loop (A1): ReAct = Reason → Act → Observe, until done.

At each step the model returns one JSON object, constrained by a schema
(Ollama `format`, the same mechanism as M2/M3):

    {"step_summary": "...", "action": "<tool>|final_answer", "args": {...}, "answer": "..."}

The code then runs the action through the ToolRegistry (permissions,
confirmation, sandbox), and sends back the result as an observation. The
model adjusts and goes on until it gives a final answer, a non-recoverable
error happens, or a limit is reached.

Why a JSON step instead of Ollama's native tool calling: it works with any
model (native tool calling depends on the model's template), the schema
forces exactly one action per step, and a malformed step is detected and
retried instead of silently becoming chat text.

Privacy of reasoning: the model is asked for a one-sentence public
`step_summary` (what it does and why), not for its chain of thought, and
`think` stays as configured (off by default). The trace only stores these
summaries, the actions requested and their results.

Limits: `max_steps` actions, `max_parse_errors` malformed steps in a row,
and the conversation is kept within the token budget by shortening the
oldest observations first.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..memory.tokens import estimate_message
from ..observability import EventLogger, NullLogger, new_interaction_id
from ..ollama_client import ChatStats, OllamaClient, OllamaError
from .tools import ToolRegistry

FINAL = "final_answer"

SYSTEM_PROMPT = """Tu es Sensai en mode agent : un assistant qui aide un apprenant en langues à préparer son matériel d'étude (fiches de vocabulaire, exercices, petits programmes Python d'entraînement) dans son espace de travail.

Tu avances par étapes. À chaque étape tu réponds UNIQUEMENT avec un objet JSON :
{{"step_summary": "...", "action": "...", "args": {{...}}, "answer": "..."}}
- step_summary : UNE phrase courte et publique qui dit ce que tu fais et pourquoi (pas ton raisonnement détaillé).
- action : le nom d'un outil ci-dessous, ou "final_answer" quand la tâche est terminée.
- args : les arguments de l'outil (objet vide pour final_answer).
- answer : seulement pour final_answer, la réponse finale pour l'apprenant (en français), sinon chaîne vide.

Outils :
{tools}

Règles :
- Une seule action par étape. Après chaque action tu reçois une « Observation » : appuie-toi dessus.
- Si du code échoue, lis stderr, corrige le code et relance (sans dépasser le nombre d'étapes).
- Si une permission est refusée, ne cherche pas à la contourner (autre chemin, lien, ..) : adapte-toi ou explique.
- Si l'utilisateur refuse une action, ne la redemande pas : propose une alternative ou termine.
- N'invente pas le contenu d'un fichier : lis-le.
- Termine dès que possible avec final_answer.

{permissions}
{sandbox}
Tu disposes d'au plus {max_steps} étapes."""


@dataclass
class Step:
    index: int
    summary: str
    action: str
    args: dict = field(default_factory=dict)
    status: str = "ok"  # ok | error | denied | refused | invalid | final
    observation: dict = field(default_factory=dict)
    duration_ms: float = 0.0

    def line(self) -> str:
        args = ", ".join(f"{k}={_short(v)}" for k, v in self.args.items() if v not in ("", [], None))
        mark = {"ok": "✓", "final": "✓", "refused": "✗ refusé", "denied": "✗ permission", "invalid": "⚠ invalide"}.get(
            self.status, "✗ erreur"
        )
        call = self.action if self.action == FINAL else f"{self.action}({args})"
        return f"[étape {self.index}] {self.summary or '—'}\n           → {call} {mark} ({self.duration_ms:.0f} ms)"


@dataclass
class AgentResult:
    outcome: str  # done | max_steps | error
    answer: str
    steps: list[Step]
    interaction_id: str
    duration_ms: float = 0.0
    error: str | None = None

    def trace(self) -> str:
        lines = [f"Interaction {self.interaction_id} — {self.outcome}, {len(self.steps)} étapes, {self.duration_ms / 1000:.1f} s"]
        for step in self.steps:
            lines.append(step.line())
            if step.status not in ("ok", "final") and step.observation.get("error"):
                lines.append(f"           {step.observation['error']}")
            elif step.observation.get("outcome"):
                lines.append(f"           sandbox : {step.observation['outcome']}, code {step.observation.get('exit_code')}")
        return "\n".join(lines)


def _short(value, limit: int = 40) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = text.replace("\n", "⏎")
    return text if len(text) <= limit else text[:limit] + "…"


class ReActAgent:
    def __init__(
        self,
        client: OllamaClient,
        tools: ToolRegistry,
        logger: EventLogger | None = None,
        max_steps: int = 8,
        max_parse_errors: int = 2,
        observation_chars: int = 2000,
        token_budget: int = 6000,
        on_step: Callable[[Step], None] | None = None,
    ):
        self.client = client
        self.tools = tools
        self.logger = logger or NullLogger()
        self.max_steps = max_steps
        self.max_parse_errors = max_parse_errors
        self.observation_chars = observation_chars
        self.token_budget = token_budget
        self.on_step = on_step or (lambda step: None)

    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "step_summary": {"type": "string"},
                "action": {"type": "string", "enum": [*self.tools.names(), FINAL]},
                "args": {"type": "object", "properties": self.tools.arg_schema()},
                "answer": {"type": "string"},
            },
            "required": ["step_summary", "action", "args", "answer"],
        }

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            tools=self.tools.describe(),
            permissions=self.tools.guard.describe(),
            sandbox=self.tools.sandbox.describe(),
            max_steps=self.max_steps,
        )

    def run(self, task: str, context: str | None = None, interaction_id: str | None = None) -> AgentResult:
        interaction_id = interaction_id or new_interaction_id()
        start = time.perf_counter()
        user = f"{context}\n\nTâche : {task}" if context else f"Tâche : {task}"
        messages = [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": user}]
        steps: list[Step] = []
        parse_errors = 0
        outcome, answer, error = "max_steps", "", None

        for index in range(1, self.max_steps + 1):
            step_start = time.perf_counter()
            self._fit(messages)
            try:
                raw = self.client.chat(
                    messages, ChatStats(), schema=self.schema(), options={"temperature": 0.2},
                    purpose="react_step", interaction_id=interaction_id,
                )
            except OllamaError as exc:
                outcome, error = "error", f"modèle indisponible : {exc}"
                break

            parsed, problem = self._parse(raw)
            if problem:
                parse_errors += 1
                step = Step(index, "", "?", status="invalid", observation={"error": problem})
                self._record(step, step_start, interaction_id)
                steps.append(step)
                if parse_errors > self.max_parse_errors:
                    outcome, error = "error", f"réponses du modèle invalides ({problem})"
                    break
                messages.append({"role": "assistant", "content": raw[:500]})
                messages.append({"role": "user", "content": f"Réponse invalide : {problem}. Réponds avec un seul objet JSON au format demandé."})
                continue
            parse_errors = 0
            messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})

            if parsed["action"] == FINAL:
                answer = parsed["answer"].strip() or parsed["step_summary"]
                step = Step(index, parsed["step_summary"], FINAL, status="final")
                self._record(step, step_start, interaction_id)
                steps.append(step)
                outcome = "done"
                break

            result = self.tools.call(parsed["action"], parsed["args"], parsed["step_summary"], interaction_id)
            step = Step(index, parsed["step_summary"], parsed["action"], self._display_args(parsed["args"]),
                        result.status, result.observation)
            self._record(step, step_start, interaction_id)
            steps.append(step)
            observation = json.dumps(result.observation, ensure_ascii=False)
            if len(observation) > self.observation_chars:
                observation = observation[: self.observation_chars] + "… [observation tronquée]"
            messages.append({"role": "user", "content": f"Observation ({result.status}) : {observation}"})

        if outcome == "max_steps":
            answer = (
                f"Je n'ai pas terminé en {self.max_steps} étapes. "
                "Voici où j'en suis : " + "; ".join(s.summary for s in steps if s.summary)
            )
        elif outcome == "error" and not answer:
            answer = f"Arrêt de la boucle : {error}"
        duration = round((time.perf_counter() - start) * 1000, 1)
        self.logger.log(
            "react_run", interaction_id, outcome=outcome, steps=len(steps), duration_ms=duration,
            status="ok" if outcome == "done" else ("error" if outcome == "error" else "limit"),
            error=error, task_chars=len(task), model=self.client.model,
        )
        return AgentResult(outcome, answer, steps, interaction_id, duration, error)

    # --- helpers --------------------------------------------------------------

    def _parse(self, raw: str) -> tuple[dict, str | None]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, f"JSON invalide ({exc.msg})"
        if not isinstance(data, dict):
            return {}, "la réponse n'est pas un objet JSON"
        action = data.get("action")
        if action not in (*self.tools.names(), FINAL):
            return {}, f"action inconnue « {action} »"
        args = data.get("args")
        return {
            "step_summary": str(data.get("step_summary") or "").strip()[:300],
            "action": action,
            "args": args if isinstance(args, dict) else {},
            "answer": str(data.get("answer") or ""),
        }, None

    @staticmethod
    def _display_args(args: dict) -> dict:
        return {k: v for k, v in args.items() if v not in ("", [], None)}

    def _record(self, step: Step, step_start: float, interaction_id: str) -> None:
        step.duration_ms = round((time.perf_counter() - step_start) * 1000, 1)
        self.logger.log(
            "react_step", interaction_id, index=step.index, action=step.action, status=step.status,
            summary=step.summary, duration_ms=step.duration_ms, error=step.observation.get("error"),
        )
        self.on_step(step)

    def _fit(self, messages: list[dict]) -> None:
        """Shorten the oldest observations until the conversation fits the token budget."""
        def total() -> int:
            return sum(estimate_message(m["content"]) for m in messages)

        for msg in messages[2:-2]:
            if total() <= self.token_budget:
                return
            if msg["role"] == "user" and msg["content"].startswith("Observation") and len(msg["content"]) > 200:
                msg["content"] = msg["content"][:200] + "… [ancienne observation abrégée]"
