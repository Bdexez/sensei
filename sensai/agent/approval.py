"""Human-in-the-loop confirmation of sensitive actions (A3).

Before a tool with a sensitive level runs (write, delete, execute), the agent
suspends and the user sees: the proposed action, the files or resources it
touches, a summary of the change (diff, code excerpt), the main risks and why
the agent wants to do it. Only an explicit "o"/"oui"/"y"/"yes" proceeds;
anything else — "n", an empty line, Ctrl-D, Ctrl-C, no answer before the
timeout — cancels the action, and the agent is told it was refused.

Reads (list, read) are never confirmed: they cannot change anything and are
already restricted by the permission layer (T4).
"""

import select
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..observability import EventLogger, NullLogger

LEVELS = ("read", "write", "delete", "execute")
MANDATORY = {"write", "delete"}  # actions that modify the project are always confirmed
YES = {"o", "oui", "y", "yes"}


@dataclass
class ApprovalRequest:
    action: str  # tool name
    title: str  # one line, e.g. "Écrire workspace/fiches.md (nouveau fichier)"
    level: str
    resources: list[str] = field(default_factory=list)
    summary: str = ""  # diff, code excerpt...
    risks: list[str] = field(default_factory=list)
    reason: str = ""  # the agent's own summary of the step


@dataclass
class Decision:
    approved: bool
    reason: str  # approved | refused | no_answer | timeout | not_required


def render_request(req: ApprovalRequest) -> str:
    bar = "─" * 60
    lines = [f"┌{bar}", "│ ⚠  Confirmation requise", f"│ Action     : {req.action} — {req.title}"]
    if req.resources:
        lines.append(f"│ Ressources : {', '.join(req.resources)}")
    if req.reason:
        lines.append(f"│ Motif      : {req.reason}")
    if req.summary:
        lines.append("│ Changement :")
        lines += [f"│   {line}" for line in req.summary.splitlines()]
    if req.risks:
        lines.append("│ Risques    :")
        lines += [f"│   - {risk}" for risk in req.risks]
    lines.append(f"└{bar}")
    return "\n".join(lines)


class ConsoleApprover:
    """Asks on the terminal. `input_fn` / `output` are injectable for tests."""

    def __init__(
        self,
        input_fn: Callable[[str], str] | None = None,
        output: Callable[[str], None] = print,
        timeout: float = 0,
    ):
        self.input_fn = input_fn
        self.output = output
        self.timeout = timeout

    def __call__(self, req: ApprovalRequest) -> Decision:
        self.output(render_request(req))
        prompt = "Confirmer ? [o/N] "
        try:
            if self.input_fn is not None:
                answer = self.input_fn(prompt)
            else:
                answer = self._read(prompt)
        except (EOFError, KeyboardInterrupt):
            self.output("\n→ pas de réponse : action annulée")
            return Decision(False, "no_answer")
        except TimeoutError:
            self.output(f"\n→ pas de réponse en {self.timeout:g} s : action annulée")
            return Decision(False, "timeout")
        if answer.strip().lower() in YES:
            self.output("→ confirmé")
            return Decision(True, "approved")
        self.output("→ refusé : action annulée")
        return Decision(False, "refused")

    def _read(self, prompt: str) -> str:
        if not self.timeout or not hasattr(sys.stdin, "fileno"):
            return input(prompt)
        print(prompt, end="", flush=True)
        try:
            ready, _, _ = select.select([sys.stdin], [], [], self.timeout)
        except (OSError, ValueError):
            return input()
        if not ready:
            raise TimeoutError
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line


class ApprovalGate:
    """Decides whether a request needs confirmation, asks, and logs the outcome."""

    def __init__(
        self,
        approver: Callable[[ApprovalRequest], Decision],
        confirm_levels: list[str] | set[str] | None = None,
        logger: EventLogger | None = None,
    ):
        levels = set(confirm_levels if confirm_levels is not None else ("write", "delete", "execute"))
        unknown = levels - set(LEVELS)
        if unknown:
            raise ValueError(f"unknown confirmation levels {sorted(unknown)}")
        self.levels = levels | MANDATORY
        self.approver = approver
        self.logger = logger or NullLogger()

    def needs_confirmation(self, level: str) -> bool:
        return level in self.levels

    def check(self, req: ApprovalRequest, interaction_id: str | None = None) -> Decision:
        if not self.needs_confirmation(req.level):
            return Decision(True, "not_required")
        start = time.perf_counter()
        try:
            decision = self.approver(req)
        except Exception as exc:  # a broken approver must never mean "yes"
            decision = Decision(False, f"error: {exc}")
        self.logger.log(
            "confirmation",
            interaction_id,
            action=req.action,
            level=req.level,
            resources=req.resources,
            decision=decision.reason if not decision.reason.startswith("error") else "error",
            approved=decision.approved,
            wait_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return decision
