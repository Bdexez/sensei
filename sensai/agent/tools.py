"""Tools the ReAct agent may call, and the single place where they are run.

Every call goes through ToolRegistry.call(), which in order:
1. rejects unknown tools and malformed arguments;
2. for sensitive levels, builds the confirmation request — the permission
   check happens here first, so the user is never asked to approve something
   the permission layer would refuse anyway — and asks the user (A3);
3. runs the handler; file tools only use FileGuard (T4), code tools only use
   the Sandbox (T2), and workspace files given to the sandbox are read
   through FileGuard as well;
4. turns every outcome (success, error, permission denied, user refusal) into
   a structured observation for the model, and logs a `tool_call` event.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from ..observability import EventLogger, NullLogger
from .approval import ApprovalGate, ApprovalRequest
from .permissions import FileGuard, FileOperationError, PermissionDenied
from .sandbox import Sandbox, SandboxError


class ToolError(Exception):
    """Bad arguments or a recoverable failure the model can fix."""


@dataclass
class Param:
    type: str  # "string" | "array"
    description: str
    required: bool = True


@dataclass
class Tool:
    name: str
    description: str
    level: str  # read | write | delete | execute
    params: dict[str, Param]
    handler: Callable[[dict, str | None], dict]
    confirm: Callable[[dict], ApprovalRequest] | None = None


@dataclass
class ToolOutcome:
    status: str  # ok | error | denied | refused
    observation: dict = field(default_factory=dict)


def _excerpt(text: str, max_lines: int = 25) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"… ({len(lines) - max_lines} lignes de plus)"]
    return "\n".join(lines)


class ToolRegistry:
    def __init__(
        self,
        guard: FileGuard,
        sandbox: Sandbox,
        gate: ApprovalGate,
        logger: EventLogger | None = None,
        max_read_chars: int = 6000,
    ):
        self.guard = guard
        self.sandbox = sandbox
        self.gate = gate
        self.logger = logger or NullLogger()
        self.max_read_chars = max_read_chars
        self.tools: dict[str, Tool] = {}
        for tool in self._builtin_tools():
            self.tools[tool.name] = tool

    # --- public -------------------------------------------------------------

    def names(self) -> list[str]:
        return list(self.tools)

    def describe(self) -> str:
        lines = []
        for t in self.tools.values():
            args = ", ".join(f"{n}{'' if p.required else '?'}: {p.type} — {p.description}" for n, p in t.params.items())
            confirm = " [confirmation de l'utilisateur]" if self.gate.needs_confirmation(t.level) else ""
            lines.append(f"- {t.name}({args}) : {t.description}{confirm}")
        return "\n".join(lines)

    def arg_schema(self) -> dict:
        """JSON-schema properties covering every tool argument (for the step schema)."""
        props: dict = {}
        for t in self.tools.values():
            for name, p in t.params.items():
                props[name] = {"type": "array", "items": {"type": "string"}} if p.type == "array" else {"type": "string"}
        return props

    def call(self, name: str, args: dict, reason: str = "", interaction_id: str | None = None) -> ToolOutcome:
        start = time.perf_counter()
        tool = self.tools.get(name)
        outcome = self._call(tool, name, args if isinstance(args, dict) else {}, reason, interaction_id)
        self.logger.log(
            "tool_call",
            interaction_id,
            tool=name,
            level=tool.level if tool else None,
            status=outcome.status,
            args=args,
            error=outcome.observation.get("error"),
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return outcome

    # --- dispatch -------------------------------------------------------------

    def _call(self, tool: Tool | None, name: str, args: dict, reason: str, interaction_id: str | None) -> ToolOutcome:
        if tool is None:
            return ToolOutcome("error", {"error": f"outil inconnu « {name} »", "outils": self.names()})
        try:
            args = self._validate(tool, args)
            if self.gate.needs_confirmation(tool.level):
                request = tool.confirm(args) if tool.confirm else ApprovalRequest(tool.name, tool.name, tool.level)
                request.reason = reason
                decision = self.gate.check(request, interaction_id)
                if not decision.approved:
                    return ToolOutcome(
                        "refused",
                        {
                            "error": "action refusée par l'utilisateur, elle n'a pas été exécutée",
                            "decision": decision.reason,
                            "consigne": "ne recommence pas la même action ; propose autre chose ou termine",
                        },
                    )
            return ToolOutcome("ok", tool.handler(args, interaction_id))
        except PermissionDenied as exc:
            return ToolOutcome(
                "denied",
                {"error": f"permission refusée : {exc.reason}", "path": exc.path, "regles": self.guard.describe()},
            )
        except FileOperationError as exc:
            return ToolOutcome("error", {"error": exc.reason, "path": exc.path})
        except (ToolError, SandboxError) as exc:
            return ToolOutcome("error", {"error": str(exc)})
        except Exception as exc:  # a tool bug must not kill the loop
            return ToolOutcome("error", {"error": f"erreur interne de l'outil : {type(exc).__name__}: {exc}"})

    def _validate(self, tool: Tool, args: dict) -> dict:
        clean = {}  # arguments of other tools are ignored
        for name, p in tool.params.items():
            value = args.get(name)
            if value is None or value == "" or value == []:
                if p.required:
                    raise ToolError(f"argument « {name} » manquant pour {tool.name}")
                clean[name] = [] if p.type == "array" else ""
                continue
            if p.type == "array":
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ToolError(f"« {name} » doit être une liste de chaînes")
            elif not isinstance(value, str):
                raise ToolError(f"« {name} » doit être une chaîne")
            clean[name] = value
        return clean

    # --- tools ------------------------------------------------------------------

    def _builtin_tools(self) -> list[Tool]:
        return [
            Tool(
                "list_dir", "liste un répertoire autorisé (chemin vide = racines autorisées)", "read",
                {"path": Param("string", "répertoire, ex. workspace", required=False)},
                self._list_dir,
            ),
            Tool(
                "read_file", "lit un fichier texte autorisé", "read",
                {"path": Param("string", "chemin du fichier, ex. docs/lecon.md")},
                self._read_file,
            ),
            Tool(
                "write_file", "crée ou remplace un fichier texte (contenu complet)", "write",
                {"path": Param("string", "chemin du fichier"), "content": Param("string", "contenu complet du fichier")},
                self._write_file, self._confirm_write,
            ),
            Tool(
                "delete_file", "supprime un fichier", "delete",
                {"path": Param("string", "chemin du fichier")},
                self._delete_file, self._confirm_delete,
            ),
            Tool(
                "run_python", "exécute du code Python dans une sandbox et renvoie stdout, stderr et le code de sortie",
                "execute",
                {
                    "code": Param("string", "programme Python complet"),
                    "files": Param("array", "fichiers du workspace à copier à côté du programme", required=False),
                },
                self._run_python, self._confirm_run_python,
            ),
            Tool(
                "run_tests", "lance les tests unittest (fichiers test_*.py) dans une sandbox", "execute",
                {"files": Param("array", "fichiers du workspace : tests et modules testés")},
                self._run_tests, self._confirm_run_tests,
            ),
        ]

    def _list_dir(self, args: dict, _iid) -> dict:
        entries = self.guard.list_dir(args["path"])
        return {
            "path": args["path"] or "(racines)",
            "entries": [f"{e.name}/" if e.is_dir else f"{e.name} ({e.size} o)" for e in entries],
        }

    def _read_file(self, args: dict, _iid) -> dict:
        text = self.guard.read_text(args["path"])
        out = {"path": args["path"], "content": text[: self.max_read_chars]}
        if len(text) > self.max_read_chars:
            out["truncated"] = f"{len(text) - self.max_read_chars} caractères non affichés"
        return out

    def _write_file(self, args: dict, _iid) -> dict:
        existed = self.guard.exists(args["path"])
        size = self.guard.write_text(args["path"], args["content"])
        return {"path": args["path"], "bytes": size, "action": "remplacé" if existed else "créé"}

    def _delete_file(self, args: dict, _iid) -> dict:
        self.guard.delete(args["path"])
        return {"path": args["path"], "action": "supprimé"}

    def _workspace_files(self, paths: list[str]) -> dict[str, str]:
        """Copy workspace files into the sandbox, through the permission layer."""
        files: dict[str, str] = {}
        for path in paths:
            name = PurePosixPath(path).name
            if name in files:
                raise ToolError(f"deux fichiers portent le même nom « {name} »")
            files[name] = self.guard.read_text(path)
        return files

    def _sandbox_observation(self, result) -> dict:
        obs = {"outcome": result.outcome, "exit_code": result.exit_code, "duration_ms": result.duration_ms}
        if result.stdout:
            obs["stdout"] = result.stdout
        if result.stderr:
            obs["stderr"] = result.stderr
        if not result.ok:
            obs["consigne"] = "analyse l'erreur, corrige le code et relance"
        return obs

    def _run_python(self, args: dict, iid) -> dict:
        files = self._workspace_files(args["files"])
        if "main.py" in files:
            raise ToolError("« main.py » est réservé au programme exécuté")
        return self._sandbox_observation(self.sandbox.run("python", args["code"], files, iid))

    def _run_tests(self, args: dict, iid) -> dict:
        files = self._workspace_files(args["files"])
        if not any(PurePosixPath(n).name.startswith("test") for n in files):
            raise ToolError("aucun fichier de test (test*.py) parmi les fichiers donnés")
        return self._sandbox_observation(self.sandbox.run("python-unittest", files=files, interaction_id=iid))

    # --- confirmation requests (permission is checked first) ---------------------

    def _confirm_write(self, args: dict) -> ApprovalRequest:
        path = args["path"]
        existed = self.guard.exists(path)
        self.guard.check("write" if existed else "create", path)
        risks = ["le fichier existant sera remplacé (contenu actuel perdu)"] if existed else ["crée un nouveau fichier dans le projet"]
        lines = args["content"].count("\n") + 1
        return ApprovalRequest(
            "write_file",
            f"{'Remplacer' if existed else 'Créer'} {path} ({lines} lignes, {len(args['content'].encode())} o)",
            "write",
            [path],
            self.guard.diff_preview(path, args["content"]),
            risks,
        )

    def _confirm_delete(self, args: dict) -> ApprovalRequest:
        path = args["path"]
        self.guard.check("delete", path)
        if not self.guard.exists(path):
            raise ToolError(f"fichier introuvable : {path}")
        content = self.guard.read_text(path)
        return ApprovalRequest(
            "delete_file", f"Supprimer {path} ({len(content.encode())} o)", "delete", [path],
            _excerpt(content, 10), ["suppression définitive : pas de corbeille ni d'annulation"],
        )

    def _execute_risks(self) -> list[str]:
        risks = [f"exécute du code généré par le modèle (limite {self.sandbox.timeout:g} s)"]
        if self.sandbox.backend == "subprocess":
            risks.append("sandbox processus : même utilisateur, accès réseau et lecture hors projet possibles")
        return risks

    def _confirm_run_python(self, args: dict) -> ApprovalRequest:
        for path in args["files"]:
            self.guard.check("read", path)
        return ApprovalRequest(
            "run_python", "Exécuter un programme Python dans la sandbox", "execute",
            ["sandbox:python", *args["files"]], _excerpt(args["code"]), self._execute_risks(),
        )

    def _confirm_run_tests(self, args: dict) -> ApprovalRequest:
        for path in args["files"]:
            self.guard.check("read", path)
        return ApprovalRequest(
            "run_tests", f"Lancer les tests de {len(args['files'])} fichier(s) dans la sandbox", "execute",
            ["sandbox:python-unittest", *args["files"]], "", self._execute_risks(),
        )
