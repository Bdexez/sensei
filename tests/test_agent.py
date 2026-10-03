"""A1 + integration: the ReAct loop driving the file tools (T4), the sandbox (T2),
the confirmation gate (A3), with every step logged (EV3).

The model is replaced by a scripted fake that returns one JSON step per call
and records what it was sent, so the loop is tested deterministically.
"""

import json
import tempfile
import unittest
from pathlib import Path

from sensai.agent.approval import ApprovalGate, Decision
from sensai.agent.permissions import FileGuard
from sensai.agent.react import FINAL, ReActAgent
from sensai.agent.sandbox import Sandbox
from sensai.agent.tools import ToolRegistry
from sensai.monitoring import compute_metrics, load_events
from sensai.observability import EventLogger
from sensai.ollama_client import OllamaUnavailable


def step(action, summary="étape", answer="", **args):
    return json.dumps({"step_summary": summary, "action": action, "args": args, "answer": answer})


class FakeClient:
    model = "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, stats, schema=None, options=None, purpose="chat", interaction_id=None):
        self.calls.append({"messages": [dict(m) for m in messages], "schema": schema, "purpose": purpose})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class Approver:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.requests = []

    def __call__(self, req):
        self.requests.append(req)
        return Decision(True, "approved") if self.decisions.pop(0) else Decision(False, "refused")


BUGGY = "mots = {'neko': 'chat'}\nfor m, t in mots.items()\n    print(m, '=', t)\n"
FIXED = "mots = {'neko': 'chat'}\nfor m, t in mots.items():\n    print(m, '=', t)\n"


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        (self.base / "docs").mkdir()
        (self.base / "docs" / "lecon.md").write_text("neko = chat\ninu = chien\n")
        (self.base / "secret.txt").write_text("hunter2")
        self.log = self.base / "events.jsonl"
        self.logger = EventLogger(str(self.log))
        self.guard = FileGuard({"workspace": "rw", "docs": "ro"}, allowed_ext=[".md", ".py", ".txt"],
                               base_dir=self.base, logger=self.logger)
        self.sandbox = Sandbox(timeout=5, logger=self.logger)

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, replies, *decisions, max_steps=8, levels=None):
        self.approver = Approver(*decisions)
        self.client = FakeClient(replies)
        self.shown = []
        gate = ApprovalGate(self.approver, levels, self.logger)
        tools = ToolRegistry(self.guard, self.sandbox, gate, self.logger)
        return ReActAgent(self.client, tools, self.logger, max_steps=max_steps, on_step=self.shown.append)

    def observations(self):
        return [m["content"] for m in self.client.calls[-1]["messages"] if m["content"].startswith("Observation")]


class ReActLoopTest(AgentTestCase):
    def test_self_correcting_code_then_confirmed_write(self):
        agent = self.agent(
            [
                step("read_file", "Je lis la leçon", path="docs/lecon.md"),
                step("run_python", "Je teste le script de révision", code=BUGGY),
                step("run_python", "Je corrige la syntaxe (deux-points manquants)", code=FIXED),
                step("write_file", "J'enregistre le script", path="workspace/revision.py", content=FIXED),
                step(FINAL, "Terminé", answer="Script créé dans workspace/revision.py"),
            ],
            True, True, True,  # run, run, write
        )
        result = agent.run("Crée un script de révision du vocabulaire de la leçon")
        self.assertEqual(result.outcome, "done")
        self.assertEqual(result.answer, "Script créé dans workspace/revision.py")
        self.assertEqual([s.status for s in result.steps], ["ok", "ok", "ok", "ok", "final"])
        # The failing run was observed with its traceback, then fixed.
        self.assertEqual(result.steps[1].observation["outcome"], "error")
        self.assertIn("SyntaxError", result.steps[1].observation["stderr"])
        self.assertIn("SyntaxError", self.observations()[1])
        self.assertEqual(result.steps[2].observation["stdout"].strip(), "neko = chat")
        self.assertEqual((self.base / "workspace" / "revision.py").read_text(), FIXED)
        # Reads are not confirmed; the two runs and the write are.
        self.assertEqual([r.action for r in self.approver.requests], ["run_python", "run_python", "write_file"])
        write_req = self.approver.requests[2]
        self.assertEqual(write_req.resources, ["workspace/revision.py"])
        self.assertIn("+mots = {'neko': 'chat'}", write_req.summary)
        self.assertTrue(write_req.risks)
        self.assertEqual(write_req.reason, "J'enregistre le script")
        self.assertEqual(len(self.shown), 5)

    def test_user_refusal_cancels_cleanly(self):
        agent = self.agent(
            [
                step("write_file", "J'écris la fiche", path="workspace/fiche.md", content="neko = chat"),
                step(FINAL, answer="D'accord, je n'ai rien écrit."),
            ],
            False,
        )
        result = agent.run("Écris une fiche")
        self.assertEqual(result.outcome, "done")
        self.assertEqual(result.steps[0].status, "refused")
        self.assertFalse((self.base / "workspace" / "fiche.md").exists())
        self.assertIn("refusée par l'utilisateur", self.observations()[0])

    def test_permission_denied_is_observed_without_asking_user(self):
        agent = self.agent(
            [
                step("read_file", "Je lis le secret", path="../secret.txt"),
                step("write_file", "J'écris dans les docs", path="docs/lecon.md", content="x"),
                step(FINAL, answer="Je n'ai pas accès à ces fichiers."),
            ],
        )
        result = agent.run("Lis secret.txt puis modifie la leçon")
        self.assertEqual([s.status for s in result.steps[:2]], ["denied", "denied"])
        self.assertEqual(self.approver.requests, [])  # never asked to approve a forbidden action
        self.assertNotIn("hunter2", json.dumps([c["messages"] for c in self.client.calls]))
        self.assertIn("neko", (self.base / "docs" / "lecon.md").read_text())

    def test_max_steps(self):
        agent = self.agent([step("list_dir", f"Je regarde {i}", path="docs") for i in range(3)], max_steps=3)
        result = agent.run("Boucle")
        self.assertEqual(result.outcome, "max_steps")
        self.assertEqual(len(result.steps), 3)
        self.assertIn("pas terminé en 3 étapes", result.answer)
        self.assertEqual(self.client.replies, [])

    def test_invalid_steps_are_retried_then_abort(self):
        agent = self.agent(["pas du json", json.dumps({"action": "rm_rf"}), step(FINAL, answer="ok")])
        result = agent.run("x")
        self.assertEqual(result.outcome, "done")
        self.assertEqual([s.status for s in result.steps], ["invalid", "invalid", "final"])
        agent = self.agent(["a", "b", "c", step(FINAL, answer="jamais")])
        result = agent.run("x")
        self.assertEqual(result.outcome, "error")
        self.assertIn("invalides", result.error)

    def test_model_unavailable_is_not_recoverable(self):
        agent = self.agent([step("list_dir", path="docs"), OllamaUnavailable("connection refused")])
        result = agent.run("x")
        self.assertEqual(result.outcome, "error")
        self.assertIn("connection refused", result.answer)
        self.assertEqual(len(result.steps), 1)

    def test_tool_errors_are_recoverable(self):
        agent = self.agent(
            [
                step("read_file", "Je lis"),  # missing path
                step("nope_tool"),  # rejected by the parser
                step("read_file", "Je lis", path="docs/absent.md"),
                step(FINAL, answer="fini"),
            ]
        )
        result = agent.run("x")
        self.assertEqual(result.outcome, "done")
        self.assertEqual([s.status for s in result.steps], ["error", "invalid", "error", "final"])
        self.assertIn("introuvable", result.steps[2].observation["error"])
        self.assertIn("manquant", result.steps[0].observation["error"])

    def test_run_tests_imports_workspace_files_through_guard(self):
        (self.base / "workspace").mkdir(exist_ok=True)
        (self.base / "workspace" / "kana.py").write_text("def romaji(k):\n    return {'あ': 'a'}.get(k, '?')\n")
        (self.base / "workspace" / "test_kana.py").write_text(
            "import unittest\nfrom kana import romaji\nclass T(unittest.TestCase):\n"
            "    def test_a(self):\n        self.assertEqual(romaji('あ'), 'a')\n"
        )
        agent = self.agent(
            [
                step("run_tests", "Je lance les tests", files=["workspace/kana.py", "workspace/test_kana.py"]),
                step("run_tests", "Je tente un fichier hors périmètre", files=["../secret.txt"]),
                step(FINAL, answer="Tests OK"),
            ],
            True,
        )
        result = agent.run("Teste kana.py")
        self.assertEqual(result.steps[0].observation["outcome"], "ok")
        self.assertEqual(result.steps[1].status, "denied")
        self.assertEqual(len(self.approver.requests), 1)

    def test_execute_without_confirmation_when_configured(self):
        agent = self.agent([step("run_python", code="print(1)"), step(FINAL, answer="1")], levels=["write", "delete"])
        result = agent.run("x")
        self.assertEqual(result.steps[0].observation["stdout"].strip(), "1")
        self.assertEqual(self.approver.requests, [])

    def test_delete_requires_confirmation(self):
        (self.base / "workspace").mkdir(exist_ok=True)
        (self.base / "workspace" / "old.md").write_text("vieux")
        agent = self.agent([step("delete_file", "Je supprime", path="workspace/old.md"), step(FINAL, answer="ok")], True)
        agent.run("supprime old.md")
        self.assertFalse((self.base / "workspace" / "old.md").exists())
        self.assertIn("définitive", self.approver.requests[0].risks[0])

    def test_prompt_schema_and_trace(self):
        agent = self.agent([step(FINAL, "Rien à faire", answer="ok")])
        result = agent.run("x", context="Profil : japonais")
        call = self.client.calls[0]
        self.assertEqual(call["purpose"], "react_step")
        self.assertIn("final_answer", call["schema"]["properties"]["action"]["enum"])
        self.assertIn("write_file", call["messages"][0]["content"])
        self.assertIn("lecture seule", call["messages"][0]["content"])
        self.assertIn("Profil : japonais", call["messages"][1]["content"])
        self.assertIn("Rien à faire", result.trace())

    def test_old_observations_are_shortened_to_fit_budget(self):
        (self.base / "docs" / "long.md").write_text("mot = traduction\n" * 300)
        agent = self.agent([step("read_file", path="docs/long.md")] * 3 + [step(FINAL, answer="ok")])
        agent.token_budget = 1500
        agent.run("x")
        first_obs = [m["content"] for m in self.client.calls[-1]["messages"] if m["content"].startswith("Observation")][0]
        self.assertIn("abrégée", first_obs)

    def test_everything_is_logged_and_measured(self):
        agent = self.agent(
            [
                step("read_file", path="docs/lecon.md"),
                step("run_python", code="print('ok')"),
                step("write_file", path="workspace/a.md", content="x"),
                step(FINAL, answer="ok"),
            ],
            True, False,
        )
        result = agent.run("x")
        events = load_events(self.log)
        kinds = {e["event"] for e in events}
        self.assertTrue({"react_step", "react_run", "tool_call", "confirmation", "file_access", "sandbox_run"} <= kinds)
        self.assertTrue(all(e.get("interaction_id") == result.interaction_id
                            for e in events if e["event"] in ("react_step", "react_run", "tool_call", "confirmation")))
        m = compute_metrics(events)
        self.assertEqual(m["react"]["outcomes"], {"done": 1})
        self.assertEqual(m["tools"]["write_file"]["refused"], 1)
        self.assertEqual(m["confirmations"], {"approved": 1, "refused": 1})


if __name__ == "__main__":
    unittest.main()


class CliAgentCommandTest(AgentTestCase):
    def test_agent_trace_tools_commands(self):
        from unittest import mock

        from sensai.cli import Chat
        from sensai.config import Config
        from sensai.memory.store import MemoryStore

        agent = self.agent([step("list_dir", "Je regarde les docs", path="docs"), step(FINAL, answer="2 fichiers")])
        config = Config(db_path=str(self.base / "t.db"), log_path=str(self.log), memory=False)
        store = MemoryStore(config.db_path)
        chat = Chat(config, store, self.client, self.logger, agent)
        with mock.patch("builtins.print") as printed:
            chat.handle_command("/agent liste les docs")
            chat.handle_command("/trace")
            chat.handle_command("/tools")
            chat.handle_command("/agent")
        out = "\n".join(str(c.args[0]) if c.args else "" for c in printed.call_args_list)
        self.assertIn("2 fichiers", out)
        self.assertIn("Je regarde les docs", out)
        self.assertIn("run_python", out)
        self.assertIn("Usage : /agent", out)
        self.assertEqual([m.role for m in chat.history], ["user", "assistant"])
        self.assertEqual(store.count_messages(chat.session_id), 2)
        chat.close()
        store.close()
