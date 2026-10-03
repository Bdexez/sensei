"""A3: confirmation gate — explicit yes only, everything else cancels."""

import tempfile
import unittest
from pathlib import Path

from sensai.agent.approval import ApprovalGate, ApprovalRequest, ConsoleApprover, Decision, render_request
from sensai.monitoring import load_events
from sensai.observability import EventLogger


def request(level="write"):
    return ApprovalRequest(
        action="write_file",
        title="Écrire workspace/fiches.md (nouveau fichier)",
        level=level,
        resources=["workspace/fiches.md"],
        summary="+neko = chat",
        risks=["crée un fichier dans le projet"],
        reason="Je sauvegarde la fiche de vocabulaire",
    )


def answers(*replies):
    it = iter(replies)

    def fn(_prompt):
        value = next(it)
        if isinstance(value, BaseException):
            raise value
        return value

    return fn


class ConsoleApproverTest(unittest.TestCase):
    def ask(self, reply):
        shown = []
        decision = ConsoleApprover(input_fn=answers(reply), output=shown.append)(request())
        return decision, "\n".join(shown)

    def test_request_shows_action_resources_change_risks(self):
        _, shown = self.ask("o")
        for part in ("write_file", "workspace/fiches.md", "+neko = chat", "crée un fichier", "Je sauvegarde"):
            self.assertIn(part, shown)

    def test_only_explicit_yes_approves(self):
        for reply in ("o", "oui", "Y", " yes "):
            with self.subTest(reply=reply):
                self.assertEqual(self.ask(reply)[0], Decision(True, "approved"))
        for reply in ("", "n", "non", "ok", "peut-être", "oo"):
            with self.subTest(reply=reply):
                self.assertEqual(self.ask(reply)[0], Decision(False, "refused"))

    def test_eof_and_interrupt_cancel(self):
        self.assertEqual(self.ask(EOFError())[0], Decision(False, "no_answer"))
        self.assertEqual(self.ask(KeyboardInterrupt())[0], Decision(False, "no_answer"))

    def test_timeout_cancels(self):
        self.assertEqual(self.ask(TimeoutError())[0], Decision(False, "timeout"))

    def test_render(self):
        text = render_request(ApprovalRequest("run_python", "Exécuter du code", "execute"))
        self.assertIn("Confirmation requise", text)
        self.assertNotIn("Risques", text)


class ApprovalGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "e.jsonl"
        self.asked = []

    def tearDown(self):
        self.tmp.cleanup()

    def gate(self, decision, levels=None):
        def approver(req):
            self.asked.append(req)
            return decision

        return ApprovalGate(approver, levels, EventLogger(str(self.log)))

    def test_reads_are_not_confirmed(self):
        gate = self.gate(Decision(False, "refused"))
        self.assertEqual(gate.check(request("read")), Decision(True, "not_required"))
        self.assertEqual(self.asked, [])

    def test_write_and_delete_always_confirmed(self):
        gate = self.gate(Decision(True, "approved"), levels=[])
        self.assertTrue(gate.needs_confirmation("write"))
        self.assertTrue(gate.needs_confirmation("delete"))
        self.assertFalse(gate.needs_confirmation("execute"))

    def test_execute_confirmation_is_configurable(self):
        self.assertTrue(self.gate(Decision(True, "approved")).needs_confirmation("execute"))

    def test_decision_is_logged(self):
        self.gate(Decision(False, "refused")).check(request(), interaction_id="i1")
        (event,) = load_events(self.log)
        self.assertEqual((event["event"], event["decision"], event["approved"]), ("confirmation", "refused", False))
        self.assertEqual(event["resources"], ["workspace/fiches.md"])
        self.assertEqual(event["interaction_id"], "i1")

    def test_broken_approver_means_no(self):
        def broken(_req):
            raise RuntimeError("terminal closed")

        decision = ApprovalGate(broken, logger=EventLogger(str(self.log))).check(request())
        self.assertFalse(decision.approved)
        self.assertEqual(load_events(self.log)[0]["decision"], "error")

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            ApprovalGate(lambda r: Decision(True, "approved"), ["admin"])


if __name__ == "__main__":
    unittest.main()
