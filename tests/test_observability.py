"""EV3: structured logging, secret masking and metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from sensai.monitoring import compute_metrics, load_events, main as monitoring_main, render_metrics
from sensai.observability import MASK, EventLogger, redact, sanitize, truncate
from sensai.ollama_client import ChatStats, OllamaClient, OllamaUnavailable


class RedactTest(unittest.TestCase):
    def test_masks_known_secret_formats(self):
        cases = {
            "Authorization: Bearer abc.def-123": "abc.def-123",
            "token ghp_" + "a" * 36: "ghp_",
            "key sk-" + "b" * 32: "sk-",
            "AKIAABCDEFGHIJKLMNOP": "AKIA",
            "password=hunter2": "hunter2",
            "api_key: 'xyz987'": "xyz987",
            "https://bob:s3cr3t@example.com/repo": "s3cr3t",
            "mail victor@example.com": "victor@example.com",
            "carte 4111 1111 1111 1111": "4111 1111",
            "tel 06 12 34 56 78": "06 12 34 56 78",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----": "MIIE",
        }
        for text, secret in cases.items():
            with self.subTest(text=text):
                self.assertNotIn(secret, redact(text))

    def test_keeps_ordinary_text(self):
        text = "Le modèle a utilisé 120 tokens pour traduire « neko » en 1,2 s"
        self.assertEqual(redact(text), text)

    def test_sensitive_keys_are_masked_recursively(self):
        out = sanitize({"args": {"password": "p", "path": "notes.md"}, "api_key": "k"}, 50)
        self.assertEqual(out["api_key"], MASK)
        self.assertEqual(out["args"]["password"], MASK)
        self.assertEqual(out["args"]["path"], "notes.md")

    def test_truncation(self):
        self.assertEqual(truncate("abcdef", 3), "abc… [+3 chars]")
        self.assertEqual(truncate("abcdef", 0), "<6 chars>")
        self.assertEqual(truncate("abc", 10), "abc")
        long_list = sanitize(list(range(30)), 10)
        self.assertEqual(len(long_list), 21)


class EventLoggerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "logs" / "events.jsonl"
        self.logger = EventLogger(str(self.path), content_chars=20)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_one_json_object_per_event(self):
        self.logger.log("tool_call", "abc", tool="read_file", content="x" * 100, token="secret-value")
        self.logger.log("tool_call", "abc", tool="write_file")
        events = load_events(self.path)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["interaction_id"], "abc")
        self.assertEqual(events[0]["token"], MASK)
        self.assertLess(len(events[0]["content"]), 50)
        self.assertNotIn("secret-value", self.path.read_text())

    def test_timed_records_duration_and_errors(self):
        with self.logger.timed("step", "i1") as ev:
            ev["detail"] = "ok"
        with self.assertRaises(ValueError):
            with self.logger.timed("step", "i1"):
                raise ValueError("boom")
        ok, err = load_events(self.path)
        self.assertEqual(ok["status"], "ok")
        self.assertIn("duration_ms", ok)
        self.assertEqual(err["status"], "error")
        self.assertIn("boom", err["error"])

    def test_rotation(self):
        logger = EventLogger(str(self.path), max_bytes=200)
        for _ in range(20):
            logger.log("x", detail="y" * 30)
        self.assertTrue(self.path.with_suffix(".jsonl.1").exists())
        self.assertLess(self.path.stat().st_size, 400)

    def test_io_errors_never_raise(self):
        logger = EventLogger(str(self.path))
        self.path.mkdir(parents=True)  # the log path is now a directory: writes fail
        logger.log("x")
        self.assertEqual(logger.write_errors, 1)

    def test_corrupted_lines_are_skipped(self):
        self.logger.log("a")
        with self.path.open("a") as f:
            f.write("{not json\n")
        self.logger.log("b")
        self.assertEqual([e["event"] for e in load_events(self.path)], ["a", "b"])


class MetricsTest(unittest.TestCase):
    def events(self):
        return [
            {"ts": 1_700_000_000, "event": "model_call", "interaction_id": "a", "model": "m", "status": "ok",
             "duration_ms": 100, "prompt_tokens": 50, "output_tokens": 10},
            {"ts": 1_700_000_001, "event": "model_call", "interaction_id": "b", "model": "m", "status": "timeout",
             "duration_ms": 300},
            {"ts": 1_700_100_000, "event": "model_call", "interaction_id": "b", "model": "m", "status": "ok",
             "duration_ms": 200, "prompt_tokens": 70, "output_tokens": 30},
            {"ts": 1_700_000_002, "event": "tool_call", "interaction_id": "b", "tool": "read_file", "status": "ok", "duration_ms": 2},
            {"ts": 1_700_000_003, "event": "tool_call", "interaction_id": "b", "tool": "write_file", "status": "refused", "duration_ms": 1},
            {"ts": 1_700_000_004, "event": "file_access", "op": "read", "allowed": True},
            {"ts": 1_700_000_005, "event": "file_access", "op": "read", "allowed": False},
            {"ts": 1_700_000_006, "event": "confirmation", "decision": "refused"},
            {"ts": 1_700_000_007, "event": "sandbox_run", "outcome": "timeout", "status": "timeout", "duration_ms": 1000},
            {"ts": 1_700_000_008, "event": "react_run", "outcome": "done", "steps": 3},
        ]

    def test_aggregates(self):
        m = compute_metrics(self.events())
        self.assertEqual(m["interactions"], 2)
        self.assertEqual(m["model_calls"]["count"], 3)
        self.assertEqual(m["model_calls"]["errors"], 1)
        self.assertEqual(m["model_calls"]["prompt_tokens"], 120)
        self.assertEqual(m["timeouts"], 2)
        self.assertEqual(m["tools"]["write_file"]["refused"], 1)
        self.assertEqual(m["file_access"], {"allowed": 1, "denied": 1, "by_op": {"read:ok": 1, "read:denied": 1}})
        self.assertEqual(m["confirmations"], {"refused": 1})
        self.assertEqual(m["react"]["avg_steps"], 3)
        self.assertEqual(len(m["per_day"]), 2)
        text = render_metrics(m)
        for part in ("Appels modèle : 3", "Outils", "Sandbox", "Confirmations", "Évolution par jour"):
            self.assertIn(part, text)

    def test_empty(self):
        self.assertIn("Aucun", render_metrics(compute_metrics([])))

    def test_cli_entry_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in self.events()))
            with mock.patch("builtins.print") as printed:
                self.assertEqual(monitoring_main(["--log", str(path), "--json"]), 0)
            self.assertEqual(json.loads(printed.call_args[0][0])["events"], 10)


class FakeResponse:
    def __init__(self, status=200, payload=None, lines=None):
        self.status_code = status
        self.payload = payload or {}
        self.lines = lines or []

    def json(self):
        return self.payload

    def iter_lines(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ModelCallLoggingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "e.jsonl"
        self.client = OllamaClient("http://x", "tiny", 2048, logger=EventLogger(str(self.path)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_chat_logs_model_and_tokens_without_content(self):
        payload = {"message": {"content": "secret answer"}, "prompt_eval_count": 12, "eval_count": 3}
        with mock.patch("requests.post", return_value=FakeResponse(payload=payload)):
            self.client.chat([{"role": "user", "content": "my private question"}], ChatStats(), interaction_id="i9")
        (event,) = load_events(self.path)
        self.assertEqual(event["event"], "model_call")
        self.assertEqual((event["model"], event["prompt_tokens"], event["output_tokens"]), ("tiny", 12, 3))
        self.assertEqual(event["interaction_id"], "i9")
        self.assertNotIn("private", self.path.read_text())
        self.assertNotIn("secret answer", self.path.read_text())

    def test_stream_logs_timeout(self):
        with mock.patch("requests.post", side_effect=requests.Timeout("read timed out")):
            with self.assertRaises(OllamaUnavailable):
                list(self.client.chat_stream([], ChatStats()))
        (event,) = load_events(self.path)
        self.assertEqual(event["status"], "timeout")

    def test_stream_logs_success(self):
        lines = [json.dumps({"message": {"content": "a"}}).encode(),
                 json.dumps({"message": {"content": "b"}, "done": True, "eval_count": 2}).encode()]
        with mock.patch("requests.post", return_value=FakeResponse(lines=lines)):
            self.assertEqual("".join(self.client.chat_stream([], ChatStats())), "ab")
        (event,) = load_events(self.path)
        self.assertEqual((event["status"], event["output_tokens"]), ("ok", 2))


if __name__ == "__main__":
    unittest.main()
