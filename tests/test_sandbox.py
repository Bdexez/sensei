"""T2: sandboxed execution — results, limits, cleanup and isolation."""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from sensai.agent.sandbox import Sandbox, SandboxError
from sensai.monitoring import load_events
from sensai.observability import EventLogger


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class SandboxTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "e.jsonl"
        self.sandbox = Sandbox(timeout=3, max_output=2000, logger=EventLogger(str(self.log)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_captures_stdout_and_exit_code(self):
        r = self.sandbox.run("python", "print('konnichiwa')")
        self.assertEqual((r.outcome, r.exit_code, r.stdout.strip()), ("ok", 0, "konnichiwa"))
        self.assertFalse(r.timed_out)

    def test_error_returns_traceback(self):
        r = self.sandbox.run("python", "def f(:\n  pass")
        self.assertEqual(r.outcome, "error")
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("SyntaxError", r.stderr)
        r = self.sandbox.run("python", "import sys\nprint('x')\nsys.exit(3)")
        self.assertEqual((r.outcome, r.exit_code), ("error", 3))

    def test_host_paths_are_hidden(self):
        r = self.sandbox.run("python", "import os\nprint(os.getcwd())\nraise ValueError('boom')")
        self.assertIn("<sandbox>", r.stdout)
        self.assertIn("ValueError: boom", r.stderr)
        self.assertNotIn("sensai-sbx-", r.stdout + r.stderr)

    def test_infinite_loop_is_killed(self):
        start = time.monotonic()
        r = self.sandbox.run("python", "while True:\n    pass")
        self.assertLess(time.monotonic() - start, 6)
        self.assertEqual(r.outcome, "timeout")
        self.assertTrue(r.timed_out)
        self.assertIsNone(r.exit_code)
        self.assertIn("délai", r.stderr)

    def test_blocked_on_sleep_is_killed(self):
        r = self.sandbox.run("python", "import time\ntime.sleep(60)")
        self.assertEqual(r.outcome, "timeout")

    def test_output_limit(self):
        start = time.monotonic()
        r = self.sandbox.run("python", "while True:\n    print('x' * 100)")
        self.assertLess(time.monotonic() - start, 6)
        self.assertEqual(r.outcome, "output_limit")
        self.assertTrue(r.truncated)
        self.assertLessEqual(len(r.stdout), 2000)

    def test_orphan_processes_are_killed(self):
        code = (
            "import subprocess, sys\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print(p.pid)"
        )
        r = self.sandbox.run("python", code)
        pid = int(r.stdout.strip())
        time.sleep(0.3)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_environment_is_not_inherited(self):
        os.environ["SENSAI_TEST_SECRET"] = "s3cret"
        try:
            r = self.sandbox.run("python", "import os\nprint(sorted(os.environ))\nprint(os.environ.get('SENSAI_TEST_SECRET'))")
        finally:
            del os.environ["SENSAI_TEST_SECRET"]
        self.assertNotIn("s3cret", r.stdout)
        self.assertNotIn("OLLAMA", r.stdout)

    def test_workdir_is_fresh_and_deleted(self):
        r1 = self.sandbox.run("python", "open('trace.txt', 'w').write('x')\nimport os\nprint(os.getcwd()[::-1])")
        workdir = r1.stdout.strip()[::-1]
        self.assertFalse(Path(workdir).exists())
        r2 = self.sandbox.run("python", "import os\nprint(os.path.exists('trace.txt'))")
        self.assertEqual(r2.stdout.strip(), "False")

    def test_extra_files_and_unittest_runtime(self):
        files = {
            "conjugaison.py": "def masu(verb):\n    return verb[:-1] + 'imasu'\n",
            "test_conjugaison.py": (
                "import unittest\nfrom conjugaison import masu\n"
                "class T(unittest.TestCase):\n"
                "    def test_kaku(self):\n        self.assertEqual(masu('kaku'), 'kakimasu')\n"
                "    def test_taberu(self):\n        self.assertEqual(masu('taberu'), 'tabemasu')\n"
            ),
        }
        r = self.sandbox.run("python-unittest", files=files)
        self.assertEqual(r.outcome, "error")
        self.assertIn("FAILED (failures=1)", r.stderr)
        files["conjugaison.py"] = (
            "def masu(verb):\n    if verb.endswith('ru'):\n        return verb[:-2] + 'masu'\n"
            "    return verb[:-1] + 'imasu'\n"
        )
        self.assertEqual(self.sandbox.run("python-unittest", files=files).outcome, "ok")

    def test_refusals(self):
        with self.assertRaises(SandboxError):
            self.sandbox.run("bash", "rm -rf /")
        with self.assertRaises(SandboxError):
            self.sandbox.run("python", "x", files={"../evil.py": "x"})
        with self.assertRaises(SandboxError):
            self.sandbox.run("python-unittest", "print(1)")
        with self.assertRaises(SandboxError):
            self.sandbox.run("python")
        restricted = Sandbox(runtimes=["python-unittest"])
        with self.assertRaises(SandboxError):
            restricted.run("python", "print(1)")
        with self.assertRaises(ValueError):
            Sandbox(runtimes=["ruby"])
        with self.assertRaises(ValueError):
            Sandbox(backend="vm")

    def test_runs_are_logged_without_output_content(self):
        self.sandbox.run("python", "print('mot de passe: abc')")
        with self.assertRaises(SandboxError):
            self.sandbox.run("bash", "ls")
        ok, refused = load_events(self.log)
        self.assertEqual((ok["event"], ok["outcome"], ok["exit_code"]), ("sandbox_run", "ok", 0))
        self.assertIn("duration_ms", ok)
        self.assertEqual(refused["outcome"], "refused")
        self.assertNotIn("abc", self.log.read_text())

    def test_docker_command_is_locked_down(self):
        cmd = Sandbox(backend="docker", memory_mb=128).docker_command("n", "/w", ["python", "main.py"])
        joined = " ".join(cmd)
        for flag in ("--network none", "--read-only", "--cap-drop ALL", "--memory 128m", "--pids-limit 64",
                     "/w:/sandbox:ro", "no-new-privileges"):
            self.assertIn(flag, joined)

    @unittest.skipUnless(os.environ.get("SENSAI_DOCKER_TESTS") and docker_available(), "Docker daemon not available")
    def test_docker_backend(self):
        sandbox = Sandbox(backend="docker", timeout=30)
        r = sandbox.run("python", "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), 2)\n"
                                  "    print('net')\nexcept OSError:\n    print('no-net')")
        self.assertEqual(r.stdout.strip(), "no-net")


if __name__ == "__main__":
    unittest.main()
