"""T4: permission layer — nominal operations, refusals and attack attempts."""

import os
import tempfile
import unittest
from pathlib import Path

from sensai.agent.permissions import FileGuard, PermissionDenied
from sensai.monitoring import load_events
from sensai.observability import EventLogger


class FileGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        (self.base / "docs").mkdir()
        (self.base / "docs" / "cours.md").write_text("# Leçon 1\nneko = chat\n")
        (self.base / "secret").mkdir()
        (self.base / "secret" / "keys.txt").write_text("hunter2")
        self.log = self.base / "events.jsonl"
        self.guard = FileGuard(
            {"workspace": "rw", "docs": "ro"},
            allowed_ext=[".md", ".txt", ".csv", ".py", ".json"],
            denied_ext=[".env", ".pem", ".db"],
            max_bytes=1000,
            base_dir=self.base,
            logger=EventLogger(str(self.log)),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def assertDenied(self, fn, *args, reason=""):
        with self.assertRaises(PermissionDenied) as ctx:
            fn(*args)
        self.assertIn(reason, ctx.exception.reason)
        return ctx.exception

    # --- nominal --------------------------------------------------------

    def test_rw_root_is_created(self):
        self.assertTrue((self.base / "workspace").is_dir())

    def test_read_list_write_delete(self):
        self.assertIn("neko", self.guard.read_text("docs/cours.md"))
        self.assertEqual(self.guard.write_text("workspace/fiches.md", "neko = chat"), 11)
        self.assertEqual(self.guard.read_text("workspace/fiches.md"), "neko = chat")
        self.assertEqual([e.name for e in self.guard.list_dir("workspace")], ["fiches.md"])
        self.assertEqual({e.name for e in self.guard.list_dir("")}, {"workspace", "docs"})
        self.guard.delete("workspace/fiches.md")
        self.assertFalse((self.base / "workspace" / "fiches.md").exists())

    def test_absolute_path_inside_root_is_accepted(self):
        self.assertIn("neko", self.guard.read_text(str(self.base / "docs" / "cours.md")))

    def test_internal_dotdot_that_stays_inside_is_accepted(self):
        self.assertIn("neko", self.guard.read_text("docs/../docs/cours.md"))

    def test_diff_preview(self):
        self.guard.write_text("workspace/a.md", "un\ndeux\n")
        diff = self.guard.diff_preview("workspace/a.md", "un\ntrois\n")
        self.assertIn("-deux", diff)
        self.assertIn("+trois", diff)
        self.assertIn("+nouveau", self.guard.diff_preview("workspace/new.md", "nouveau"))

    def test_no_overwrite(self):
        self.guard.write_text("workspace/a.md", "x")
        self.assertDenied(self.guard.write_text, "workspace/a.md", "y", False, reason="existe déjà")

    # --- refusals -------------------------------------------------------

    def test_read_only_root(self):
        self.assertDenied(self.guard.write_text, "docs/cours.md", "x", reason="lecture seule")
        self.assertDenied(self.guard.delete, "docs/cours.md", reason="lecture seule")
        self.assertIn("neko", (self.base / "docs" / "cours.md").read_text())

    def test_outside_roots(self):
        self.assertDenied(self.guard.read_text, "secret/keys.txt", reason="hors des répertoires")
        self.assertDenied(self.guard.read_text, "/etc/passwd", reason="hors des répertoires")
        self.assertDenied(self.guard.read_text, "~/.ssh/id_rsa", reason="hors des répertoires")

    def test_path_traversal(self):
        for path in ("workspace/../secret/keys.txt", "docs/../../../../etc/passwd", "workspace/../../secret/keys.txt",
                     "../" * 10 + "etc/passwd"):
            with self.subTest(path=path):
                self.assertDenied(self.guard.read_text, path)
        self.assertDenied(self.guard.write_text, "workspace/../evil.md", "x")
        self.assertFalse((self.base / "evil.md").exists())

    def test_empty_and_nul(self):
        self.assertDenied(self.guard.read_text, "", reason="vide")
        self.assertDenied(self.guard.read_text, "docs/cours.md\x00.txt", reason="NUL")

    def test_hidden_files(self):
        (self.base / "workspace" / ".git").mkdir()
        (self.base / "workspace" / ".git" / "config.txt").write_text("x")
        self.assertDenied(self.guard.read_text, "workspace/.git/config.txt", reason="caché")
        self.assertDenied(self.guard.write_text, "workspace/.env", "TOKEN=x")
        self.assertEqual(self.guard.list_dir("workspace"), [])

    def test_extensions(self):
        self.assertDenied(self.guard.write_text, "workspace/run.sh", "rm -rf /", reason="non autorisée")
        self.assertDenied(self.guard.write_text, "workspace/key.pem", "x", reason="interdite")
        self.assertDenied(self.guard.write_text, "workspace/prod.env", "x", reason="interdite")
        self.assertDenied(self.guard.write_text, "workspace/Makefile", "x", reason="non autorisée")

    def test_size_limits(self):
        self.assertDenied(self.guard.write_text, "workspace/big.md", "x" * 1001, reason="trop gros")
        (self.base / "docs" / "big.md").write_text("x" * 2000)
        self.assertDenied(self.guard.read_text, "docs/big.md", reason="trop gros")

    def test_binary_and_missing(self):
        (self.base / "docs" / "bin.txt").write_bytes(b"\xff\xfe\x00")
        self.assertDenied(self.guard.read_text, "docs/bin.txt", reason="binaire")
        self.assertDenied(self.guard.read_text, "docs/absent.md", reason="introuvable")
        self.assertDenied(self.guard.delete, "workspace/absent.md", reason="introuvable")
        self.assertDenied(self.guard.write_text, "workspace/no/dir.md", "x", reason="parent")

    # --- symlinks -------------------------------------------------------

    def test_symlink_escape_is_refused(self):
        os.symlink(self.base / "secret" / "keys.txt", self.base / "workspace" / "link.txt")
        os.symlink(self.base / "secret", self.base / "workspace" / "dir")
        self.assertDenied(self.guard.read_text, "workspace/link.txt", reason="lien symbolique")
        self.assertDenied(self.guard.read_text, "workspace/dir/keys.txt", reason="lien symbolique")
        self.assertDenied(self.guard.write_text, "workspace/link.txt", "pwned", reason="lien symbolique")
        self.assertEqual((self.base / "secret" / "keys.txt").read_text(), "hunter2")
        self.assertEqual(self.guard.list_dir("workspace"), [])

    def test_symlinks_allowed_only_inside_roots(self):
        guard = FileGuard({"workspace": "rw", "docs": "ro"}, allow_symlinks=True, base_dir=self.base)
        os.symlink(self.base / "docs" / "cours.md", self.base / "workspace" / "cours.md")
        os.symlink(self.base / "secret" / "keys.txt", self.base / "workspace" / "out.txt")
        self.assertIn("neko", guard.read_text("workspace/cours.md"))
        # The target lives in a read-only root: writing through the link is refused.
        self.assertDenied(guard.write_text, "workspace/cours.md", "x", reason="lecture seule")
        self.assertDenied(guard.read_text, "workspace/out.txt", reason="extérieur")

    def test_symlinked_root_parent_does_not_leak(self):
        os.symlink(self.base / "secret", self.base / "docs" / "s")
        self.assertDenied(self.guard.read_text, "docs/s/keys.txt")

    # --- logging --------------------------------------------------------

    def test_every_decision_is_logged(self):
        self.guard.read_text("docs/cours.md")
        with self.assertRaises(PermissionDenied):
            self.guard.read_text("../secret/keys.txt")
        allowed, denied = load_events(self.log)
        self.assertEqual((allowed["op"], allowed["allowed"]), ("read", True))
        self.assertEqual((denied["op"], denied["allowed"]), ("read", False))
        self.assertIn("hors", denied["reason"])
        self.assertNotIn("hunter2", self.log.read_text())

    def test_exists_is_silent(self):
        self.assertFalse(self.guard.exists("../secret/keys.txt"))
        self.assertTrue(self.guard.exists("docs/cours.md"))
        self.assertFalse(self.log.exists() and self.log.read_text())

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            FileGuard({"x": "admin"}, base_dir=self.base)


if __name__ == "__main__":
    unittest.main()
