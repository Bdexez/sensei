"""File access within permissions (T4).

FileGuard is the only way the agent touches the filesystem: every agent tool
(read, list, write, delete, and the sandbox when it imports a workspace file)
goes through it. Each operation is checked *before* anything is opened:

1. the path is normalized and must stay inside an authorized root
   (`..`, absolute paths elsewhere and NUL bytes are refused);
2. symbolic links are refused, or, if allowed, must resolve inside an
   authorized root (the mode of the *target* root applies);
3. hidden files and directories (.git, .env, .ssh...) are refused;
4. the extension must not be denied and, if an allow-list is set, be in it;
5. writes need a root in read-write mode;
6. sizes are capped, for reads and for writes.

Every decision, allowed or refused, is logged as a `file_access` event.
Files are opened with O_NOFOLLOW so a symlink swapped in after the check
is not followed.
"""

import difflib
import os
from dataclasses import dataclass
from pathlib import Path

from ..observability import EventLogger, NullLogger

MODES = ("ro", "rw")
WRITE_OPS = ("write", "create", "delete")


class PermissionDenied(Exception):
    def __init__(self, op: str, path: str, reason: str):
        super().__init__(f"{op} {path!r} refusé : {reason}")
        self.op = op
        self.path = path
        self.reason = reason


class FileOperationError(Exception):
    """The operation was allowed but failed (missing file, binary content...)."""

    def __init__(self, op: str, path: str, reason: str):
        super().__init__(f"{op} {path!r} impossible : {reason}")
        self.op = op
        self.path = path
        self.reason = reason


@dataclass(frozen=True)
class Root:
    name: str  # as configured, shown to the model
    path: Path  # absolute, symlinks resolved
    mode: str


@dataclass
class Entry:
    name: str
    is_dir: bool
    size: int


class FileGuard:
    def __init__(
        self,
        roots: dict[str, str],
        allowed_ext: list[str] | None = None,
        denied_ext: list[str] | None = None,
        max_bytes: int = 100_000,
        allow_symlinks: bool = False,
        base_dir: str | Path | None = None,
        logger: EventLogger | None = None,
        create_roots: bool = True,
    ):
        self.base = Path(base_dir or Path.cwd()).resolve()
        self.allowed_ext = {e.lower() for e in allowed_ext or []}
        self.denied_ext = {e.lower() for e in denied_ext or []}
        self.max_bytes = max_bytes
        self.allow_symlinks = allow_symlinks
        self.logger = logger or NullLogger()
        self.roots: list[Root] = []
        for name, mode in roots.items():
            if mode not in MODES:
                raise ValueError(f"root {name!r}: mode must be one of {MODES}")
            path = Path(name) if Path(name).is_absolute() else self.base / name
            if create_roots and mode == "rw":
                path.mkdir(parents=True, exist_ok=True)
            self.roots.append(Root(name, path.resolve(), mode))

    # --- public operations ------------------------------------------------

    def read_text(self, path: str) -> str:
        target = self.check("read", path)
        size = self._size(target, "read", path)
        if size > self.max_bytes:
            self._deny("read", path, f"fichier trop gros ({size} o > {self.max_bytes} o)")
        data = self._open_read(target, "read", path)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            self._fail("read", path, "fichier binaire ou non UTF-8")
        self._allow("read", path, bytes=len(data))
        return text

    def list_dir(self, path: str = "") -> list[Entry]:
        if not path:
            # The virtual top level: the roots themselves.
            self._allow("list", "", entries=len(self.roots))
            return [Entry(r.name, True, 0) for r in self.roots]
        target = self.check("list", path, is_dir=True)
        if not target.is_dir():
            self._fail("list", path, "n'est pas un répertoire")
        entries = []
        for child in sorted(target.iterdir()):
            if child.name.startswith(".") or (child.is_symlink() and not self.allow_symlinks):
                continue  # never even reveal hidden files or refused links
            entries.append(Entry(child.name, child.is_dir(), child.stat().st_size if child.is_file() else 0))
        self._allow("list", path, entries=len(entries))
        return entries

    def write_text(self, path: str, content: str, overwrite: bool = True) -> int:
        data = content.encode("utf-8")
        exists = self.exists(path)
        op = "write" if exists else "create"
        target = self.check(op, path)
        if len(data) > self.max_bytes:
            self._deny(op, path, f"contenu trop gros ({len(data)} o > {self.max_bytes} o)")
        if exists and not overwrite:
            self._fail(op, path, "le fichier existe déjà")
        if not target.parent.is_dir():
            self._fail(op, path, "le répertoire parent n'existe pas")
        tmp = target.with_name(f".{target.name}.sensai-tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o644)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, target)  # atomic: a reader never sees half a file
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            self._fail(op, path, f"erreur d'écriture ({exc.strerror})")
        self._allow(op, path, bytes=len(data))
        return len(data)

    def delete(self, path: str) -> None:
        target = self.check("delete", path)
        if not target.is_file():
            self._fail("delete", path, "fichier introuvable (seuls les fichiers peuvent être supprimés)")
        target.unlink()
        self._allow("delete", path)

    def exists(self, path: str) -> bool:
        """Silent existence test (not logged); False for paths outside the roots."""
        try:
            target = self._resolve(path)[0]
        except PermissionDenied:
            return False
        return target.exists()

    def diff_preview(self, path: str, new_content: str, max_lines: int = 40) -> str:
        """Unified diff between the current file (if any) and `new_content`, for confirmations."""
        old = self.read_text(path) if self.exists(path) else ""
        lines = list(
            difflib.unified_diff(
                old.splitlines(), new_content.splitlines(), f"a/{path}", f"b/{path}", lineterm="", n=2
            )
        )
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"… ({len(lines) - max_lines} lignes de diff en plus)"]
        return "\n".join(lines) or "(aucun changement)"

    def describe(self) -> str:
        """Human/model-readable summary of what is allowed."""
        roots = ", ".join(f"{r.name} ({'lecture-écriture' if r.mode == 'rw' else 'lecture seule'})" for r in self.roots)
        ext = ", ".join(sorted(self.allowed_ext)) or "toutes sauf interdites"
        return f"Répertoires autorisés : {roots}. Extensions : {ext}. Taille max : {self.max_bytes} octets."

    # --- checks -------------------------------------------------------------

    def check(self, op: str, path: str, is_dir: bool = False) -> Path:
        """Return the absolute path if `op` is allowed on `path`, else log and raise PermissionDenied."""
        try:
            target, root = self._resolve(path, op)
        except PermissionDenied as exc:
            self._deny(op, exc.path, exc.reason)
        if op in WRITE_OPS and root.mode != "rw":
            self._deny(op, path, f"« {root.name} » est en lecture seule")
        if not is_dir and target != root.path:
            name = target.name.lower()
            ext = Path(name).suffix or (name if name.startswith(".") else "")
            if ext in self.denied_ext or any(name.endswith(e) for e in self.denied_ext):
                self._deny(op, path, f"extension interdite ({ext or name})")
            if self.allowed_ext and ext not in self.allowed_ext:
                self._deny(op, path, f"extension non autorisée ({ext or 'aucune'})")
        return target

    def _resolve(self, path: str, op: str = "check") -> tuple[Path, Root]:
        """Locate `path` inside a root; raises PermissionDenied without logging (callers log)."""
        if not isinstance(path, str) or not path.strip():
            raise PermissionDenied(op, str(path), "chemin vide")
        if "\x00" in path:
            raise PermissionDenied(op, path, "caractère NUL dans le chemin")
        raw = Path(os.path.expanduser(path)) if path.startswith("~") else Path(path)
        candidate = Path(os.path.normpath(raw if raw.is_absolute() else self.base / raw))
        root = self._root_of(candidate)
        if root is None:
            raise PermissionDenied(op, path, "hors des répertoires autorisés")
        rel = candidate.relative_to(self._lexical_root(root))
        if any(part.startswith(".") for part in rel.parts):
            raise PermissionDenied(op, path, "fichier ou répertoire caché")

        # Symlinks: walk every existing component below the root.
        current = self._lexical_root(root)
        for part in rel.parts:
            current = current / part
            if current.is_symlink():
                if not self.allow_symlinks:
                    raise PermissionDenied(op, path, "lien symbolique refusé")
                real = current.resolve()
                target_root = self._root_of(real, resolved=True)
                if target_root is None:
                    raise PermissionDenied(op, path, "lien symbolique vers l'extérieur des répertoires autorisés")
        real = candidate.resolve()
        real_root = self._root_of(real, resolved=True)
        if real_root is None:
            raise PermissionDenied(op, path, "hors des répertoires autorisés après résolution")
        return real, real_root

    def _lexical_root(self, root: Root) -> Path:
        configured = Path(root.name)
        return Path(os.path.normpath(configured if configured.is_absolute() else self.base / configured))

    def _root_of(self, candidate: Path, resolved: bool = False) -> Root | None:
        # Longest match first, so a nested read-only root wins over its parent.
        for root in sorted(self.roots, key=lambda r: len(r.path.parts), reverse=True):
            base = root.path if resolved else self._lexical_root(root)
            if candidate == base or candidate.is_relative_to(base):
                return root
        return None

    # --- helpers ------------------------------------------------------------

    def _size(self, target: Path, op: str, path: str) -> int:
        try:
            if not target.is_file():
                self._fail(op, path, "fichier introuvable")
            return target.stat().st_size
        except OSError as exc:
            self._fail(op, path, f"inaccessible ({exc.strerror})")

    def _open_read(self, target: Path, op: str, path: str) -> bytes:
        try:
            fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as f:
                return f.read(self.max_bytes + 1)[: self.max_bytes]
        except OSError as exc:
            self._fail(op, path, f"lecture impossible ({exc.strerror})")

    def _allow(self, op: str, path: str, **fields) -> None:
        self.logger.log("file_access", op=op, path=path, allowed=True, **fields)

    def _fail(self, op: str, path: str, reason: str):
        self.logger.log("file_access", op=op, path=path, allowed=True, status="error", reason=reason)
        raise FileOperationError(op, path, reason)

    def _deny(self, op: str, path: str, reason: str):
        self.logger.log("file_access", op=op, path=path, allowed=False, status="denied", reason=reason)
        raise PermissionDenied(op, path, reason)
