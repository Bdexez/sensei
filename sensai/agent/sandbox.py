"""Sandboxed code execution (T2).

The agent runs the code it wrote (or a workspace file it imported through
FileGuard) and gets back stdout, stderr, the exit code and a status it can
react to — typically a traceback it then fixes.

Two backends:

- "subprocess" (default, no dependency): a fresh temporary directory per run,
  only the files explicitly passed are copied in; Python in isolated mode
  (-I: no user site-packages, no PYTHON* variables, cwd not on sys.path);
  an empty environment (no API key or token inherited from the shell);
  POSIX resource limits (CPU time, memory where the OS supports it, file size,
  open files, no core dump); its own process group, killed entirely on
  timeout or when the output limit is exceeded; directory deleted afterwards.
  This is NOT a security boundary against malicious code: the process runs as
  the same user and still has network access and read access to whatever
  that user can read. It protects against mistakes (infinite loops, runaway
  output, stray files), which is the realistic threat for a tutor agent.

- "docker" (opt-in): the same run inside a throwaway container with no
  network, a read-only filesystem, all capabilities dropped, an unprivileged
  user, and memory / CPU / process limits. Real isolation, but needs a
  running Docker daemon and adds ~1 s of start-up per run.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass

from ..observability import EventLogger, NullLogger

# runtime -> (entry file written from `code`, command run in the work dir)
RUNTIMES: dict[str, tuple[str | None, list[str]]] = {
    "python": ("main.py", ["{python}", "-I", "-B", "-X", "utf8", "main.py"]),
    "python-unittest": (None, ["{python}", "-I", "-B", "-X", "utf8", "-m", "unittest", "discover", "-v", "-s", ".", "-p", "test*.py"]),
}
BACKENDS = ("subprocess", "docker")
MAX_FILES = 20


class SandboxError(Exception):
    """Invalid request (unknown runtime, bad file name...): nothing was executed."""


@dataclass
class SandboxResult:
    runtime: str
    outcome: str  # ok | error | timeout | output_limit
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    truncated: bool = False
    backend: str = "subprocess"

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"

    def to_dict(self) -> dict:
        return asdict(self)


class _Capture(threading.Thread):
    """Drain one pipe, keep at most `limit` bytes, signal when the limit is crossed."""

    def __init__(self, pipe, limit: int, on_overflow):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.limit = limit
        self.on_overflow = on_overflow
        self.data = bytearray()
        self.overflow = False

    def run(self) -> None:
        for chunk in iter(lambda: self.pipe.read(4096), b""):
            room = self.limit - len(self.data)
            if room > 0:
                self.data += chunk[:room]
            if len(chunk) > room and not self.overflow:
                self.overflow = True
                self.on_overflow()
        self.pipe.close()


class Sandbox:
    def __init__(
        self,
        runtimes: list[str] | None = None,
        timeout: float = 10,
        max_output: int = 10_000,
        memory_mb: int = 256,
        backend: str = "subprocess",
        docker_image: str = "python:3.12-alpine",
        logger: EventLogger | None = None,
    ):
        runtimes = runtimes if runtimes is not None else ["python", "python-unittest"]
        unknown = set(runtimes) - set(RUNTIMES)
        if unknown:
            raise ValueError(f"unknown sandbox runtimes {sorted(unknown)}; available: {sorted(RUNTIMES)}")
        if backend not in BACKENDS:
            raise ValueError(f"sandbox backend must be one of {BACKENDS}")
        self.runtimes = runtimes
        self.timeout = timeout
        self.max_output = max_output
        self.memory_mb = memory_mb
        self.backend = backend
        self.docker_image = docker_image
        self.logger = logger or NullLogger()

    def describe(self) -> str:
        return (
            f"Runtimes autorisés : {', '.join(self.runtimes)}. Durée max {self.timeout:g} s, "
            f"sortie max {self.max_output} caractères, pas de fichiers persistants, backend {self.backend}."
        )

    def run(
        self,
        runtime: str,
        code: str | None = None,
        files: dict[str, str] | None = None,
        interaction_id: str | None = None,
    ) -> SandboxResult:
        """Execute `code` (written to the runtime's entry file) with `files` copied alongside."""
        if runtime not in self.runtimes:
            self.logger.log("sandbox_run", interaction_id, runtime=runtime, outcome="refused", status="denied",
                            reason="runtime non autorisé")
            raise SandboxError(f"runtime « {runtime} » non autorisé (autorisés : {', '.join(self.runtimes)})")
        entry, template = RUNTIMES[runtime]
        files = dict(files or {})
        if code is not None:
            if entry is None:
                raise SandboxError(f"le runtime « {runtime} » n'exécute que des fichiers, pas de code direct")
            files[entry] = code
        if not files:
            raise SandboxError("rien à exécuter")
        if len(files) > MAX_FILES:
            raise SandboxError(f"trop de fichiers (max {MAX_FILES})")
        for name in files:
            if not name or "/" in name or "\\" in name or name.startswith(".") or "\x00" in name:
                raise SandboxError(f"nom de fichier invalide dans la sandbox : {name!r}")

        workdir = tempfile.mkdtemp(prefix="sensai-sbx-")
        try:
            for name, content in files.items():
                with open(os.path.join(workdir, name), "w", encoding="utf-8") as f:
                    f.write(content)
            result = self._execute(runtime, template, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        # Hide the host path of the temporary directory from the model.
        result.stdout = result.stdout.replace(workdir, "<sandbox>")
        result.stderr = result.stderr.replace(workdir, "<sandbox>")
        self.logger.log(
            "sandbox_run",
            interaction_id,
            runtime=runtime,
            backend=self.backend,
            outcome=result.outcome,
            status="timeout" if result.timed_out else ("ok" if result.ok else "error"),
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            timed_out=result.timed_out,
            truncated=result.truncated,
            stdout_chars=len(result.stdout),
            stderr_chars=len(result.stderr),
            files=sorted(files),
        )
        return result

    # --- execution ----------------------------------------------------------

    def _execute(self, runtime: str, template: list[str], workdir: str) -> SandboxResult:
        if self.backend == "docker":
            name = f"sensai-sbx-{uuid.uuid4().hex[:10]}"
            inner = [part.replace("{python}", "python") for part in template]
            cmd = self.docker_command(name, workdir, inner)
            kill = lambda proc: subprocess.run(["docker", "kill", name], capture_output=True, timeout=10)  # noqa: E731
            popen_kwargs: dict = {}
        else:
            cmd = [part.replace("{python}", sys.executable) for part in template]
            kill = self._kill_group
            popen_kwargs = {"start_new_session": True}
            if os.name == "posix":
                popen_kwargs["preexec_fn"] = self._limits

        env = {
            "PATH": os.defpath,
            "HOME": workdir,
            "TMPDIR": workdir,
            "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        start = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, **popen_kwargs,
            )
        except OSError as exc:
            raise SandboxError(f"impossible de lancer le runtime : {exc}") from exc

        overflow = threading.Event()

        def on_overflow():
            overflow.set()
            kill(proc)

        captures = [_Capture(proc.stdout, self.max_output, on_overflow), _Capture(proc.stderr, self.max_output, on_overflow)]
        for c in captures:
            c.start()
        timed_out = False
        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill(proc)
            proc.wait()
        finally:
            if self.backend == "subprocess":
                self._kill_group(proc)  # orphans left by the program
        for c in captures:
            c.join(timeout=2)
        duration = round((time.perf_counter() - start) * 1000, 1)

        stdout, stderr = (c.data.decode("utf-8", "replace") for c in captures)
        if overflow.is_set():
            outcome = "output_limit"
            stderr += f"\n[sortie tronquée à {self.max_output} caractères, processus arrêté]"
        elif timed_out:
            outcome = "timeout"
            stderr += f"\n[délai de {self.timeout:g} s dépassé, processus arrêté]"
        else:
            outcome = "ok" if proc.returncode == 0 else "error"
        return SandboxResult(
            runtime=runtime,
            outcome=outcome,
            exit_code=None if timed_out else proc.returncode,
            stdout=stdout,
            stderr=stderr.strip("\n") if outcome != "ok" else stderr,
            duration_ms=duration,
            timed_out=timed_out,
            truncated=overflow.is_set(),
            backend=self.backend,
        )

    def docker_command(self, name: str, workdir: str, inner: list[str]) -> list[str]:
        return [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:rw,size=16m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--memory", f"{self.memory_mb}m", "--cpus", "1", "--pids-limit", "64",
            "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{workdir}:/sandbox:ro", "-w", "/sandbox",
            self.docker_image, *inner,
        ]

    def _limits(self) -> None:
        """Runs in the child before exec (POSIX only)."""
        import resource

        cpu = int(self.timeout) + 1
        limits = [
            (resource.RLIMIT_CPU, (cpu, cpu)),
            (resource.RLIMIT_FSIZE, (5 * 1024 * 1024, 5 * 1024 * 1024)),
            (resource.RLIMIT_NOFILE, (64, 64)),
            (resource.RLIMIT_CORE, (0, 0)),
        ]
        if sys.platform.startswith("linux"):
            # macOS does not enforce RLIMIT_AS; Linux does.
            mem = self.memory_mb * 1024 * 1024
            limits.append((resource.RLIMIT_AS, (mem, mem)))
        for kind, value in limits:
            try:
                resource.setrlimit(kind, value)
            except (ValueError, OSError):
                pass

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
