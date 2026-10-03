# Sensai

A local language tutor in your terminal, built on a model served by [Ollama](https://ollama.com) and modeled on Duolingo. It runs short exercises, corrects your mistakes and remembers your progress. In **agent mode** it can also prepare your study material: it reads your lessons, writes revision sheets and Python quizzes, runs them in a sandbox to check them, and always asks for your confirmation before changing anything.

No LLM framework is used: every model call goes straight to Ollama's HTTP API (`/api/chat`).

## Features

| ID | Feature | Where |
|---|---|---|
| B1 | CLI chatbot with streamed answers, conversation history and error handling | `sensai/cli.py`, `sensai/ollama_client.py` |
| M1 | Persistent sessions and learner profile (SQLite) | `sensai/memory/store.py` |
| M2 | Token budget and compression of old turns | `sensai/context.py`, `sensai/memory/compressor.py` |
| M3 | Structured learner memory: vocabulary with spaced repetition, recurring mistakes | `sensai/memory/learner.py`, `sensai/memory/extractor.py` |
| **A1** | **ReAct reasoning loop**: analyse → action → observation → adjustment | `sensai/agent/react.py`, `sensai/agent/tools.py` |
| **T2** | **Sandboxed code execution**: timeout, output cap, process isolation | `sensai/agent/sandbox.py` |
| **T4** | **File access with permissions**: authorised folders, read-only/read-write, symlinks, extensions, sizes | `sensai/agent/permissions.py` |
| **A3** | **Human-in-the-loop**: confirmation before any write, deletion or execution | `sensai/agent/approval.py` |
| **EV3** | **Structured logs, secret masking and metrics** | `sensai/observability.py`, `sensai/monitoring.py` |

Detailed documentation:
- [docs/memoire_contexte.md](docs/memoire_contexte.md): memory and context (M1, M2, M3);
- [docs/agent_securite_observabilite.md](docs/agent_securite_observabilite.md): agent, sandbox, permissions, confirmations and monitoring (A1, T2, T4, A3, EV3), with architecture, design choices, security limits, **user stories** and the keynote demo.

## Installation

Requirements: **Python ≥ 3.10** and a running Ollama server.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull qwen3.8:4b        # or any other model; check with `ollama list`
```

## Usage

```bash
.venv/bin/python -m sensai                          # model and settings from config.json
.venv/bin/python -m sensai --model ministral-3:14b  # pick a different model without touching the code
.venv/bin/python -m sensai --user camille --db data/camille.db
```

Chat commands (full list with `/help`):

| Command | Purpose |
|---|---|
| `/profile set langue_cible japonais` | fill in the learner profile |
| `/vocab`, `/mistakes`, `/summary`, `/context` | learner memory and context |
| `/agent <task>` | agent mode (A1): tools, sandbox, confirmations |
| `/trace` | steps of the last agent task (public summaries, actions, results) |
| `/tools` | agent tools, allowed folders, sandbox limits |
| `/metrics` | metrics: latency, tokens, errors, tools, confirmations, sandbox |

Example agent session:

```text
> /agent Write a Python quiz from docs/lecons/animaux.md, test it, then save it to workspace/quiz.py
[step 1] I read the lesson to get the vocabulary
           → read_file(path=docs/lecons/animaux.md) ✓ (1645 ms)
┌────────────────────────────────────────────────────────────
│ ⚠  Confirmation required
│ Action     : run_python — Run a Python program in the sandbox
│ ...
Confirm? [y/N] y
[step 2] I test the quiz
           → run_python(code=import random⏎…) ✓ (11839 ms)
           sandbox : error, code 1 — EOFError: EOF when reading a line
[step 3] I replace input() with a simulation mode so it can be tested
...
```

The metrics are also available outside the chat:

```bash
.venv/bin/python -m sensai.monitoring            # dashboard
.venv/bin/python -m sensai.monitoring --json     # raw metrics
tail -f data/logs/sensai.jsonl                   # live events (secrets masked)
```

## Configuration

Everything lives in `config.json` (CLI arguments take precedence over it). Main keys for agent mode:

| Key | Default | Meaning |
|---|---|---|
| `agent_max_steps` | 8 | maximum number of ReAct iterations |
| `file_roots` | `{"workspace": "rw", "docs": "ro"}` | folders the agent may access, and how |
| `file_allowed_ext` / `file_denied_ext` | `.md .txt .csv .json .py` / `.env .pem .key .db…` | allowed and denied extensions |
| `file_max_bytes` | 100000 | maximum size for a read or a write |
| `file_allow_symlinks` | false | symbolic links (when true, only towards an authorised folder) |
| `sandbox_backend` | `subprocess` | `docker` for real isolation (needs the daemon) |
| `sandbox_runtimes` | `python`, `python-unittest` | runtimes the sandbox may use |
| `sandbox_timeout` / `sandbox_max_output` | 10 s / 8000 | limits for one execution |
| `confirm_actions` | `write`, `delete`, `execute` | actions that need confirmation (`write` and `delete` always do) |
| `confirm_timeout` | 300 | after this many seconds with no answer, the action is refused |
| `log_path` / `log_content_chars` | `data/logs/sensai.jsonl` / 120 | event log and longest text excerpt it keeps |

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
SENSAI_DOCKER_TESTS=1 .venv/bin/python -m unittest tests.test_sandbox   # + Docker backend, if the daemon is running
```

The tests use no network and no model: the ReAct loop is driven by a scripted fake model, while the permission layer, the sandbox and the confirmations are the real ones. Benchmarks that need Ollama live in `benchmarks/` (M2, M3).

## User stories

User stories for A1, T2, T4, A3 and EV3 (at least two per feature, with acceptance criteria): [docs/agent_securite_observabilite.md](docs/agent_securite_observabilite.md#9-user-stories).
