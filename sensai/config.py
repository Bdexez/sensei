"""Configuration: config.json values, overridable from the command line."""

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass
class Config:
    model: str = "qwen3.8:4b"
    host: str = "http://localhost:11434"
    num_ctx: int = 8192
    response_reserve: float = 0.2
    think: bool | None = False
    db_path: str = "data/sensai.db"
    user: str = "default"
    system_prompt: str = "You are Sensai, a friendly language tutor."
    # M2: compression of old turns
    compression: bool = True
    compress_trigger: float = 0.8
    compress_keep: float = 0.4
    summary_share: float = 0.1
    # M3: structured learner memory
    memory: bool = True
    memory_share: float = 0.1


class ConfigError(Exception):
    pass


def _load_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    return data


def load_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(prog="sensai", description="Local language tutor on Ollama")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", help="Ollama model tag (see `ollama list`)")
    parser.add_argument("--host", help="Ollama base URL")
    parser.add_argument("--num-ctx", type=int, dest="num_ctx", help="context window size in tokens")
    parser.add_argument("--db", dest="db_path", help="SQLite database path")
    parser.add_argument("--user", help="learner profile name")
    args = parser.parse_args(argv)

    values = _load_file(args.config)
    for key in ("model", "host", "num_ctx", "db_path", "user"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    config = Config(**values)
    for key in ("response_reserve", "compress_trigger", "compress_keep", "summary_share", "memory_share"):
        if not 0 < getattr(config, key) < 1:
            raise ConfigError(f"{key} must be between 0 and 1")
    if config.compress_keep >= config.compress_trigger:
        raise ConfigError("compress_keep must be lower than compress_trigger")
    if config.summary_share + config.memory_share > 0.5:
        raise ConfigError("summary_share + memory_share must stay under 0.5 to leave room for history and RAG")
    return config
