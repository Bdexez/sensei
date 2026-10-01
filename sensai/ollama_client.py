"""Minimal streaming client for the Ollama HTTP API (/api/chat)."""

import json
from collections.abc import Iterator
from dataclasses import dataclass

import requests


class OllamaError(Exception):
    pass


class OllamaUnavailable(OllamaError):
    pass


class ModelNotFound(OllamaError):
    pass


@dataclass
class ChatStats:
    """Token counts reported by Ollama in the final chunk of a response."""

    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration_ms: float = 0.0


class OllamaClient:
    def __init__(self, host: str, model: str, num_ctx: int, think: bool | None = None, timeout: float = 300):
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout

    def check(self) -> None:
        """Fail early if the server is down or the model is not pulled."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"Ollama unreachable at {self.host} (is `ollama serve` running?)") from exc
        names = {m["name"] for m in resp.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise ModelNotFound(f"model '{self.model}' not found; pull it with `ollama pull {self.model}`")

    def chat_stream(self, messages: list[dict], stats: ChatStats) -> Iterator[str]:
        """Yield response text pieces as they arrive; fill `stats` when done."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": self.num_ctx},
        }
        if self.think is not None:
            payload["think"] = self.think
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, stream=True, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"connection to Ollama failed: {exc}") from exc

        with resp:
            if resp.status_code != 200:
                raise self._error_from(resp)
            try:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(chunk["error"])
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        stats.prompt_eval_count = chunk.get("prompt_eval_count", 0)
                        stats.eval_count = chunk.get("eval_count", 0)
                        stats.total_duration_ms = chunk.get("total_duration", 0) / 1e6
            except requests.RequestException as exc:
                raise OllamaUnavailable(f"connection lost during generation: {exc}") from exc

    def _error_from(self, resp: requests.Response) -> OllamaError:
        try:
            message = resp.json().get("error", resp.text)
        except ValueError:
            message = resp.text
        if resp.status_code == 404:
            return ModelNotFound(message)
        return OllamaError(f"HTTP {resp.status_code}: {message}")
