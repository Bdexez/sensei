"""Minimal streaming client for the Ollama HTTP API (/api/chat)."""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import requests

from .observability import EventLogger, NullLogger


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
    def __init__(
        self,
        host: str,
        model: str,
        num_ctx: int,
        think: bool | None = None,
        timeout: float = 300,
        logger: EventLogger | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        self.logger = logger or NullLogger()

    def _log_call(
        self, purpose: str, interaction_id: str | None, start: float, stats: ChatStats, error: Exception | None
    ) -> None:
        """EV3: one `model_call` event per request, success or not (no prompt or answer content)."""
        status = "ok"
        if isinstance(error, requests.Timeout) or "timed out" in str(error or ""):
            status = "timeout"
        elif error is not None:
            status = "error"
        self.logger.log(
            "model_call",
            interaction_id,
            model=self.model,
            purpose=purpose,
            status=status,
            error=f"{type(error).__name__}: {error}" if error else None,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
            prompt_tokens=stats.prompt_eval_count,
            output_tokens=stats.eval_count,
            model_duration_ms=round(stats.total_duration_ms, 1),
        )

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

    def _payload(self, messages: list[dict], stream: bool, options: dict | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"num_ctx": self.num_ctx, **(options or {})},
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def chat(
        self,
        messages: list[dict],
        stats: ChatStats,
        schema: dict | None = None,
        options: dict | None = None,
        purpose: str = "chat",
        interaction_id: str | None = None,
    ) -> str:
        """Non-streaming call; with `schema`, Ollama constrains the output to that JSON schema."""
        start = time.perf_counter()
        error: Exception | None = None
        try:
            return self._chat(messages, stats, schema, options)
        except Exception as exc:
            error = exc
            raise
        finally:
            self._log_call(purpose, interaction_id, start, stats, error)

    def _chat(self, messages: list[dict], stats: ChatStats, schema: dict | None, options: dict | None) -> str:
        payload = self._payload(messages, stream=False, options=options)
        if schema is not None:
            payload["format"] = schema
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"connection to Ollama failed: {exc}") from exc
        if resp.status_code != 200:
            raise self._error_from(resp)
        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError(f"unreadable response from Ollama: {exc}") from exc
        if "error" in data:
            raise OllamaError(data["error"])
        stats.prompt_eval_count = data.get("prompt_eval_count", 0)
        stats.eval_count = data.get("eval_count", 0)
        stats.total_duration_ms = data.get("total_duration", 0) / 1e6
        return data.get("message", {}).get("content", "")

    def chat_stream(
        self, messages: list[dict], stats: ChatStats, purpose: str = "chat", interaction_id: str | None = None
    ) -> Iterator[str]:
        """Yield response text pieces as they arrive; fill `stats` when done."""
        start = time.perf_counter()
        error: Exception | None = None
        try:
            yield from self._chat_stream(messages, stats)
        except (GeneratorExit, KeyboardInterrupt):
            error = InterruptedError("generation interrupted")
            raise
        except Exception as exc:
            error = exc
            raise
        finally:
            self._log_call(purpose, interaction_id, start, stats, error)

    def _chat_stream(self, messages: list[dict], stats: ChatStats) -> Iterator[str]:
        payload = self._payload(messages, stream=True)
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
