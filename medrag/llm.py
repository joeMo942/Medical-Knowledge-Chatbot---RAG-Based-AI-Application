"""LLM clients behind one small interface: local Ollama (default) or any
OpenAI-compatible endpoint (off unless configured through env vars)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import Settings

KEEP_ALIVE = "24h"  # same value for warm-up and chat, so Ollama never reloads the model
READY_TIMEOUT_S = 2.0  # /health must never be slow
DEFAULT_MAX_OUTPUT_TOKENS = 1200  # hard cap on one reply; the gate passes a smaller one per call


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLMTruncated(LLMError):
    """The reply hit the output-token cap, so it is incomplete (and usually not valid JSON).
    The pipeline answers with the fixed not-found text instead of an error."""

    def __init__(self, completion_tokens: int | None, cap: int, prompt_tokens: int | None = None):
        super().__init__(f"reply cut off at {completion_tokens} tokens (cap {cap})")
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.cap = cap


@dataclass(frozen=True)
class LLMResult:
    """Parsed JSON plus what the backend reported about the call (for #9; not logged yet).
    Durations are in seconds; None where the backend does not report them."""

    data: dict[str, Any]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_s: float | None = None
    prompt_eval_s: float | None = None
    eval_s: float | None = None


class LLMClient(Protocol):
    model: str

    async def complete_json(self, system: str, user: str, schema: dict[str, Any],
                            max_tokens: int | None = None) -> LLMResult:
        """One chat completion constrained to `schema`, at most `max_tokens` output
        tokens (the client's own maximum when None). Raises LLMTruncated at the cap."""
        ...


def _parse_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model did not return valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMError("model JSON is not an object")
    return obj


def _secs(ns: Any) -> float | None:
    return ns / 1e9 if isinstance(ns, (int, float)) else None


class OllamaClient:
    def __init__(self, base_url: str, model: str, num_ctx: int = 8192, timeout: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS):
        self.model = model
        self.timeout = timeout
        self._num_ctx = num_ctx
        self._max_out = max_output_tokens
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def complete_json(self, system: str, user: str, schema: dict[str, Any],
                            max_tokens: int | None = None) -> LLMResult:
        cap = min(max_tokens or self._max_out, self._max_out)
        try:
            r = await self._http.post("/api/chat", json={
                "model": self.model,
                "stream": False,
                "think": False,  # no reasoning tokens: keeps latency low
                "format": schema,
                "keep_alive": KEEP_ALIVE,
                "options": {"temperature": 0, "num_ctx": self._num_ctx, "num_predict": cap},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            })
            r.raise_for_status()
            body = r.json()
            if body.get("done_reason") == "length":
                raise LLMTruncated(body.get("eval_count"), cap, body.get("prompt_eval_count"))
            return LLMResult(
                data=_parse_object(body["message"]["content"]),
                prompt_tokens=body.get("prompt_eval_count"),
                completion_tokens=body.get("eval_count"),
                total_s=_secs(body.get("total_duration")),
                prompt_eval_s=_secs(body.get("prompt_eval_duration")),
                eval_s=_secs(body.get("eval_duration")),
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"ollama call failed: {type(exc).__name__}: {exc}") from exc

    async def warm(self) -> None:
        """Load the model with the same context size as chat, so the first real
        question does not trigger a reload."""
        try:
            await self._http.post("/api/generate", json={
                "model": self.model, "keep_alive": KEEP_ALIVE, "options": {"num_ctx": self._num_ctx}})
        except Exception:
            pass  # best effort

    async def ready(self) -> bool:
        """True when Ollama answers and the configured model is present."""
        try:
            r = await self._http.get("/api/tags", timeout=READY_TIMEOUT_S)
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
            want = self.model if ":" in self.model else self.model + ":latest"
            return want in names
        except Exception:
            return False


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS):
        self.model = model
        self.timeout = timeout
        self._max_out = max_output_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers,
                                       timeout=timeout, transport=transport)

    async def complete_json(self, system: str, user: str, schema: dict[str, Any],
                            max_tokens: int | None = None) -> LLMResult:
        cap = min(max_tokens or self._max_out, self._max_out)
        t0 = time.monotonic()
        try:
            r = await self._http.post("/chat/completions", json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": cap,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "answer", "schema": schema, "strict": True}},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            })
            r.raise_for_status()
            body = r.json()
            if body["choices"][0].get("finish_reason") == "length":
                usage = body.get("usage") or {}
                raise LLMTruncated(usage.get("completion_tokens"), cap, usage.get("prompt_tokens"))
            usage = body.get("usage") or {}
            return LLMResult(
                data=_parse_object(body["choices"][0]["message"]["content"]),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_s=time.monotonic() - t0,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"openai-compatible call failed: {type(exc).__name__}: {exc}") from exc

    async def ready(self) -> bool:
        """True when the endpoint answers GET /models with a success status."""
        try:
            r = await self._http.get("/models", timeout=READY_TIMEOUT_S)
            return r.status_code < 400
        except Exception:
            return False


def build_llm(s: Settings) -> LLMClient:
    if s.backend == "openai":
        if not (s.openai_base_url and s.openai_model):
            raise ValueError("MEDRAG_LLM_BACKEND=openai needs MEDRAG_OPENAI_BASE_URL and MEDRAG_OPENAI_MODEL")
        return OpenAICompatClient(s.openai_base_url, s.openai_model, s.openai_api_key,
                                  max_output_tokens=s.max_output_tokens)
    if s.backend == "ollama":
        return OllamaClient(s.ollama_url, s.ollama_model, s.ollama_num_ctx,
                            max_output_tokens=s.max_output_tokens)
    raise ValueError(f"unknown MEDRAG_LLM_BACKEND {s.backend!r} (use 'ollama' or 'openai')")
