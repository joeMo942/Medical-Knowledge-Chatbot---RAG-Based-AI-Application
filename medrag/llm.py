"""LLM clients behind one small interface: local Ollama (default) or any
OpenAI-compatible endpoint (off unless configured through env vars)."""
from __future__ import annotations

import json
from typing import Any, Protocol

import httpx

from .config import Settings


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLMClient(Protocol):
    model: str

    async def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """One chat completion constrained to `schema`; returns the parsed JSON object."""
        ...


def _parse_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model did not return valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMError("model JSON is not an object")
    return obj


class OllamaClient:
    def __init__(self, base_url: str, model: str, num_ctx: int = 8192, timeout: float = 120.0):
        self.model = model
        self._num_ctx = num_ctx
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._http.post("/api/chat", json={
                "model": self.model,
                "stream": False,
                "think": False,  # no reasoning tokens: keeps latency low
                "format": schema,
                "options": {"temperature": 0, "num_ctx": self._num_ctx},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            })
            r.raise_for_status()
            return _parse_object(r.json()["message"]["content"])
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"ollama call failed: {type(exc).__name__}: {exc}") from exc

    async def warm(self) -> None:
        """Load the model into memory so the first real question is not slow."""
        try:
            await self._http.post("/api/generate", json={"model": self.model, "keep_alive": "24h"})
        except Exception:
            pass  # best effort


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 60.0):
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    async def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._http.post("/chat/completions", json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "answer", "schema": schema, "strict": True}},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            })
            r.raise_for_status()
            return _parse_object(r.json()["choices"][0]["message"]["content"])
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"openai-compatible call failed: {type(exc).__name__}: {exc}") from exc


def build_llm(s: Settings) -> LLMClient:
    if s.backend == "openai":
        if not (s.openai_base_url and s.openai_model):
            raise ValueError("MEDRAG_LLM_BACKEND=openai needs MEDRAG_OPENAI_BASE_URL and MEDRAG_OPENAI_MODEL")
        return OpenAICompatClient(s.openai_base_url, s.openai_model, s.openai_api_key)
    if s.backend == "ollama":
        return OllamaClient(s.ollama_url, s.ollama_model, s.ollama_num_ctx)
    raise ValueError(f"unknown MEDRAG_LLM_BACKEND {s.backend!r} (use 'ollama' or 'openai')")
