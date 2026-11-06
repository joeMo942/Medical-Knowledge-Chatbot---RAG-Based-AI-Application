"""Serialises LLM calls (Ollama runs one at a time) and fails fast instead of
queueing a request that cannot finish within its time budget."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from .llm import LLMClient, LLMResult


class Busy(RuntimeError):
    """The LLM queue is too long for this request to finish within its budget."""


class LLMGate:
    def __init__(self, llm: LLMClient, concurrency: int = 1, est_s: float = 5.0):
        self._llm = llm
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._concurrency = max(1, concurrency)
        self._est = est_s  # running estimate of one call's duration, seconds
        self._occupied = 0  # calls running or waiting

    @property
    def model(self) -> str:
        return self._llm.model

    def __getattr__(self, name: str) -> Any:  # warm(), ready(), ... of the wrapped client
        return getattr(self._llm, name)

    async def complete_json(self, system: str, user: str, schema: dict[str, Any],
                            deadline: float | None = None) -> LLMResult:
        if deadline is not None:
            waves_ahead = self._occupied // self._concurrency
            expected = (waves_ahead + 1) * self._est
            if expected > deadline - time.monotonic():
                raise Busy(f"{self._occupied} requests ahead; about {expected:.0f}s needed")
        self._occupied += 1
        try:
            async with self._sem:
                t0 = time.monotonic()
                result = await self._llm.complete_json(system, user, schema)
                self._est = 0.5 * self._est + 0.5 * (time.monotonic() - t0)
                return result
        finally:
            self._occupied -= 1
