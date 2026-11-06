"""Serialises LLM calls (Ollama runs one at a time) and fails fast instead of
queueing a request that cannot finish within its time budget."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from .llm import LLMClient, LLMResult

RECENT_CALLS = 5  # the estimate is the slowest of the last few calls


class Busy(RuntimeError):
    """The LLM queue is too long for this request to finish within its budget."""


class LLMGate:
    def __init__(self, llm: LLMClient, concurrency: int = 1, est_s: float = 5.0):
        self._llm = llm
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._concurrency = max(1, concurrency)
        self._floor = est_s  # the estimate never drops below this
        self._recent: deque[float] = deque(maxlen=RECENT_CALLS)  # durations of finished calls
        self._occupied = 0  # calls running or waiting

    @property
    def model(self) -> str:
        return self._llm.model

    @property
    def estimate(self) -> float:
        """Expected duration of one call: the slowest recent call, never below the
        configured floor. Pessimistic on purpose: a quick out_of_scope reply must
        not make the gate admit a queue that a normal answer cannot clear."""
        return max([self._floor, *self._recent])

    def __getattr__(self, name: str) -> Any:  # warm(), ready(), ... of the wrapped client
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._llm, name)

    async def complete_json(self, system: str, user: str, schema: dict[str, Any],
                            deadline: float | None = None) -> LLMResult:
        if deadline is not None:
            waves_ahead = self._occupied // self._concurrency
            expected = (waves_ahead + 1) * self.estimate
            if expected > deadline - time.monotonic():
                raise Busy(f"{self._occupied} requests ahead; about {expected:.0f}s needed")
        self._occupied += 1
        try:
            async with self._sem:
                # The wait may have been longer than estimated: do not start a call
                # (and hold the GPU) that cannot finish before the deadline.
                if deadline is not None and deadline - time.monotonic() < self.estimate:
                    raise Busy(f"only {deadline - time.monotonic():.1f}s left after queueing")
                t0 = time.monotonic()
                try:
                    result = await self._llm.complete_json(system, user, schema)
                except Exception:
                    # A finished call, even a failed or truncated one, held the GPU this
                    # long. A cancelled call (timeout, client gone) says nothing: skipped.
                    self._recent.append(time.monotonic() - t0)
                    raise
                self._recent.append(time.monotonic() - t0)
                return result
        finally:
            self._occupied -= 1
