"""Serialises LLM calls (Ollama runs one at a time), fails fast instead of queueing
a request that cannot finish within its time budget, and sizes each call's output
cap to the time that request has left."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from .llm import DEFAULT_MAX_OUTPUT_TOKENS, LLMClient, LLMResult, LLMTruncated

log = logging.getLogger("medrag")

RECENT_CALLS = 5  # the duration estimate is the slowest of the last few calls
RATE_WEIGHT = 0.3  # weight of a new observation in the decode-rate average
MIN_RATE_SAMPLE = (50, 0.5)  # (tokens, seconds): shorter decodes are too noisy to learn from


class Busy(RuntimeError):
    """The LLM queue is too long for this request to finish within its budget."""


class LLMGate:
    def __init__(self, llm: LLMClient, concurrency: int = 1, est_s: float = 5.0,
                 decode_tok_s: float = 53.0, margin_s: float = 2.5,
                 min_tokens: int = 150, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS):
        self._llm = llm
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._concurrency = max(1, concurrency)
        self._floor = est_s  # the duration estimate never drops below this
        self._recent: deque[float] = deque(maxlen=RECENT_CALLS)  # durations of finished calls
        self._occupied = 0  # calls running or waiting
        self._rate_cfg = decode_tok_s  # configured decode speed, tokens/s
        self._rate_seen = decode_tok_s  # running average of observed decode speed
        self._margin = margin_s  # prompt evaluation + HTTP + post-processing
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._llm.model

    @property
    def estimate(self) -> float:
        """Expected duration of one call: the slowest recent call, never below the
        configured floor. Pessimistic on purpose: a quick out_of_scope reply must
        not make the gate admit a queue that a normal answer cannot clear."""
        return max([self._floor, *self._recent])

    @property
    def decode_rate(self) -> float:
        """Tokens per second used to size output caps: the observed average, but
        never above the configured rate (learning may only make caps safer)."""
        return min(self._rate_cfg, self._rate_seen)

    def output_cap(self, remaining_s: float | None) -> int:
        """Output tokens that fit in the time left: (remaining - margin) x decode
        rate, at most the hard maximum. May be below the minimum: then no call."""
        if remaining_s is None:
            return self._max_tokens
        return max(0, min(self._max_tokens, int((remaining_s - self._margin) * self.decode_rate)))

    def __getattr__(self, name: str) -> Any:  # warm(), ready(), ... of the wrapped client
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._llm, name)

    def _observe(self, result: LLMResult) -> None:
        tokens, secs = result.completion_tokens, result.eval_s
        if tokens and secs and tokens >= MIN_RATE_SAMPLE[0] and secs >= MIN_RATE_SAMPLE[1]:
            self._rate_seen = (1 - RATE_WEIGHT) * self._rate_seen + RATE_WEIGHT * (tokens / secs)

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
                remaining = None if deadline is None else deadline - time.monotonic()
                cap = self.output_cap(remaining)
                # The wait may have been longer than estimated: do not start a call
                # (and hold the GPU) that cannot finish, or not say enough, in time.
                if remaining is not None and (remaining < self.estimate or cap < self._min_tokens):
                    raise Busy(f"only {remaining:.1f}s left after queueing (cap {cap} tokens)")
                t0 = time.monotonic()
                try:
                    result = await self._llm.complete_json(system, user, schema, max_tokens=cap)
                except LLMTruncated as exc:
                    self._recent.append(time.monotonic() - t0)
                    log.warning("LLM reply truncated: %s completion tokens used, cap %d, %.1fs",
                                exc.completion_tokens, cap, time.monotonic() - t0)
                    raise
                except Exception:
                    # A finished call, even a failed one, held the GPU this long.
                    # A cancelled call (timeout, client gone) says nothing: skipped.
                    self._recent.append(time.monotonic() - t0)
                    raise
                self._recent.append(time.monotonic() - t0)
                self._observe(result)
                log.info("LLM call: cap %d, prompt %s, completion %s tokens, %.1fs",
                         cap, result.prompt_tokens, result.completion_tokens, time.monotonic() - t0)
                return result
        finally:
            self._occupied -= 1
