"""Serialises LLM calls (Ollama runs one at a time), fails fast instead of queueing
a request that cannot finish within its time budget, and sizes each call's output
cap to the time that request has left.

Two kinds of call share the queue: the short routing call that opens every request
(escalation, see routing.py) and the answer call. Routing calls go first (priority),
so a burst of requests is classified before any answer is generated, and the two kinds
keep separate duration estimates: a slow routing call (a cold model) must not make
answer calls look too slow to admit."""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import time
from collections import deque
from typing import Any

from .llm import DEFAULT_MAX_OUTPUT_TOKENS, LLMClient, LLMResult, LLMTruncated
from .usage import record_call

log = logging.getLogger("medrag")

RECENT_CALLS = 5  # the duration estimate is the slowest of the last few calls
RATE_WEIGHT = 0.3  # weight of a new observation in the decode-rate average
MIN_RATE_SAMPLE = (50, 0.5)  # (tokens, seconds): shorter decodes are too noisy to learn from
ROUTE_EST_S = 1.5  # floor of the routing call's duration estimate (about 1 s measured, with some room)
ANSWER_PRIORITY, ROUTE_PRIORITY = 1, 0  # lower runs first


class Busy(RuntimeError):
    """The LLM queue is too long for this request to finish within its budget."""


class _Slots:
    """`n` slots handed out by priority, then in arrival order. Cancellation-safe."""

    def __init__(self, n: int):
        self._free = n
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()

    async def acquire(self, priority: int) -> None:
        if self._free > 0 and not self._waiters:
            self._free -= 1
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():  # the slot was already handed to us: give it back
                self.release()
            raise

    def release(self) -> None:
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():  # skip waiters that gave up
                fut.set_result(None)
                return
        self._free += 1


class LLMGate:
    def __init__(self, llm: LLMClient, concurrency: int = 1, est_s: float = 5.0,
                 decode_tok_s: float = 53.0, margin_s: float = 2.5,
                 min_tokens: int = 150, max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS):
        self._llm = llm
        self._slots = _Slots(max(1, concurrency))
        self._concurrency = max(1, concurrency)
        self._floor = est_s  # the duration estimate never drops below this
        self._recent: deque[float] = deque(maxlen=RECENT_CALLS)  # durations of finished answer calls
        self._recent_route: deque[float] = deque(maxlen=RECENT_CALLS)  # same, routing calls
        self._occupied = 0  # calls running or waiting
        self._running: list[bool] = []  # one entry per running call: True for a routing call
        self._queued_routes = 0  # routing calls waiting for a slot
        self._rate_cfg = decode_tok_s  # configured decode speed, tokens/s
        self._rate_seen = decode_tok_s  # running average of observed decode speed
        self._margin = margin_s  # prompt evaluation + HTTP + post-processing
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._llm.model

    @property
    def idle(self) -> bool:
        """True when no LLM call is running or waiting."""
        return self._occupied == 0

    @property
    def estimate(self) -> float:
        """Expected duration of one answer call: the slowest recent one, never below the
        configured floor. Pessimistic on purpose: a quick out_of_scope reply must
        not make the gate admit a queue that a normal answer cannot clear."""
        return max([self._floor, *self._recent])

    @property
    def route_estimate(self) -> float:
        """Expected duration of one routing call (its own history, never the answers')."""
        return max([ROUTE_EST_S, *self._recent_route])

    def can_route(self, deadline: float | None) -> bool:
        """False when a routing call would leave this request without the time for its answer
        call: the wait for the call in flight and the routing calls queued before it (they go
        first), the call itself, and then one answer call."""
        if deadline is None:
            return True
        wait = sum(self.route_estimate if r else self.estimate for r in self._running) / self._concurrency
        wait += self._queued_routes * self.route_estimate / self._concurrency
        return deadline - time.monotonic() >= wait + self.route_estimate + self.estimate

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
                            deadline: float | None = None, max_tokens: int | None = None,
                            routing: bool = False) -> LLMResult:
        """`max_tokens` is a smaller output cap for a call known to be short. `routing` marks the
        short classification call: it queues ahead of answer calls and has its own duration history."""
        history = self._recent_route if routing else self._recent
        est = self.route_estimate if routing else self.estimate
        if deadline is not None and not routing:
            waves_ahead = self._occupied // self._concurrency
            expected = (waves_ahead + 1) * self.estimate
            if expected > deadline - time.monotonic():
                raise Busy(f"{self._occupied} requests ahead; about {expected:.0f}s needed")
        self._occupied += 1
        if routing:
            self._queued_routes += 1
        queued = routing
        try:
            await self._slots.acquire(ROUTE_PRIORITY if routing else ANSWER_PRIORITY)
            if queued:
                self._queued_routes -= 1
                queued = False
            self._running.append(routing)
            try:
                remaining = None if deadline is None else deadline - time.monotonic()
                cap = self.output_cap(remaining)
                floor = self._min_tokens
                if max_tokens is not None:
                    cap, floor = min(cap, max_tokens), min(floor, max_tokens)
                # The wait may have been longer than estimated: do not start a call
                # (and hold the GPU) that cannot finish, or not say enough, in time.
                if remaining is not None and (remaining < est or cap < floor):
                    raise Busy(f"only {remaining:.1f}s left after queueing (cap {cap} tokens)")
                t0 = time.monotonic()
                try:
                    result = await self._llm.complete_json(system, user, schema, max_tokens=cap)
                except LLMTruncated as exc:
                    history.append(time.monotonic() - t0)
                    record_call(exc.prompt_tokens, exc.completion_tokens)
                    log.warning("LLM reply truncated: %s completion tokens used, cap %d, %.1fs",
                                exc.completion_tokens, cap, time.monotonic() - t0)
                    raise
                except Exception:
                    # A finished call, even a failed one, held the GPU this long.
                    # A cancelled call (timeout, client gone) says nothing: skipped.
                    history.append(time.monotonic() - t0)
                    record_call(None, None)
                    raise
                history.append(time.monotonic() - t0)
                if not routing:
                    self._observe(result)
                record_call(result.prompt_tokens, result.completion_tokens)
                log.info("LLM call (%s): cap %d, prompt %s, completion %s tokens, %.1fs",
                         "routing" if routing else "answer", cap, result.prompt_tokens,
                         result.completion_tokens, time.monotonic() - t0)
                return result
            finally:
                self._running.remove(routing)
                self._slots.release()
        finally:
            if queued:  # cancelled while waiting
                self._queued_routes -= 1
            self._occupied -= 1
