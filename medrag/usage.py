"""Per-request model usage: every call is recorded with its kind, model, token counts and latency. The totals of the
LLM calls are reported in the optional X-Medrag-Usage response header (never in the response body: the contract is
fixed); the per-call records feed the call log (analytics.py).

A request installs a UsageMeter in a ContextVar before its pipeline task is created;
asyncio tasks and to_thread copy the context, so the gate adds to the same meter."""
from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field

USAGE_HEADER = "X-Medrag-Usage"
ROUTING, ANSWER, EMBEDDING = "routing", "answer", "embedding"  # the kinds of call a request makes
LLM_KINDS = (ROUTING, ANSWER)  # the calls of the model server; an embedding runs in-process and is not an LLM call


@dataclass(frozen=True)
class CallRecord:
    """One model call. Tokens are what the model reported (the server's usage, or the tokenizer's own count for
    the in-process embedding), 0 when it did not report them."""
    kind: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class UsageMeter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    calls: list[CallRecord] = field(default_factory=list)  # every call, LLM or embedding, in order

    def add(self, prompt_tokens: int | None, completion_tokens: int | None, kind: str = ANSWER,
            model: str = "", latency_s: float = 0.0) -> None:
        """One model call; tokens the backend did not report count as 0. Only LLM calls enter the header totals."""
        self.calls.append(CallRecord(kind, model, prompt_tokens or 0, completion_tokens or 0, latency_s))
        if kind in LLM_KINDS:
            self.llm_calls += 1
            self.prompt_tokens += prompt_tokens or 0
            self.completion_tokens += completion_tokens or 0

    def header_value(self) -> str:
        # Compact JSON: ASCII only, so it is a legal header value.
        return json.dumps({"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                           "llm_calls": self.llm_calls}, separators=(",", ":"))


_current: ContextVar[UsageMeter | None] = ContextVar("medrag_usage", default=None)


def start_meter() -> UsageMeter:
    meter = UsageMeter()
    _current.set(meter)
    return meter


def record_call(prompt_tokens: int | None, completion_tokens: int | None, kind: str = ANSWER,
                model: str = "", latency_s: float = 0.0) -> None:
    meter = _current.get()
    if meter is not None:
        meter.add(prompt_tokens, completion_tokens, kind, model, latency_s)
