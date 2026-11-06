"""Per-request LLM usage (tokens and number of calls), reported in the optional
X-Medrag-Usage response header. Never part of the response body (the contract is fixed).

A request installs a UsageMeter in a ContextVar before its pipeline task is created;
asyncio tasks and to_thread copy the context, so the gate adds to the same meter."""
from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass

USAGE_HEADER = "X-Medrag-Usage"


@dataclass
class UsageMeter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    def add(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        """One LLM call; tokens the backend did not report count as 0."""
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


def record_call(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    meter = _current.get()
    if meter is not None:
        meter.add(prompt_tokens, completion_tokens)
