"""The LLM gate's fail-fast estimate and its re-check after queueing (fake LLM, no Ollama)."""
import asyncio
import logging
import time

import pytest
from synthetic import FakeLLM, answered

from medrag.gate import RECENT_CALLS, Busy, LLMGate
from medrag.config import Settings
from medrag.llm import LLMError, LLMResult, LLMTruncated


def call(gate, deadline=None):
    return gate.complete_json("s", "u", {}, deadline)


def test_estimate_never_drops_below_the_floor():
    gate = LLMGate(FakeLLM(answered()), est_s=3.0)

    async def run():
        for _ in range(4):  # instant calls, like quick out_of_scope replies
            await call(gate)

    asyncio.run(run())
    assert gate.estimate == 3.0


def test_estimate_is_the_slowest_recent_call_then_forgets_it():
    llm = FakeLLM(answered(), delay=0.3)
    gate = LLMGate(llm, est_s=0.05)

    async def run():
        await call(gate)
        assert gate.estimate >= 0.3
        llm.delay = 0.0
        for _ in range(RECENT_CALLS):
            await call(gate)

    asyncio.run(run())
    assert gate.estimate == 0.05


def test_failed_calls_count_cancelled_calls_do_not():
    gate = LLMGate(FakeLLM(exc=LLMError("boom"), delay=0.2), est_s=0.01)

    async def run():
        with pytest.raises(LLMError):
            await call(gate)
        assert gate.estimate >= 0.2
        slow = LLMGate(FakeLLM(answered(), delay=5), est_s=0.01)
        task = asyncio.create_task(call(slow))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert slow.estimate == 0.01

    asyncio.run(run())


def test_request_that_waited_too_long_fails_fast_without_calling_the_llm():
    # The floor (0.4 s) underestimates the real call (0.8 s): the second request
    # is admitted, but after waiting ~0.8 s only ~0.2 s of its 1 s budget is left.
    llm = FakeLLM(answered(), delay=0.8)
    gate = LLMGate(llm, concurrency=1, est_s=0.4)

    async def run():
        first = asyncio.create_task(call(gate))
        await asyncio.sleep(0)  # let it take the slot
        t0 = time.monotonic()
        with pytest.raises(Busy):
            await call(gate, deadline=time.monotonic() + 1.0)
        waited = time.monotonic() - t0
        await first
        return waited

    waited = asyncio.run(run())
    assert len(llm.calls) == 1  # the doomed request never reached the model
    assert waited < 0.95  # refused as soon as the slot freed, not at the deadline


# --- output cap sized to the time left ------------------------------------

def test_cap_is_the_time_left_times_the_decode_rate_within_bounds():
    gate = LLMGate(FakeLLM(answered()), decode_tok_s=50, margin_s=2.0, min_tokens=100, max_tokens=1200)
    assert gate.output_cap(None) == 1200
    assert gate.output_cap(12.0) == 500        # (12 - 2) x 50
    assert gate.output_cap(30.0) == 1200       # clamped to the hard maximum
    assert gate.output_cap(1.0) == 0           # nothing fits


def test_the_cap_passed_to_the_llm_shrinks_with_the_remaining_budget():
    llm = FakeLLM(answered())
    gate = LLMGate(llm, est_s=0.01, decode_tok_s=50, margin_s=2.0, min_tokens=10, max_tokens=1200)

    async def run():
        await call(gate, deadline=time.monotonic() + 12.0)
        await call(gate, deadline=time.monotonic() + 40.0)  # (40 - 2) x 50 > 1200

    asyncio.run(run())
    assert 490 <= llm.caps[0] <= 500 and llm.caps[1] == 1200


def test_no_call_when_the_cap_would_be_below_the_minimum():
    llm = FakeLLM(answered())
    gate = LLMGate(llm, est_s=0.01, decode_tok_s=50, margin_s=2.0, min_tokens=150)

    async def run():
        with pytest.raises(Busy):
            await call(gate, deadline=time.monotonic() + 4.0)  # (4 - 2) x 50 = 100 < 150

    asyncio.run(run())
    assert llm.calls == []


class Timed(FakeLLM):
    """Reports a decode of `tokens` tokens in `secs` seconds."""

    def __init__(self, tokens, secs):
        super().__init__(answered())
        self.tokens, self.secs = tokens, secs

    async def complete_json(self, system, user, schema, max_tokens=None):
        r = await super().complete_json(system, user, schema, max_tokens)
        return LLMResult(data=r.data, completion_tokens=self.tokens, eval_s=self.secs)


def test_decode_rate_learns_slower_speeds_but_never_exceeds_the_configured_rate():
    slow = LLMGate(Timed(200, 10.0), decode_tok_s=50)   # 20 tok/s observed
    fast = LLMGate(Timed(1000, 10.0), decode_tok_s=50)  # 100 tok/s observed
    noisy = LLMGate(Timed(10, 0.1), decode_tok_s=50)    # too short to learn from

    async def run():
        for g in (slow, fast, noisy):
            for _ in range(5):
                await call(g)

    asyncio.run(run())
    assert slow.decode_rate < 30
    assert fast.decode_rate == 50
    assert noisy.decode_rate == 50


def test_a_truncated_reply_is_logged_with_tokens_and_cap(caplog):
    gate = LLMGate(FakeLLM(exc=LLMTruncated(321, 321)), est_s=0.01, decode_tok_s=50, margin_s=0.0,
                   min_tokens=1)

    async def run():
        with pytest.raises(LLMTruncated):
            await call(gate, deadline=time.monotonic() + 20.0)

    with caplog.at_level(logging.WARNING, logger="medrag"):
        asyncio.run(run())
    (rec,) = [r for r in caplog.records if "truncated" in r.getMessage()]
    assert "321 completion tokens" in rec.getMessage() and "cap " in rec.getMessage()


def test_settings_refuse_a_budget_not_above_the_estimate_floor(monkeypatch):
    with pytest.raises(ValueError):
        Settings(budget_s=5.0, llm_est_s=5.0)
    monkeypatch.setenv("MEDRAG_BUDGET_S", "4")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_cap_settings_come_from_the_environment(monkeypatch):
    for k, v in {"MEDRAG_MAX_OUTPUT_TOKENS": "900", "MEDRAG_MIN_OUTPUT_TOKENS": "80",
                 "MEDRAG_DECODE_TOK_S": "40.5", "MEDRAG_LLM_MARGIN_S": "3.5"}.items():
        monkeypatch.setenv(k, v)
    s = Settings.from_env()
    assert (s.max_output_tokens, s.min_output_tokens, s.decode_tok_s, s.llm_margin_s) == (900, 80, 40.5, 3.5)
