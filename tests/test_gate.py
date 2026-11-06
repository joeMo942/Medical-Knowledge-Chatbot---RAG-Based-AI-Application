"""The LLM gate's fail-fast estimate and its re-check after queueing (fake LLM, no Ollama)."""
import asyncio
import time

import pytest
from synthetic import FakeLLM, answered

from medrag.gate import RECENT_CALLS, Busy, LLMGate
from medrag.llm import LLMError


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
