"""The order of decisions for one message:

    rules (+ routing call) -> Escalated? return it
                           -> else retrieval (+ no-match shortcut) -> answer call -> Citations -> invariants

Escalation is decided BEFORE retrieval and before the answer call, so an emergency or a Dose
request is never dismissed as out_of_scope by the no-match shortcut and never reaches the model
that writes answers. The rules can only raise a status; a failed routing call is not an error.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .answering import ChatResponse, answer_question
from .escalation import escalated
from .gate import LLMGate
from .models import Section
from .routing import Route, route_message
from .rules import check

Search = Callable[[str], Awaitable[list[Section]]]


@dataclass(frozen=True)
class Decision:
    trigger: str | None  # emergency | dose | diagnosis | treatment_decision, or None
    route: Route | None  # None when the routing call was skipped or failed


async def decide(message: str, llm: LLMGate, deadline: float | None = None) -> Decision:
    """The rules first: a message they flag is Escalated at once, without a routing call (it could
    only confirm or, wrongly, not). Otherwise the routing call; if it fails the rules' "no" stands."""
    flagged = check(message)
    if flagged:
        return Decision(flagged, None)
    route = await route_message(message, llm, deadline)
    return Decision(route.trigger if route else None, route)


async def respond(message: str, language: str, search: Search, llm: LLMGate,
                  deadline: float | None = None) -> ChatResponse:
    decision = await decide(message, llm, deadline)
    if decision.trigger:
        return escalated(decision.trigger, language)
    sections = await search(message)
    return await answer_question(message, language, sections, llm, deadline)
