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

from .answering import ChatResponse, answer_question, escalated
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
    """Rules first, then the routing call; the two are combined so that nothing is ever lowered.

    - Emergency wording: Escalated at once (no routing call needed).
    - Dose wording: still routed, because the router may UPGRADE it to an emergency ("what dose is
      toxic" after a child swallowed pills needs the emergency text); it can never downgrade it.
    - Nothing flagged: the router's trigger, if any. If the router failed or was skipped, the
      answer call carries a backstop classification (see answering.answer_question).
    """
    flagged = check(message)
    if flagged == "emergency":
        return Decision("emergency", None)
    route = await route_message(message, llm, deadline)
    if route and route.trigger == "emergency":
        return Decision("emergency", route)
    if flagged:
        return Decision(flagged, route)
    return Decision(route.trigger if route else None, route)


async def respond(message: str, language: str, search: Search, llm: LLMGate,
                  deadline: float | None = None) -> ChatResponse:
    decision = await decide(message, llm, deadline)
    if decision.trigger:
        return escalated(decision.trigger, language)
    sections = await search(message)
    return await answer_question(message, language, sections, llm, deadline, backstop=decision.route is None)
