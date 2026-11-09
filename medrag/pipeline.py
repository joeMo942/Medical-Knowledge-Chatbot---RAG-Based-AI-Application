"""The order of decisions for one message:

    rules (+ routing call) -> Escalated? return it
                           -> else retrieval (+ no-match shortcut) -> answer call -> Citations -> dose check -> support check -> invariants

Escalation is decided BEFORE retrieval and before the answer call, so an emergency or a Dose
request is never dismissed as out_of_scope by the no-match shortcut and never reaches the model
that writes answers. The rules can only raise a status; a failed routing call is not an error.

In a Session the routing call also gets the last few turns and returns the message as a standalone
question; retrieval, the answer call and the support check then use that question. Earlier turns are
never given to the answer call, so they are never a source of facts. Every turn is checked for
escalation on its own, on the new message and on its standalone rewrite (so "how many is dangerous?"
after "my son swallowed Panadol" is still caught); history can only raise a status, never lower one.
The rewrite is validated against the message and the KB's drug names before use (support.rewrite_ok), and the
escalation rules run on it unvalidated (they can only raise). If the routing call is skipped or fails, or the
rewrite is refused, the raw message is used as before.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from .answering import ChatResponse, answer_question, escalated
from .gate import LLMGate
from .models import Section
from .routing import Route, route_message
from .rules import check
from .sessions import Turn
from .support import SupportCheck

log = logging.getLogger("medrag")
Search = Callable[[str], Awaitable[list[Section]]]


@dataclass(frozen=True)
class Decision:
    trigger: str | None  # emergency | dose | diagnosis | treatment_decision, or None
    route: Route | None  # None when the routing call was skipped or failed


async def decide(message: str, llm: LLMGate, deadline: float | None = None,
                 history: Sequence[Turn] = ()) -> Decision:
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
    route = await route_message(message, llm, deadline, history)
    # The rewrite stands in for the message only in a Session, and the wording rules run on it too.
    rewrite_flag = check(route.question) if history and route and route.question else None
    if rewrite_flag == "emergency" or (route and route.trigger == "emergency"):
        return Decision("emergency", route)
    if flagged or rewrite_flag:
        return Decision(flagged or rewrite_flag, route)
    return Decision(route.trigger if route else None, route)


def standalone_question(message: str, decision: Decision, history: Sequence[Turn],
                        support: SupportCheck | None = None) -> str:
    """The text retrieval, the answer call and the support check work on: the router's standalone
    rewrite in a Session, the raw message otherwise. The rewrite is only used when it passes
    `SupportCheck.rewrite_ok` (it adds nothing but KB drug or class names to the message), so a fact
    from an earlier reply cannot enter the question through it; otherwise the raw message is used."""
    if not (history and decision.route and decision.route.question):
        return message
    rewrite = decision.route.question
    if support is None or not support.rewrite_ok(message, rewrite):
        log.info("standalone rewrite refused (it carries words that are not in the message or the KB's drug names)")
        return message
    return rewrite


async def respond(message: str, language: str, search: Search, llm: LLMGate,
                  deadline: float | None = None, support: SupportCheck | None = None,
                  history: Sequence[Turn] = ()) -> ChatResponse:
    """`language` is that of the current message, whatever the history was written in."""
    decision = await decide(message, llm, deadline, history)
    if decision.trigger:
        return escalated(decision.trigger, language)
    question = standalone_question(message, decision, history, support)
    sections = await search(question)
    return await answer_question(question, language, sections, llm, deadline, backstop=decision.route is None,
                                 support=support)
