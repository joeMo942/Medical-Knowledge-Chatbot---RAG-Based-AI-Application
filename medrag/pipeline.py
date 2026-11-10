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

A Franco-Arabic message (language `mixed`) is also rewritten, on the first turn too, into Arabic letters (word for
word, see `standalone_question`); Trade Names and Active Ingredients in Arabic letters are then mapped to the KB's
English spelling (arabicnames.py). The Answer Language is decided once (language.py) and handed to the answer call.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from .alternatives import ALT_WORDS
from .answering import ChatResponse, answer_question, escalated
from .arabicnames import ArabicNames
from .drugs import tokens
from .gate import LLMGate
from .language import detect_language, refine
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
    The rules run on the raw message always, and on the router's rewrite wherever the rewrite stands in for the
    message (in a Session, and for a Franco-Arabic message): a trigger found there can only raise.
    """
    flagged = check(message)
    if flagged == "emergency":
        return Decision("emergency", None)
    route = await route_message(message, llm, deadline, history)
    # The rewrite is rule-checked exactly when it may stand in for the message (see standalone_question): in a Session, or
    # when the Answer Language, as refined by the router, is Franco-Arabic.
    franco = refine(detect_language(message), message, route.language if route else None) == "mixed"
    rewrite_flag = check(route.question) if (history or franco) and route and route.question else None
    if rewrite_flag == "emergency" or (route and route.trigger == "emergency"):
        return Decision("emergency", route)
    if flagged or rewrite_flag:
        return Decision(flagged or rewrite_flag, route)
    return Decision(route.trigger if route else None, route)


def standalone_question(message: str, decision: Decision, history: Sequence[Turn],
                        support: SupportCheck | None = None, language: str = "en") -> str:
    """The text retrieval, the answer call and the support check work on: the router's rewrite, or the raw message.

    The rewrite is used in two cases, and only when it passes `SupportCheck.rewrite_ok`:
      - in a Session: it resolves "it" / "its" (it adds nothing but KB drug or Drug Class names to the message), so a
        fact from an earlier reply cannot enter the question through it;
      - for a Franco-Arabic message (`language == "mixed"`): it is the message in Arabic letters, word for word, because
        neither the KB nor the support check reads Latin-letter Arabic (`SupportCheck.translated`): every word of it
        that is not the message's own word in Arabic letters is deleted (an added gloss could name a condition the user did
        not), and what is left must keep every content word of the message and what it asks. A translation into a
        different word is how a condition the user never named would be smuggled in, so that rewrite is refused.
    Otherwise, or when the rewrite is refused, the raw message is used."""
    if not (decision.route and decision.route.question):
        return message
    franco = language == "mixed"
    if not (history or franco):
        return message
    rewrite = decision.route.question
    if support is not None and franco:
        usable = support.translated(message, rewrite, from_history=bool(history))
    else:
        usable = rewrite if support is not None and support.rewrite_ok(message, rewrite) else None
    if usable is None:
        log.info("standalone rewrite refused (it carries words that are not in the message%s)",
                 "" if franco else " or the KB's drug names")
        return message
    return usable


async def respond(message: str, language: str, search: Search, llm: LLMGate,
                  deadline: float | None = None, support: SupportCheck | None = None,
                  history: Sequence[Turn] = (), names: ArabicNames | None = None) -> ChatResponse:
    """`language` is the heuristic Answer Language of the current message (language.detect_language), whatever the history
    was written in. It is refined once by the routing call (`language.refine`) and then fixed: the escalation text, the
    answer call (told to write in it) and the returned label all use this one value."""
    decision = await decide(message, llm, deadline, history)
    language = refine(language, message, decision.route.language if decision.route else None)
    if decision.trigger:
        return escalated(decision.trigger, language)
    question = standalone_question(message, decision, history, support, language)
    if names is not None:  # Trade Names and Active Ingredients typed in Arabic letters, in the KB's English spelling
        question, mapped = names.map(question)
        if mapped:
            log.info("Arabic-letter names mapped: %s", ", ".join(f"{m.original} -> {m.name}" for m in mapped))
    if question != message:
        log.info("searching for %r instead of the message", question)
    sections = await search(question)
    alternatives = bool({*tokens(question), *tokens(message)} & ALT_WORDS)
    return await answer_question(question, language, sections, llm, deadline, backstop=decision.route is None,
                                 support=support, alternatives=alternatives)
