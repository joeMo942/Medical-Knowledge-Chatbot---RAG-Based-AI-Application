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

A question word that is a typo of exactly one KB word is corrected before search and the support check (typos.py); the
escalation rules run on the raw message AND the corrected one (so "pregnent" is still a pregnancy), and an Answered reply then
starts with "Did you mean: ...?".

A first-person self-care question with no risk qualifier (selfcare.py, ADR 0003) is the one place a status is lowered: the
router's treatment_decision becomes none. It never lowers an emergency, a dose or a diagnosis. The Answered reply then ends with
the fixed note that a pharmacist can say which one suits the person, added here so the model cannot leave it out.
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
from .selfcare import judge, pharmacist_note
from .sessions import Turn
from .support import SupportCheck
from .typos import Corrected, TypoCorrector, did_you_mean

log = logging.getLogger("medrag")
Search = Callable[[str], Awaitable[list[Section]]]


@dataclass(frozen=True)
class Decision:
    trigger: str | None  # emergency | dose | diagnosis | treatment_decision, or None
    route: Route | None  # None when the routing call was skipped or failed
    self_care: bool = False  # the user's own symptom, asking what to take, no risk qualifier (selfcare.py): the note is added
    lowered: bool = False  # a lookup or self-care question: a backstop treatment_decision is ignored too


def _check_all(texts: Sequence[str]) -> str | None:
    """The rule trigger of the strongest of several forms of one message (emergency before dose)."""
    found = {check(t) for t in texts}
    return "emergency" if "emergency" in found else "dose" if "dose" in found else None


async def decide(message: str, llm: LLMGate, deadline: float | None = None,
                 history: Sequence[Turn] = (), also: Sequence[str] = ()) -> Decision:
    """Rules first, then the routing call; the two are combined so that nothing is ever lowered.

    - Emergency wording: Escalated at once (no routing call needed).
    - Dose wording: still routed, because the router may UPGRADE it to an emergency ("what dose is
      toxic" after a child swallowed pills needs the emergency text); it can never downgrade it.
    - Nothing flagged: the router's trigger, if any. If the router failed or was skipped, the
      answer call carries a backstop classification (see answering.answer_question).
    The rules run on the raw message always, and on the router's rewrite wherever the rewrite stands in for the
    message (in a Session, and for a Franco-Arabic message): a trigger found there can only raise. `also` holds other forms
    of the message (its typo-corrected text): the rules run on them too, and they can only raise as well.
    `selfcare.judge` applies the self-care and lookup policy (ADR 0003): a first-person "what can I take / use" ask is a lookup
    or a self-care question and lowers the router's treatment_decision to none (and nothing else, never emergency, dose or
    diagnosis); with a risk qualifier in the message, its corrected form, the router's rewrite or an earlier turn, or about
    another specific person, it is RAISED to a treatment_decision even when the router said none; with dose wording it is a dose.
    """
    flagged = _check_all([message, *also])
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
    trigger = route.trigger if route else None
    verdict = judge(message, *also, context=tuple(t.message for t in history), rewrite=route.question if route else None)
    if verdict.raise_to and trigger in (None, "treatment_decision"):
        if trigger != verdict.raise_to:
            log.info("first-person ask for what to take: escalated as %s by the self-care rules", verdict.raise_to)
        return Decision(verdict.raise_to, route)
    if verdict.lower and trigger == "treatment_decision":
        log.info("self-care or lookup question: the router's treatment_decision is lowered to none")
        trigger = None
    return Decision(trigger, route, verdict.self_care and trigger is None, verdict.lower and trigger is None)


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
                  history: Sequence[Turn] = (), names: ArabicNames | None = None,
                  typos: TypoCorrector | None = None) -> ChatResponse:
    """`language` is the heuristic Answer Language of the current message (language.detect_language), whatever the history
    was written in. It is refined once by the routing call (`language.refine`) and then fixed: the escalation text, the
    answer call (told to write in it) and the returned label all use this one value."""
    typed = typos.correct(message, language) if typos is not None else Corrected(message)
    decision = await decide(message, llm, deadline, history, (typed.text,) if typed.changes else ())
    language = refine(language, message, decision.route.language if decision.route else None)
    if decision.trigger:
        return escalated(decision.trigger, language)
    question = standalone_question(message, decision, history, support, language)
    if names is not None:  # Trade Names and Active Ingredients typed in Arabic letters, in the KB's English spelling
        question, mapped = names.map(question)
        if mapped:
            log.info("Arabic-letter names mapped: %s", ", ".join(f"{m.original} -> {m.name}" for m in mapped))
    fixed = Corrected(question)
    if typos is not None and language != "mixed":
        fixed = typed if question == message and typed.changes else typos.correct(question, language)
        question = fixed.text
    if question != message:
        log.info("searching for %r instead of the message", question)
    sections = await search(question)
    alternatives = bool({*tokens(question), *tokens(message)} & ALT_WORDS)
    result = await answer_question(question, language, sections, llm, deadline, backstop=decision.route is None,
                                   support=support, alternatives=alternatives, self_care=decision.self_care,
                                   lookup=decision.lowered)
    if result.status != "answered":
        return result  # the fixed Out of Scope and Escalated texts stay exactly as they are
    answer = result.answer
    if fixed.changes:
        answer = f"{did_you_mean(fixed.words, language)}\n{answer}"
    if decision.self_care:
        answer = f"{answer}\n\n{pharmacist_note(language)}"
    return result.model_copy(update={"answer": answer})
