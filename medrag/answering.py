"""Answer generation: prompt, structured output, and server-side Citation building.

The model only chooses a `section_id` and a snippet. `doc_id`, `title` and
`section` always come from the Section itself, and a snippet that is not
verbatim text of one line or row of that Section is replaced by the closest real line. An answered reply
whose cited Sections do not support the question (support.py) becomes Out of Scope. Model free text is only ever shown for an `answered` reply, and an answered reply that
states a dose-like amount missing from its cited Sections becomes the fixed Dose escalation.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .dosecheck import unsupported_amounts
from .escalation import ESCALATION_TEXT
from .gate import LLMGate
from .llm import LLMTruncated
from .models import PATH_SEPARATOR, Line, Section
from .retrieval import STOP_WORDS, tokenize
from .routing import TRIGGERS, effective_trigger
from .rules import normalize
from .support import SupportCheck, stem
from .support import tokens as support_tokens

log = logging.getLogger("medrag")

Status = Literal["answered", "out_of_scope", "escalated"]
Language = Literal["ar", "en", "mixed"]

MIN_SNIPPET_CHARS = 8  # shorter text is only valid when it is a whole line or row

# Strict-mode compatible (OpenAI-style): every object forbids extra properties
# and lists all of its properties as required. Ollama accepts it as well.
# The model writes the answer and its Citations BEFORE it picks the status (Ollama generates the
# properties in this order): decided first, "out_of_scope" made it drop the covered half of a partly
# covered question; written last, it follows from what was cited.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"section_id": {"type": "string"}, "snippet": {"type": "string"}},
                "required": ["section_id", "snippet"],
                "additionalProperties": False,
            },
        },
        "status": {"type": "string", "enum": ["answered", "out_of_scope"]},
    },
    "required": ["answer", "citations", "status"],
    "additionalProperties": False,
}

# Backstop for a request whose routing call was skipped or failed and that the rules did not flag:
# the answer call also says whether the message is an Escalated trigger. Same fields as the router.
BACKSTOP_PROPERTIES: dict[str, Any] = {
    "specific_person": {"type": "boolean"},
    "trigger": {"type": "string", "enum": [*TRIGGERS, "none"]},
}
BACKSTOP_PROMPT = (
    " Before answering, classify the QUESTION: specific_person is true only if it is about one particular "
    "person; trigger is \"emergency\" (a person acutely unwell now, or who swallowed many pills or someone "
    "else's medicine), \"dose\" (asks an amount, frequency or duration to take or give; NOT how much of an "
    "ingredient a product contains), \"diagnosis\" (what condition a specific person has) or "
    "\"treatment_decision\" (choose, switch, start or stop a drug for a specific person), else \"none\". "
    "General questions about a drug, a class or a group of patients are always \"none\"."
)


def answer_schema(backstop: bool) -> dict[str, Any]:
    if not backstop:
        return ANSWER_SCHEMA
    props = {**BACKSTOP_PROPERTIES, **ANSWER_SCHEMA["properties"]}
    return {**ANSWER_SCHEMA, "properties": props, "required": [*BACKSTOP_PROPERTIES, *ANSWER_SCHEMA["required"]]}


SYSTEM_PROMPT = (
    "You are a drug-information assistant for pharmacists and doctors in Egypt. "
    "Answer ONLY from the knowledge base sections provided by the user message. "
    "Never use outside knowledge, even for a drug you know well, and never state doses. "
    "A drug name appearing in a section is not enough: the sections must state the fact that was asked "
    "(a drug listed in a section is not shown to treat a disease unless the section says so). "
    "Reply in the language of the question (Arabic or English); keep drug names in English. "
    "Be concise: answer only what was asked, in at most about 120 words; keep any list short "
    "(the most relevant items only) and do not repeat the sections back. "
    "Exception: when asked for alternatives (other Trade Names of the same drug), list EVERY Trade Name "
    "that the sections show with the same Active Ingredient, each once, and never a drug with a different "
    "Active Ingredient, even from the same class. "
    "Decide the status like this. If the sections state ANYTHING that answers part of the question, set "
    "status to \"answered\": give only that part, then add one plain sentence saying which part is not in "
    "the knowledge base (for example the price, or a second question). Set status to \"out_of_scope\", "
    "with no citations, ONLY when nothing that was asked is stated in the sections, or the question is "
    "not about drugs. "
    "For an answered reply, cite the sections you used: for each citation give "
    "the section_id exactly as shown and a snippet copied character for character from that "
    "section: ONE line or ONE table row only (never a table header row, never several rows). "
    "Respond with JSON only."
)

NOT_FOUND: dict[str, str] = {
    "en": "I could not find this in the approved knowledge base.",
    "ar": "لم أجد هذه المعلومة في قاعدة المعرفة المعتمدة.",
    "mixed": "I could not find this in the approved knowledge base.",
}


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    title: str
    section: str
    snippet: str


class ChatResponse(BaseModel):
    """Exactly the contract's 200 body: no extra fields."""

    model_config = ConfigDict(extra="forbid")
    answer: str
    citations: list[Citation]
    status: Status
    language: Language


def build_prompt(question: str, sections: list[Section]) -> str:
    # Context is the Sections' render() text and section_id only, never `path`.
    blocks = [f"[section_id: {s.section_id}]\n{s.render()}" for s in sections]
    return "KNOWLEDGE BASE SECTIONS\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION\n{question}"


def _units(section: Section) -> list[tuple[str, bool]]:
    """(text, is_table_row) for each body line and each data row, verbatim and in
    reading order. Never the heading line and never a table header row."""
    out: list[tuple[str, bool]] = []
    for item in section.items:
        if isinstance(item, Line):
            out.append((item.text, False))
        else:
            out.extend((r.text, True) for r in item.rows)
    return [(t, row) for t, row in out if t.strip()]


def candidates(section: Section) -> list[str]:
    """What a snippet may quote: each body line and each data row, verbatim."""
    return [t for t, _ in _units(section)]


def _is_word_char(ch: str) -> bool:
    # Letters, digits and combining marks (Arabic harakat) all belong to a word.
    return ch == "_" or unicodedata.category(ch)[0] in "LNM"


def _on_word_boundaries(s: str, text: str) -> bool:
    """True when some occurrence of `s` in `text` neither starts nor ends inside a word."""
    i = text.find(s)
    while i != -1:
        j = i + len(s)
        starts_ok = i == 0 or not (_is_word_char(text[i - 1]) and _is_word_char(s[0]))
        ends_ok = j == len(text) or not (_is_word_char(text[j]) and _is_word_char(s[-1]))
        if starts_ok and ends_ok:
            return True
        i = text.find(s, i + 1)
    return False


def is_valid_snippet(snippet: str, section: Section) -> bool:
    """True when the snippet is a whole line or row of the Section, or lies within
    ONE of them, is at least MIN_SNIPPET_CHARS long, and starts and ends on word
    boundaries (no "fen – Ibupro" fragments)."""
    s = snippet.strip()
    if not s:
        return False
    for c in candidates(section):
        if s == c.strip():
            return True
        if len(s) >= MIN_SNIPPET_CHARS and _on_word_boundaries(s, c):
            return True
    return False


# Bare lines that only introduce what follows (the per-class headings of KB-07 / KB-08).
_LABEL_LINES = frozenset(normalize(x) for x in (
    "Side Effects", "Side Effects (الأعراض الجانبية)", "Contraindications",
    "Contraindications (موانع الاستعمال)", "الأدوية", "الأدوية الموجودة في السوق المصري",
    "الأدوية والمستحضرات الموجودة في السوق المصري", "مستحضر خاص بالـMigraine",
))
_SUB_HEADING = re.compile(r"^(?:[A-Z]\.\s+\S|(?:أولًا|ثانيًا|ثالثًا|رابعًا|أولا|ثانيا|ثالثا|رابعا)\s*:)")
_CLAUSE_BREAK = re.compile("[,،;؛]")
LABEL_MAX_WORDS = 7


def _is_label_line(text: str, is_row: bool) -> bool:
    """A body line that introduces what follows and supports no claim by itself: a short line
    ending in ":" or "?" ("Common side effects:", "Trade Names في مصر:", "OTC ولا Prescription؟"),
    a bare class heading ("Contraindications"), or a lettered / ordinal sub-heading ("C. Nitroimidazoles").
    A sentence that happens to end in ":" (it has a clause break or is long) is content."""
    if is_row:
        return False
    t = text.strip()
    if not t:
        return False
    if normalize(t) in _LABEL_LINES:
        return True
    if t.endswith((":", "؟", "?")):
        return len(t.split()) <= LABEL_MAX_WORDS and not _CLAUSE_BREAK.search(t)
    return bool(_SUB_HEADING.match(t)) and not t.endswith(".")


# (text, is_row, is_label) for a snippet or a whole line/row of a Section.
Unit = tuple[str, bool, bool]


def _unit(snippet: str, section: Section) -> Unit:
    """Flags of a valid snippet, taken from the Section lines/rows that contain it:
    a row if any holder is a row; a label if every holder is a label line (so a
    piece of "Trade Names في مصر:" is a label as well)."""
    s = snippet.strip()
    holders = [(t, row) for t, row in _units(section) if s in t]
    is_row = any(row for _, row in holders)
    is_label = bool(holders) and all(_is_label_line(t, row) for t, row in holders)
    return s, is_row, is_label


def _best(units: list[Unit], hints: str) -> Unit:
    """Best supporting unit: never a label line if anything else exists, then the
    most terms shared with `hints`, then a table data row over a body line, then
    the line with more content (a bare heading word like "Contraindications" loses
    to "Pregnancy: ممنوعة أثناء الحمل."), then reading order (max keeps the first)."""
    want = set(tokenize(hints))

    def key(u: Unit) -> tuple[bool, int, bool, int]:
        toks = tokenize(u[0])
        return (not u[2], len(want & set(toks)), u[1], len(toks))

    return max(units, key=key)


def _stems(text: str) -> set[str]:
    return {stem(w) for w in support_tokens(text) if w not in STOP_WORDS and len(w) > 1}


def _closeness(given: set[str], line: str) -> float:
    """Dice similarity of the model's snippet (stemmed, Arabic-folded content words) and a real
    line: shared words count, and a short line that is mostly the paraphrase beats a long
    line that merely contains a word of it."""
    a, b = given, _stems(line)
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def _repair(section: Section, hints: str, given: str = "") -> str | None:
    """The real line or row of the Section closest to what the model wrote, for a snippet that
    is not verbatim. Ranked by: never a label line (it supports no claim, so it is chosen only
    when the Section has nothing else), then similarity to the model's own snippet, then terms
    shared with the question and answer (`hints`), then a table data row over a body line, then
    the line with more content, then reading order. Never a heading or a table header row:
    `_units` does not offer them."""
    units = [(t, row, _is_label_line(t, row)) for t, row in _units(section)]
    if not units:
        return None
    want = set(tokenize(hints))
    # Words of the Section's own heading or ID are not a paraphrase of any line (an invented path
    # "KB-04 › Pain" must not pull the repair towards a line that mentions pain).
    given_stems = _stems(given) - _stems(f"{section.kb_id} {section.render().split(chr(10))[0]}")

    def key(u: Unit) -> tuple[bool, float, int, bool, int]:
        toks = tokenize(u[0])
        return (not u[2], round(_closeness(given_stems, u[0]), 6), len(want & set(toks)), u[1], len(toks))

    return max(units, key=key)[0]


def _pick_snippet(given: str, section: Section, hints: str) -> str | None:
    """The model's snippet if it is valid and not a label. Models often paste
    several lines (a header row plus rows, or a label plus items): then the best
    of its own lines that are valid on their own, unless that is only a label.
    Otherwise a repair from the whole Section. `hints` is the question plus the
    answer. A label is verbatim KB text, but it supports no claim, so it is never
    the citation when the Section has anything else."""
    if is_valid_snippet(given, section):
        text, _, is_label = _unit(given, section)
        if not is_label:
            return text
    else:
        own = [_unit(ln, section) for ln in given.splitlines() if is_valid_snippet(ln, section)]
        if own:
            text, _, is_label = _best(own, hints)
            if not is_label:
                return text
    return _repair(section, hints, given)


def _by_prefix(section_id: Any, snippet: str, sections: list[Section]) -> Section | None:
    """A small model often writes only the start of a long section_id ("KB-03", or "KB-02 > Disease"). That names a
    retrieved Section when it is the only one starting that way, or when only one of them holds the snippet word for word;
    anything else (an invented ID, an ambiguous prefix) is still dropped."""
    if not isinstance(section_id, str) or not section_id.strip():
        return None
    sid = section_id.strip()
    starts = [s for s in sections if s.section_id.startswith(sid + PATH_SEPARATOR)]
    if len(starts) > 1:
        starts = [s for s in starts if is_valid_snippet(snippet, s)]
    return starts[0] if len(starts) == 1 else None


def build_cited(raw: Any, sections: list[Section], hints: str) -> list[tuple[Section, Citation]]:
    """Each valid Citation with the retrieved Section it points to."""
    by_id = {s.section_id: s for s in sections}
    out: list[tuple[Section, Citation]] = []
    seen: set[tuple[str, str]] = set()
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        given = c.get("snippet")
        given = given if isinstance(given, str) else ""
        sec = by_id.get(c.get("section_id")) or _by_prefix(c.get("section_id"), given, sections)
        if sec is None:
            continue  # not a retrieved Section: drop it
        snippet = _pick_snippet(given, sec, hints)
        if snippet is None or (sec.section_id, snippet) in seen:
            continue
        seen.add((sec.section_id, snippet))
        out.append((sec, Citation(doc_id=sec.kb_id, title=sec.title, section=sec.section, snippet=snippet)))
    return out


def build_citations(raw: Any, sections: list[Section], hints: str) -> list[Citation]:
    return [c for _, c in build_cited(raw, sections, hints)]


def escalated(trigger: str, language: str) -> ChatResponse:
    # Fixed server text per trigger: model text never reaches an Escalated reply.
    text = ESCALATION_TEXT[trigger]["en" if language == "en" else "ar"]
    return ChatResponse(answer=text, citations=[], status="escalated", language=language)


def out_of_scope(language: str) -> ChatResponse:
    # Fixed server text: model free text never reaches the user on out_of_scope.
    return ChatResponse(answer=NOT_FOUND[language], citations=[], status="out_of_scope", language=language)


async def answer_question(
    question: str,
    language: str,
    sections: list[Section],
    llm: LLMGate,
    deadline: float | None = None,
    backstop: bool = False,
    support: SupportCheck | None = None,
) -> ChatResponse:
    """`sections` are the retrieved Sections (retrieval runs off the event loop in the caller).
    `backstop`: the routing call did not run, so the answer call also classifies the message.
    `support`: the support check (support.py); without it every valid Citation is accepted."""
    if not sections:  # nothing related: no reason to ask the model
        log.info("out_of_scope: nothing retrieved")
        return out_of_scope(language)
    system = SYSTEM_PROMPT + BACKSTOP_PROMPT if backstop else SYSTEM_PROMPT
    try:
        result = await llm.complete_json(system, build_prompt(question, sections), answer_schema(backstop), deadline)
    except LLMTruncated:
        # The reply hit the output cap: it is incomplete, so nothing of it is shown.
        log.info("out_of_scope: the reply hit the output cap")
        return out_of_scope(language)
    raw = result.data
    if backstop:
        trigger = effective_trigger(raw)
        if trigger:
            return escalated(trigger, language)
    answer = raw.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    if raw.get("status") == "answered" and answer:
        cited = build_cited(raw.get("citations"), sections, f"{question} {answer}")
        if cited:
            cited_sections = list({s.section_id: s for s, _ in cited}.values())
            cited_text = "\n".join(s.render() for s in cited_sections)
            if unsupported_amounts(answer, cited_text):
                # A dose-like amount the cited documents do not contain: never shown. Runs first:
                # Escalated takes precedence over Out of Scope.
                return escalated("dose", language)
            if support is not None and not support.supported(question, cited_sections):
                # Real Sections, but none of them is about what was asked (drug names alone do not count).
                log.info("out_of_scope: the support check refused the cited Sections")
                return out_of_scope(language)
            return ChatResponse(answer=answer, citations=[c for _, c in cited], status="answered", language=language)
    # out_of_scope, or answered without a valid Citation (contract invariant).
    log.info("out_of_scope: the model said %r%s; reply: %.700s", raw.get("status"), "" if answer else " with no answer",
             json.dumps(raw, ensure_ascii=False))
    return out_of_scope(language)
