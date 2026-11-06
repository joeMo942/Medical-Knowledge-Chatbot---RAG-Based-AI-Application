"""Answer generation: prompt, structured output, and server-side Citation building.

The model only chooses a `section_id` and a snippet. `doc_id`, `title` and
`section` always come from the Section itself, and a snippet that is not
verbatim text of one line or row of that Section is replaced (full repair
logic is #5). Model free text is only ever shown for an `answered` reply.
"""
from __future__ import annotations

import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .gate import LLMGate
from .llm import LLMTruncated
from .models import Line, Section
from .retrieval import tokenize

Status = Literal["answered", "out_of_scope", "escalated"]
Language = Literal["ar", "en", "mixed"]

MIN_SNIPPET_CHARS = 8  # shorter text is only valid when it is a whole line or row

# Strict-mode compatible (OpenAI-style): every object forbids extra properties
# and lists all of its properties as required. Ollama accepts it as well.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "status": {"type": "string", "enum": ["answered", "out_of_scope"]},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"section_id": {"type": "string"}, "snippet": {"type": "string"}},
                "required": ["section_id", "snippet"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "status", "citations"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a drug-information assistant for pharmacists and doctors in Egypt. "
    "Answer ONLY from the knowledge base sections provided by the user message. "
    "Never use outside knowledge and never state doses. "
    "Reply in the language of the question (Arabic or English); keep drug names in English. "
    "Be concise: answer only what was asked, in at most about 120 words; keep any list short "
    "(the most relevant items only) and do not repeat the sections back. "
    "If the sections do not contain what was asked, set status to \"out_of_scope\" and cite nothing. "
    "Otherwise set status to \"answered\" and cite the sections you used: for each citation give "
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


def _repair(section: Section, hints: str) -> str | None:
    """Simple rule: the single line or row sharing the most terms with the
    question, answer and the model's snippet. On a tie a table data row beats a
    body line (a label such as "Trade Names في مصر:" supports nothing), then
    the first in reading order."""
    units = _units(section)
    if not units:
        return None
    want = set(tokenize(hints))
    # max keeps the first on ties
    return max(units, key=lambda u: (len(want & set(tokenize(u[0]))), u[1]))[0]


def _pick_snippet(given: str, section: Section, hints: str) -> str | None:
    """The model's snippet if valid; else its first line that is valid on its own
    (models often paste a header row plus several rows); else a repaired one."""
    if is_valid_snippet(given, section):
        return given.strip()
    for line in given.splitlines():
        if is_valid_snippet(line, section):
            return line.strip()
    return _repair(section, f"{hints} {given}")


def build_citations(raw: Any, sections: list[Section], hints: str) -> list[Citation]:
    by_id = {s.section_id: s for s in sections}
    out: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        sec = by_id.get(c.get("section_id"))
        if sec is None:
            continue  # not a retrieved Section: drop it
        given = c.get("snippet")
        given = given if isinstance(given, str) else ""
        snippet = _pick_snippet(given, sec, hints)
        if snippet is None or (sec.section_id, snippet) in seen:
            continue
        seen.add((sec.section_id, snippet))
        out.append(Citation(doc_id=sec.kb_id, title=sec.title, section=sec.section, snippet=snippet))
    return out


def out_of_scope(language: str) -> ChatResponse:
    # Fixed server text: model free text never reaches the user on out_of_scope.
    return ChatResponse(answer=NOT_FOUND[language], citations=[], status="out_of_scope", language=language)


async def answer_question(
    question: str,
    language: str,
    sections: list[Section],
    llm: LLMGate,
    deadline: float | None = None,
) -> ChatResponse:
    """`sections` are the retrieved Sections (retrieval runs off the event loop in the caller)."""
    if not sections:  # nothing lexically related: no reason to ask the model
        return out_of_scope(language)
    try:
        result = await llm.complete_json(SYSTEM_PROMPT, build_prompt(question, sections), ANSWER_SCHEMA, deadline)
    except LLMTruncated:
        # The reply hit the output cap: it is incomplete, so nothing of it is shown.
        return out_of_scope(language)
    raw = result.data
    answer = raw.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    if raw.get("status") == "answered" and answer:
        citations = build_citations(raw.get("citations"), sections, f"{question} {answer}")
        if citations:
            return ChatResponse(answer=answer, citations=citations, status="answered", language=language)
    # out_of_scope, or answered without a valid Citation (contract invariant).
    return out_of_scope(language)
