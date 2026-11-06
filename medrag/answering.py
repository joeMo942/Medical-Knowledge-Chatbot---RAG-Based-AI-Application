"""Answer generation: prompt, structured output, and server-side Citation building.

The model only chooses a `section_id` and a snippet. `doc_id`, `title` and
`section` always come from the Section itself, and a snippet that is not
verbatim text of that Section is replaced (full repair logic is #5).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .llm import LLMClient
from .models import Line, Section
from .retrieval import LexicalRetriever, tokenize

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
            },
        },
    },
    "required": ["answer", "status", "citations"],
}

SYSTEM_PROMPT = (
    "You are a drug-information assistant for pharmacists and doctors in Egypt. "
    "Answer ONLY from the knowledge base sections provided by the user message. "
    "Never use outside knowledge and never state doses. "
    "Reply in the language of the question (Arabic or English); keep drug names in English. "
    "If the sections do not contain what was asked, set status to \"out_of_scope\" and cite nothing. "
    "Otherwise set status to \"answered\" and cite the sections you used: for each citation give "
    "the section_id exactly as shown and a snippet copied character for character from that "
    "section (one line or one table row). Respond with JSON only."
)

NOT_FOUND = {
    "en": "I could not find this in the approved knowledge base.",
    "ar": "لم أجد هذه المعلومة في قاعدة المعرفة المعتمدة.",
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
    status: str
    language: str


def build_prompt(question: str, sections: list[Section]) -> str:
    # Context is the Sections' render() text and section_id only, never `path`.
    blocks = [f"[section_id: {s.section_id}]\n{s.render()}" for s in sections]
    return "KNOWLEDGE BASE SECTIONS\n\n" + "\n\n".join(blocks) + f"\n\nQUESTION\n{question}"


def _candidates(section: Section) -> list[str]:
    """Verbatim lines and data rows of the Section, in reading order."""
    out: list[str] = []
    for item in section.items:
        if isinstance(item, Line):
            out.append(item.text)
        else:
            out.extend(r.text for r in item.rows)
    out = [c for c in out if c.strip()]
    return out or [section.render().split("\n", 1)[0]]


def _verbatim(snippet: str, section: Section) -> str | None:
    s = snippet.strip()
    return s if s and s in section.render() else None


def _repair(section: Section, hints: str) -> str:
    """Simple rule: the Section line or row sharing the most terms with the
    question, answer and the model's snippet (first one on ties)."""
    want = set(tokenize(hints))
    cands = _candidates(section)
    return max(cands, key=lambda c: len(want & set(tokenize(c))))  # max keeps the first on ties


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
        snippet = _verbatim(given, sec) or _repair(sec, f"{hints} {given}")
        if (sec.section_id, snippet) in seen:
            continue
        seen.add((sec.section_id, snippet))
        out.append(Citation(doc_id=sec.kb_id, title=sec.title, section=sec.section, snippet=snippet))
    return out


async def answer_question(
    question: str, language: str, retriever: LexicalRetriever, llm: LLMClient, top_k: int
) -> ChatResponse:
    sections = retriever.search(question, top_k)
    if not sections:  # nothing lexically related: no reason to ask the model
        return ChatResponse(answer=NOT_FOUND[language], citations=[], status="out_of_scope", language=language)
    raw = await llm.complete_json(SYSTEM_PROMPT, build_prompt(question, sections), ANSWER_SCHEMA)
    answer = raw.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    citations: list[Citation] = []
    if raw.get("status") == "answered" and answer:
        citations = build_citations(raw.get("citations"), sections, f"{question} {answer}")
    if citations:
        return ChatResponse(answer=answer, citations=citations, status="answered", language=language)
    # Contract invariant: answered needs a valid Citation; otherwise out_of_scope with none.
    if raw.get("status") == "out_of_scope" and answer:
        return ChatResponse(answer=answer, citations=[], status="out_of_scope", language=language)
    return ChatResponse(answer=NOT_FOUND[language], citations=[], status="out_of_scope", language=language)
