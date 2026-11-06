"""Split a KB Document's body blocks into Sections (disease > drug class).

Pure logic over `Block`s so it can be tested without any .docx file.

Rules (Word heading styles are unreliable, so they are only a fallback):
  * A paragraph whose text (leading numbering stripped) equals one of the
    manifest's disease headings starts that disease. Repeating the current
    disease heading (KB-05 has it as "1." and "10.") changes nothing.
  * A paragraph that is an "OTC vs Prescription" heading starts an "otc"
    Section of its own, provided it is not a Heading 3 and has a table of
    its own before the next heading. Otherwise it stays a line inside the
    current class (KB-08 has one inside Omega-3, KB-03 has inline ones).
  * Any other numbered paragraph ("4. Fibrates", "2. NSAIDs - ...", whatever
    its style) is a drug-class heading and starts a Section.
  * Any other Heading 1/2 paragraph is also a drug-class heading.
  * Everything else (lettered sub-headings such as "A. Sucralfate", "Side
    Effects", Heading 3 labels, body lines, tables) stays inside the current
    Section.
  * An OTC classification table with no heading before it (KB-02 has one,
    after an empty Heading 2) opens an "otc" Section with no heading of its
    own; it cites its disease title.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Item, Line, Section, SectionKind
from .tables import build_table

_NUMBERING = re.compile(r"^\s*\d+\s*[.)]\s+")
_NUMBERED = re.compile(r"^\s*\d+\s*[.)]\s+\S")
# Single-line only: KB-03 has body paragraphs that open with the same words
# followed by a line break and the explanation; those are not headings.
_OTC_HEADING = re.compile(r"^(?:ملخص\s+)?OTC\s+(?:vs|ولا)\s+Prescription[^\n]*$", re.I)
_TOP_HEADING_STYLES = {"Heading 1", "Heading 2"}


@dataclass(frozen=True)
class ParagraphBlock:
    text: str
    style: str = ""


@dataclass(frozen=True)
class TableBlock:
    rows: tuple[tuple[str, ...], ...]  # raw cells, header row first


Block = ParagraphBlock | TableBlock


class SplitError(ValueError):
    pass


def strip_numbering(text: str) -> str:
    return _NUMBERING.sub("", text.strip(), count=1)


class _Builder:
    def __init__(self, kb_id: str, title: str):
        self.kb_id = kb_id
        self.title = title
        self.heading: str | None = None  # own heading, numbering stripped
        self.kind: SectionKind = "overview"
        self.items: list[Item] = []

    def build(self) -> Section | None:
        if self.kind == "overview" and not self.items:
            return None
        return Section(
            kb_id=self.kb_id,
            title=self.title,
            # No heading of its own: cite the nearest enclosing real heading.
            section=self.heading if self.heading is not None else self.title,
            kind=self.kind,
            items=tuple(self.items),
            heading_present=self.heading is not None,
        )


def _is_boundary(block: Block, known: set[str]) -> bool:
    """A paragraph that starts a new disease or Section."""
    if not isinstance(block, ParagraphBlock) or not block.text.strip():
        return False
    text = block.text.strip()
    return bool(
        strip_numbering(text) in known
        or _OTC_HEADING.match(text)
        or _NUMBERED.match(text)
        or block.style in _TOP_HEADING_STYLES
    )


def _has_own_table(blocks: list[Block], start: int, known: set[str]) -> bool:
    """Is there a table after blocks[start] before the next heading?"""
    for block in blocks[start + 1 :]:
        if isinstance(block, TableBlock):
            return True
        if _is_boundary(block, known):
            return False
    return False


def split_sections(
    kb_id: str, disease_headings: tuple[str, ...] | list[str], blocks: list[Block]
) -> tuple[tuple[str, ...], tuple[Section, ...]]:
    """Return (disease titles in order, Sections in document order)."""
    known = {strip_numbering(h) for h in disease_headings}
    expected = list(dict.fromkeys(strip_numbering(h) for h in disease_headings))

    sections: list[Section] = []
    seen_diseases: list[str] = []
    cur: _Builder | None = None

    def flush() -> None:
        if cur is not None and (s := cur.build()) is not None:
            sections.append(s)

    for i, block in enumerate(blocks):
        if isinstance(block, ParagraphBlock):
            text = block.text.strip()
            if not text:
                continue
            name = strip_numbering(text)
            if name in known:
                if cur is None or cur.title != name:
                    flush()
                    cur = _Builder(kb_id, name)
                    seen_diseases.append(name)
                continue
            if cur is None:
                raise SplitError(
                    f"{kb_id}: text before the first disease heading: {text[:40]!r}"
                )
            if _OTC_HEADING.match(text):
                if block.style != "Heading 3" and _has_own_table(blocks, i, known):
                    flush()
                    cur = _Builder(kb_id, cur.title)
                    cur.heading, cur.kind = text, "otc"
                else:
                    cur.items.append(Line(block.text))
            elif _NUMBERED.match(text) or block.style in _TOP_HEADING_STYLES:
                flush()
                cur = _Builder(kb_id, cur.title)
                cur.heading, cur.kind = name, "class"
            else:
                cur.items.append(Line(block.text))
        else:
            if cur is None:
                raise SplitError(f"{kb_id}: table before the first disease heading")
            if not block.rows or any(len(r) == 0 for r in block.rows):
                raise SplitError(f"{kb_id}: empty table or table row with no cells")
            table = build_table(block.rows)
            if table.kind == "otc" and cur.kind != "otc":
                flush()
                cur = _Builder(kb_id, cur.title)
                cur.kind = "otc"  # no heading of its own in the source
            cur.items.append(table)
    flush()

    if seen_diseases != expected:
        raise SplitError(
            f"{kb_id}: disease headings found {seen_diseases} != manifest {expected}"
        )
    paths = [s.path for s in sections]
    dupes = {p for p in paths if paths.count(p) > 1}
    if dupes:
        raise SplitError(f"{kb_id}: duplicate Section paths {sorted(dupes)}")
    return tuple(seen_diseases), tuple(sections)
