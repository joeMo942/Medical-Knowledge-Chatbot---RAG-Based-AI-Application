"""Split a KB Document's body blocks into Sections (disease > drug class).

Pure logic over `Block`s so it can be tested without any .docx file.

Rules (Word heading styles are unreliable, so they are only a fallback):
  * A paragraph whose text (leading numbering stripped) equals one of the
    manifest's disease headings starts that disease. Repeating the current
    disease heading (KB-05 has it as "1." and "10.") changes nothing.
  * A paragraph that is an "OTC vs Prescription" heading starts an "otc"
    Section of its own.
  * Any other numbered paragraph ("4. Fibrates", "2. NSAIDs - ...", whatever
    its style) is a drug-class heading and starts a Section.
  * Any other Heading 1/2 paragraph is also a drug-class heading.
  * Everything else (lettered sub-headings such as "A. Sucralfate", "Side
    Effects", Heading 3 labels, body lines, tables) stays inside the current
    Section.
  * A "Drug class" table with no heading before it (KB-02 has one, after an
    empty Heading 2) opens an "otc" Section whose name is supplied by us.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Row, Section, SectionKind, Table

_NUMBERING = re.compile(r"^\s*\d+\s*[.)]\s+")
_NUMBERED = re.compile(r"^\s*\d+\s*[.)]\s+\S")
# Single-line only: KB-03 has body paragraphs that open with the same words
# followed by a line break and the explanation; those are not headings.
_OTC_HEADING = re.compile(r"^(?:ملخص\s+)?OTC\s+(?:vs|ولا)\s+Prescription[^\n]*$", re.I)
_TOP_HEADING_STYLES = {"Heading 1", "Heading 2"}

# Name given to an OTC block that has no heading text in the source.
UNNAMED_OTC_SECTION = "OTC vs Prescription"


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
        self.section: str | None = None
        self.kind: SectionKind = "overview"
        self.heading_present = True
        self.lines: list[str] = []
        self.tables: list[Table] = []

    def build(self) -> Section | None:
        if self.kind == "overview" and not self.lines and not self.tables:
            return None
        return Section(
            kb_id=self.kb_id,
            title=self.title,
            section=self.section,
            kind=self.kind,
            lines=tuple(self.lines),
            tables=tuple(self.tables),
            heading_present=self.heading_present,
        )


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

    for block in blocks:
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
                flush()
                cur = _Builder(kb_id, cur.title)
                cur.section, cur.kind = text, "otc"
            elif _NUMBERED.match(text) or block.style in _TOP_HEADING_STYLES:
                flush()
                cur = _Builder(kb_id, cur.title)
                cur.section, cur.kind = name, "class"
            else:
                cur.lines.append(block.text)
        else:
            if cur is None:
                raise SplitError(f"{kb_id}: table before the first disease heading")
            header, *data = block.rows
            if cur.kind != "otc" and header[0].strip().lower() == "drug class":
                flush()
                cur = _Builder(kb_id, cur.title)
                cur.section, cur.kind = UNNAMED_OTC_SECTION, "otc"
                cur.heading_present = False
            cur.tables.append(
                Table(
                    header=Row.from_cells(header),
                    rows=tuple(Row.from_cells(r) for r in data),
                )
            )
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
