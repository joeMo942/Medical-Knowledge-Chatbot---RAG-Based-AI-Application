"""Data model for the Knowledge Base: KB Documents made of Sections.

Terms follow GLOSSARY.md. All text is stored exactly as it appears in the
source document; nothing here cleans, translates or invents content.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

# Joins the cells of a table row into the verbatim snippet, matching the API
# contract's Citation example "Controloc – Pantoprazole" (EN DASH, not hyphen).
CELL_SEPARATOR = " – "

# Joins title and section into the unique path, e.g. "Asthma › SABA".
PATH_SEPARATOR = " › "

SectionKind = Literal["class", "otc", "overview"]


@dataclass(frozen=True)
class Row:
    """One table row. `cells` are raw (empty cells kept); `text` is the
    snippet a Citation would quote."""

    cells: tuple[str, ...]
    text: str

    @classmethod
    def from_cells(cls, cells: tuple[str, ...] | list[str]) -> "Row":
        cells = tuple(cells)
        return cls(cells=cells, text=CELL_SEPARATOR.join(cells))


@dataclass(frozen=True)
class Table:
    """A table kept whole: its header row plus every data row, in order."""

    header: Row
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Section:
    """A drug-class block under a disease heading within one KB Document.

    kind:
      "class"    a drug-class heading (the normal case)
      "otc"      an "OTC vs Prescription" block, its own Section
      "overview" text that sits under the disease heading before its first
                 drug-class heading (`section` is None)
    `heading_present` is False only when the block has no heading text in
    the source and `section` was supplied by the splitter.
    """

    kb_id: str
    title: str
    section: str | None
    kind: SectionKind
    lines: tuple[str, ...]
    tables: tuple[Table, ...]
    heading_present: bool = True

    @property
    def path(self) -> str:
        """Unique within a KB Document: disease › drug class."""
        return PATH_SEPARATOR.join(
            p for p in (self.title, self.section) if p is not None
        )

    @property
    def section_id(self) -> str:
        """Unique across the Knowledge Base."""
        return f"{self.kb_id}{PATH_SEPARATOR}{self.path}"

    @property
    def row_count(self) -> int:
        """Data rows across all tables (header rows excluded)."""
        return sum(len(t.rows) for t in self.tables)

    @property
    def rows(self) -> tuple[Row, ...]:
        return tuple(r for t in self.tables for r in t.rows)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = self.path
        d["section_id"] = self.section_id
        d["row_count"] = self.row_count
        return d


@dataclass(frozen=True)
class KBDocument:
    kb_id: str
    file: str
    diseases: tuple[str, ...]  # disease titles, numbering stripped
    sections: tuple[Section, ...]

    def to_dict(self) -> dict:
        return {
            "kb_id": self.kb_id,
            "file": self.file,
            "diseases": list(self.diseases),
            "sections": [s.to_dict() for s in self.sections],
        }
