"""Data model for the Knowledge Base: KB Documents made of Sections.

Terms follow GLOSSARY.md. All text is stored exactly as it appears in the
source document; nothing here cleans, translates or invents content.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Joins the non-empty cells of a table row into the snippet a Citation quotes,
# matching the API contract's example "Controloc – Pantoprazole" (EN DASH).
CELL_SEPARATOR = " – "

# Joins title and section into the unique path, e.g. "Asthma › SABA".
PATH_SEPARATOR = " › "

SectionKind = Literal["class", "otc", "overview"]
TableKind = Literal["products", "side_effects", "otc", "other"]


@dataclass(frozen=True)
class Row:
    """One table row. `cells` are the raw cells (empty ones kept) and are the
    source of truth; `text` is the snippet a Citation would quote, made of the
    non-empty cells only.

    Cells can themselves contain " – " (e.g. "ORS – Pharopharma"), so
    downstream code must use `cells`, never split `text`.
    """

    cells: tuple[str, ...]
    text: str

    @classmethod
    def from_cells(cls, cells: tuple[str, ...] | list[str]) -> "Row":
        cells = tuple(cells)
        return cls(cells=cells, text=CELL_SEPARATOR.join(c for c in cells if c))

    def to_dict(self) -> dict:
        return {"cells": list(self.cells), "text": self.text}


@dataclass(frozen=True)
class Table:
    """A table kept whole: its header row plus every data row, in order.

    kind is classified from the header row only; "other" when it cannot be
    classified. For "products" tables, `trade_name_col` and
    `active_ingredient_col` give the column of each (two tables in the KB list
    the Active Ingredient first). Extra columns stay in `cells`.
    """

    header: Row
    rows: tuple[Row, ...]
    kind: TableKind = "other"
    trade_name_col: int | None = None
    active_ingredient_col: int | None = None

    def to_dict(self) -> dict:
        return {
            "type": "table",
            "kind": self.kind,
            "trade_name_col": self.trade_name_col,
            "active_ingredient_col": self.active_ingredient_col,
            "header": self.header.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
        }


@dataclass(frozen=True)
class Line:
    """One body paragraph, verbatim."""

    text: str

    def to_dict(self) -> dict:
        return {"type": "line", "text": self.text}


Item = Line | Table


@dataclass(frozen=True)
class Section:
    """A drug-class block under a disease heading within one KB Document.

    `items` is the source of truth: the lines and tables of the block in
    reading order, so a table can be tied to the sub-label ("D. Lincosamides")
    right before it. `lines`, `tables` and `rows` are derived views.

    `section` is what a Citation shows and is always a real string from the
    KB: the block's own heading (leading numbering stripped), or, when it has
    none, the nearest enclosing heading, the disease title. `heading_present`
    says which. A block has no heading of its own when it is the overview text
    under a disease heading (kind "overview") or a table with no heading text
    in the source (kind "otc").

    kind:
      "class"    a drug-class heading (the normal case)
      "otc"      an "OTC vs Prescription" block, its own Section
      "overview" text under the disease heading before its first drug class
    """

    kb_id: str
    title: str
    section: str
    kind: SectionKind
    items: tuple[Item, ...]
    heading_present: bool = True

    @property
    def label(self) -> str:
        """Heading for human-readable output. When the block has no heading
        in the source this is a descriptive label that is NOT from the
        source, marked with square brackets."""
        if self.heading_present:
            return self.section
        return "[overview, no heading in source]" if self.kind == "overview" else (
            "[table with no heading in source]"
        )

    @property
    def path(self) -> str:
        """Unique within a KB Document: disease › drug class (or disease ›
        bracketed label for a block without its own heading)."""
        if self.heading_present:
            return PATH_SEPARATOR.join((self.title, self.section))
        return PATH_SEPARATOR.join((self.title, self.label))

    @property
    def section_id(self) -> str:
        """Unique across the Knowledge Base."""
        if self.heading_present:
            return f"{self.kb_id}{PATH_SEPARATOR}{self.path}"
        # No bracketed label: only the kind, which keeps IDs unique.
        return f"{self.kb_id}{PATH_SEPARATOR}{self.title}{PATH_SEPARATOR}#{self.kind}"

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(i.text for i in self.items if isinstance(i, Line))

    @property
    def tables(self) -> tuple[Table, ...]:
        return tuple(i for i in self.items if isinstance(i, Table))

    @property
    def rows(self) -> tuple[Row, ...]:
        """Every data row of every table (header rows excluded)."""
        return tuple(r for t in self.tables for r in t.rows)

    @property
    def product_rows(self) -> tuple[Row, ...]:
        return tuple(r for t in self.tables if t.kind == "products" for r in t.rows)

    @property
    def other_rows(self) -> tuple[Row, ...]:
        return tuple(r for t in self.tables if t.kind != "products" for r in t.rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def render(self) -> str:
        """The Section as text in reading order: heading line, then each line
        and each table (header row, then data rows) as it appears. Only text
        from the KB appears: a block without a heading of its own starts with
        its disease title, never with the bracketed label."""
        out = [PATH_SEPARATOR.join((self.title, self.section))
               if self.heading_present else self.title]
        for item in self.items:
            if isinstance(item, Line):
                out.append(item.text)
            else:
                out.append(item.header.text)
                out.extend(r.text for r in item.rows)
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "kb_id": self.kb_id,
            "title": self.title,
            "section": self.section,
            "kind": self.kind,
            "heading_present": self.heading_present,
            "path": self.path,
            "section_id": self.section_id,
            "row_count": self.row_count,
            "items": [i.to_dict() for i in self.items],
        }


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
