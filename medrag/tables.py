"""Classify a table from its header row alone (never from its content).

Anything the rules below do not recognise is "other"; nothing is guessed.
"""
from __future__ import annotations

import re

from .models import Row, Table

_TRADE = re.compile(r"trade names?|brand name|الاسم التجاري", re.I)
_ACTIVE = re.compile(r"active ingredients?|\bgeneric\b", re.I)
_SIDE_EFFECTS = re.compile(r"side effects?", re.I)
_OTC = re.compile(r"^drug class\b", re.I)


def _first(pattern: re.Pattern, header: tuple[str, ...], skip: int | None = None) -> int | None:
    for i, cell in enumerate(header):
        if i != skip and pattern.search(cell.strip()):
            return i
    return None


def classify(header: tuple[str, ...]):
    """Return (kind, trade_name_col, active_ingredient_col)."""
    if _first(_SIDE_EFFECTS, header) is not None:
        return "side_effects", None, None
    trade = _first(_TRADE, header)
    active = _first(_ACTIVE, header, skip=trade)
    if trade is not None and active is not None:
        return "products", trade, active
    if _OTC.search(header[0].strip()):
        return "otc", None, None
    return "other", None, None


def build_table(rows: tuple[tuple[str, ...], ...]) -> Table:
    header, *data = rows
    kind, trade, active = classify(header)
    return Table(
        header=Row.from_cells(header),
        rows=tuple(Row.from_cells(r) for r in data),
        kind=kind,
        trade_name_col=trade,
        active_ingredient_col=active,
    )
