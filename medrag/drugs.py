"""Trade Name <-> Active Ingredient index, derived from the KB's products tables (nothing is hard-coded).

Every products table (tables.py classifies it from its header row) has a Trade Name column and an Active
Ingredient column. This module reads them once and answers:

  - which Active Ingredient(s) a Trade Name has, and which Trade Names an Active Ingredient has
    (a question that names one is also searched under the other: see hybrid.py);
  - which Trade Names are Alternatives of a Trade Name: the ones with the same Active Ingredient(s)
    (GLOSSARY: a different drug of the same Drug Class is not an Alternative);
  - which rows of which Sections hold them, so an Alternative question can be given every Section that
    lists them (`alternative_views`).

Normalisation (so that two spellings of one thing compare equal, never to invent a fact):
  - a Trade Name cell may hold several names ("Humira, Amgevita, Hyrimoz", "Constella / Linzess");
  - a parenthetical and a Strength are dropped from an Active Ingredient ("Ibuprofen (as ibuprofen arginine)",
    "Paracetamol 500 mg + Caffeine 65 mg"); "+" separates the components of a combination, so a
    combination is never an Alternative of one of its parts;
  - a Trade Name written in Arabic letters ("بعض مستحضرات Ibuprofen topical"), or equal to the ingredient it
    lists ("Oseltamivir (generic)"), is not a brand and is not indexed.
A salt written differently ("Diclofenac sodium" / "Diclofenac potassium") is a different Active Ingredient
here: the index only claims what the KB spells the same.
"""
from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from .models import Section, Table
from .rules import normalize

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_ARABIC = re.compile(r"[؀-ۿ]")
_PAREN = re.compile(r"\([^)]*\)")
_STRENGTH = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|ml|iu|units?|%)(?:/\w+)?|\b\d+(?:[.,]\d+)?\s*%|\bu-?\d+\b", re.I)
_SPLIT_NAMES = re.compile(r"\s*[,/;،]\s*")


def tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def product_cells(sections) -> Iterator[tuple[Section, Table, tuple[str, ...], str, str]]:
    """(section, table, row cells, trade-name cell, active-ingredient cell) for every row of every products table."""
    for sec in sections:
        for table in sec.tables:
            if table.kind != "products":
                continue
            t_col, a_col = table.trade_name_col, table.active_ingredient_col
            if t_col is None or a_col is None:
                continue
            for row in table.rows:
                if t_col < len(row.cells) and a_col < len(row.cells):
                    yield sec, table, row.cells, row.cells[t_col], row.cells[a_col]


def trade_names(cell: str) -> list[str]:
    """The brands a Trade Name cell holds (usually one). Arabic-letter text and bare ingredient words are not brands."""
    out = []
    for part in _SPLIT_NAMES.split(_PAREN.sub(" ", cell)):
        part = part.strip()
        if part and not _ARABIC.search(part) and tokens(part):
            out.append(part)
    return out


def ingredient_set(cell: str) -> frozenset[str]:
    """The Active Ingredient(s) of a cell: normalised component names, strengths and parentheticals dropped."""
    comps = set()
    for part in cell.split("+"):
        part = _STRENGTH.sub(" ", _PAREN.sub(" ", part))
        if _ARABIC.search(part):  # an Arabic remark, not an ingredient name
            part = _ARABIC.sub(" ", part)
        name = " ".join(tokens(part))
        if name:
            comps.add(name)
    return frozenset(comps)


@dataclass(frozen=True)
class Mentions:
    """What a question names: Trade Names (normalised) and Active Ingredient sets (see DrugIndex.mentions)."""

    trade: tuple[str, ...] = ()
    ingredients: tuple[frozenset[str], ...] = ()  # sets named directly or through a Trade Name


class DrugIndex:
    def __init__(self, sections: list[Section] | tuple[Section, ...]):
        self._display: dict[str, str] = {}  # normalised trade name -> first spelling seen
        self._sets_of: dict[str, set[frozenset[str]]] = defaultdict(set)  # trade name -> its ingredient sets
        self._names_of: dict[frozenset[str], dict[str, None]] = defaultdict(dict)  # ingredient set -> trade names (ordered)
        # (section_id, ingredient set) -> {trade name: [(table index, row index)]}
        self._where: dict[tuple[str, frozenset[str]], dict[str, list[tuple[int, int]]]] = defaultdict(dict)
        self._sections = {s.section_id: s for s in sections}
        order = {s.section_id: n for n, s in enumerate(sections)}
        for sec, table, cells, t_cell, a_cell in product_cells(sections):
            ing = ingredient_set(a_cell)
            if not ing:
                continue
            t_idx = [i for i, t in enumerate(sec.tables) if t is table][0]
            r_idx = [i for i, r in enumerate(table.rows) if r.cells == cells][0]
            for name in trade_names(t_cell):
                key = " ".join(tokens(name))
                if frozenset((key,)) == ing:  # "Oseltamivir (generic)": the ingredient itself, not a brand
                    continue
                self._display.setdefault(key, name)
                self._sets_of[key].add(ing)
                self._names_of[ing][key] = None
                self._where[(sec.section_id, ing)].setdefault(key, []).append((t_idx, r_idx))
        self._order = order
        # n-gram lookups for question matching
        self._trade_grams = {tuple(k.split()) for k in self._display}
        self._first = defaultdict(set)  # first word of a multi-word trade name -> names ("panadol" -> "panadol extra")
        for k in self._display:
            w = k.split()
            if len(w) > 1 and len(w[0]) >= 4:
                self._first[w[0]].add(k)
        self._comp_exact = defaultdict(set)  # component name -> the single-component sets of exactly it
        self._comp_alias = defaultdict(set)  # first word of a component -> single-component sets ("diclofenac")
        self._comp_grams: set[tuple[str, ...]] = set()
        self._combo_sets = {s for s in self._names_of if len(s) > 1}
        for s in self._names_of:
            for c in s:
                self._comp_grams.add(tuple(c.split()))
                if len(s) == 1:
                    self._comp_exact[c].add(s)
                    first = c.split()[0]
                    if len(first) >= 5 and first != c:
                        self._comp_alias[first].add(s)
                        self._comp_grams.add((first,))
        self.names = frozenset(self._display)

    # ---------------------------------------------------------------- lookups
    def display(self, key: str) -> str:
        return self._display.get(key, key)

    def ingredients_of(self, key: str) -> list[frozenset[str]]:
        return sorted(self._sets_of.get(key, ()), key=sorted)

    def trade_names_of(self, ing: frozenset[str]) -> list[str]:
        return list(self._names_of.get(ing, ()))

    def ingredient_text(self, ing: frozenset[str]) -> str:
        return " + ".join(sorted(ing))

    # ---------------------------------------------------------------- questions
    def mentions(self, question: str) -> Mentions:
        toks = tokens(question)
        trade: list[str] = []
        used = [False] * len(toks)
        for n in (4, 3, 2, 1):
            for i in range(len(toks) - n + 1):
                gram = tuple(toks[i:i + n])
                if gram in self._trade_grams and not any(used[i:i + n]):
                    key = " ".join(gram)
                    if key not in trade:
                        trade.append(key)
                    used[i:i + n] = [True] * n
        for i, t in enumerate(toks):  # "Panadol" is a family of products if no product is exactly that
            if not used[i] and t in self._first:
                trade.extend(k for k in sorted(self._first[t]) if k not in trade)
        sets: list[frozenset[str]] = []
        for key in trade:
            sets.extend(s for s in self.ingredients_of(key) if s not in sets)
        named_comps: list[str] = []
        for n in (3, 2, 1):
            for i in range(len(toks) - n + 1):
                gram = tuple(toks[i:i + n])
                if gram in self._comp_grams and not any(used[i:i + n]):
                    named_comps.append(" ".join(gram))
                    used[i:i + n] = [True] * n
        for c in named_comps:
            found = self._comp_exact.get(c) or self._comp_alias.get(c, ())  # an exact name beats a first-word match
            sets.extend(s for s in sorted(found, key=sorted) if s not in sets)
        if len(named_comps) > 1:  # "Paracetamol and Caffeine" may name a combination
            want = set(named_comps)
            sets.extend(s for s in sorted(self._combo_sets, key=sorted) if s <= want and s not in sets)
        return Mentions(tuple(trade), tuple(sets))

    # ---------------------------------------------------------------- Alternatives
    def alternatives(self, ings) -> dict[frozenset[str], list[str]]:
        return {i: self.trade_names_of(i) for i in ings}

    def alternative_views(self, ings, prefer: list[str] | None = None, max_sections: int = 3) -> list[Section]:
        """Sections that between them list every Trade Name of the given Active Ingredient set(s), each cut down
        to the product rows of those ingredients (same section_id, rows verbatim). Chosen greedily: the Section
        that adds the most Trade Names first (`prefer`, a list of section_ids, breaks ties), so the Section
        holding the longest list comes first and the Sections that only repeat names are left out."""
        ings = list(ings)
        cover: dict[str, dict[str, list[tuple[int, int]]]] = {}  # section_id -> {name key -> rows}
        for (sid, ing), names in self._where.items():
            if ing in ings:
                for key, rows in names.items():
                    cover.setdefault(sid, {}).setdefault(key, []).extend(rows)
        want = {(i, k) for i in ings for k in self._names_of.get(i, ())}
        pref = {sid: n for n, sid in enumerate(prefer or [])}
        chosen: list[str] = []
        seen: set[tuple[frozenset[str], str]] = set()
        while len(chosen) < max_sections:
            def gain(sid: str) -> int:
                return sum(1 for (sec_id, ing), names in self._where.items() if sec_id == sid and ing in ings
                           for k in names if (ing, k) not in seen)
            best = max((s for s in cover if s not in chosen), default=None,
                       key=lambda s: (gain(s), -pref.get(s, 10**6), -self._order[s]))
            if best is None or gain(best) == 0:
                break
            chosen.append(best)
            for (sec_id, ing), names in self._where.items():
                if sec_id == best and ing in ings:
                    seen.update((ing, k) for k in names)
            if want <= seen:
                break
        return [self._view(sid, {r for rows in cover[sid].values() for r in rows}) for sid in chosen]

    def _view(self, section_id: str, keep: set[tuple[int, int]]) -> Section:
        sec = self._sections[section_id]
        items = []
        for ti, table in enumerate(sec.tables):
            rows = tuple(r for ri, r in enumerate(table.rows) if (ti, ri) in keep)
            if rows:
                items.append(dataclasses.replace(table, rows=rows))
        return dataclasses.replace(sec, items=tuple(items))
