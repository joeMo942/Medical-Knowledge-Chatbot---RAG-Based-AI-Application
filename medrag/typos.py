"""Typo tolerance ("Did you mean"): a question word that is not in the Knowledge Base but is one edit away from exactly one
KB word is corrected before search and the support check ("headace" -> "headache", "Brufin" -> "Brufen").

The vocabulary is the KB's own: every word of every Section's text (Trade Names, Active Ingredients, condition names,
side effects ...), nothing is listed by hand. A word is a candidate for correction only when
  - it is written in Latin letters and has five letters or more,
  - it is not a KB word, nor another form of one (an exact word of the KB's text, the same stem, an English ending added or
    taken off: "headaches" is not a typo of "headache"),
  - it is not a frame, aspect or Alternatives word, not an everyday English word (COMMON_ENGLISH: "barely" is not a typo of
    the KB's "rarely"), not a number, not a Franco-Arabic word (a Franco-Arabic message is never corrected),
Its corrections are the KB's DRUG names (Trade Names and Active Ingredients) and CONDITION words (the words of the Section titles
and headings: headache, hypertension, pharyngitis ...) that are one Damerau edit away (an insertion, a deletion, a substitution,
or two neighbouring letters swapped), for every length: two edits turned laryngitis into pharyngitis, hypothyroidism into
hyperthyroidism and urethritis into arthritis. Any other KB word is not a candidate (sleepy was turned into "Sleep", ankle into
"angle"), and a five-letter candidate is only reached by a substitution or a swap, not by adding or dropping a letter (Rhinex is
not a typo of Rinex). The word is corrected ONLY when the candidates, counted after stemming, are exactly one: an ambiguous word
is left alone. A word that is itself a KB drug name is known, so it is never turned into another drug.

Arabic-script words are NOT corrected. Tried on the eval sets, the same rule on the normalised form turned ordinary dialect
and Modern Standard words the KB happens not to contain into other words ("اليوم" -> "النوم"), so it was left out; the Arabic
Trade Name mapping (arabicnames.py) already covers the misspelled Arabic brands. Arabic-letter words never change here.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .language import FRANCO_WORDS, _is_franco_digit_word
from .drugs import product_cells
from .models import Section
from .support import ABSENT, ASPECT, FRAME, INDICATION, is_number, stem
from .textnorm import normalize

log = logging.getLogger("medrag")

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_LATIN_WORD = re.compile(r"[a-z]{5,}")
_ENDINGS = ("s", "es", "d", "ed", "ing", "ly", "al")
MIN_LETTERS = 5  # a question word shorter than this is never corrected

# Everyday English words that a question may use and the KB may not, each one edit from a KB word. A word on this list (or an
# inflection of one) is never read as a typo. Not exhaustive: a miss only shows as a visible "Did you mean".
COMMON_ENGLISH = frozenset("""
about above after again against almost alone along already ankle also always among another answer anyone anything around asked
barely because become before began behind being below better between beyond bigger bring brought called cannot cause change
check clear close cold comes could course daily doing doubt drink drank during early either else enough every except
fairly feels fewer finally first found fully given going great happen having hence higher inside instead itself keeps kinds
later least leave lesser likely little lower makes maybe means might mixed money month mostly never night nobody often once
order other ought outside overall people place point quite rather really recent right rough round rarely said same seems
several shall since small smaller someone sometimes soon speak stage start state still stuff sudden taken takes tells
thank tiredness their there these thing things think third those though three throughout times today together tough toward truly
twice under until upon usual usually various wants water wheel where which while whole whose within without women worse
would write wrong yield young yours sleepy watery lasik urinating
""".split())


@dataclass(frozen=True)
class Change:
    original: str  # as the user wrote it
    corrected: str  # the KB's spelling


@dataclass(frozen=True)
class Corrected:
    text: str  # the question with every correction applied (the same text when there is none)
    changes: tuple[Change, ...] = ()

    @property
    def words(self) -> list[str]:
        return [c.corrected for c in self.changes]


def damerau(a: str, b: str, limit: int) -> int:
    """Optimal-string-alignment distance of a and b (insert, delete, substitute, swap two neighbours); `limit` + 1 when it
    is more than `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] | None = None
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return min(prev[-1], limit + 1)


class TypoCorrector:
    """Built once from the KB's Sections."""

    def __init__(self, sections: Sequence[Section]):
        forms: dict[str, Counter[str]] = defaultdict(Counter)
        self._words: set[str] = set()  # every word of the KB's text, of any length
        for sec in sections:
            for w in _WORD.findall(sec.render()):
                norm = normalize(w)
                self._words.add(norm)
                if _LATIN_WORD.fullmatch(norm):
                    forms[norm][w] += 1
        self._stems = {stem(w) for w in self._words}
        # the lower-case form when the KB has one ("headache", not the heading's "Headache"), else the commonest ("Brufen")
        self._display = {norm: norm if norm in c else c.most_common(1)[0][0] for norm, c in forms.items()}
        self._count = {norm: sum(c.values()) for norm, c in forms.items()}
        # corrections: drug names (Trade Names, Active Ingredients) and condition words (Section titles and headings) only
        pool: set[str] = set()
        for _sec, _table, _cells, trade, active in product_cells(sections):
            for cell in (trade, active):
                pool.update(normalize(w) for w in _WORD.findall(cell))
        for sec in sections:
            for text in (sec.title, sec.section):
                pool.update(normalize(w) for w in _WORD.findall(text or ""))
        self._by_length: dict[int, list[str]] = defaultdict(list)
        self._pool = pool
        for norm in forms:
            self._by_length[len(norm)].append(norm)
        self._ignore = {normalize(w) for w in (*FRAME, *ASPECT, *ABSENT, *INDICATION, *FRANCO_WORDS, *COMMON_ENGLISH)}
        self._ignore_stems = {stem(w) for w in self._ignore if len(w) >= MIN_LETTERS}

    def _is_kb_form(self, norm: str) -> bool:
        """A KB word, or another form of one: the same stem, or the word with an English ending added or taken off
        ("headaches" / "headache", "capsule" / "capsules")."""
        if norm in self._words or stem(norm) in self._stems:
            return True
        bases = {norm[:-len(e)] for e in _ENDINGS if norm.endswith(e)}
        return any(b in self._words or b + "e" in self._words for b in bases) or any(norm + e in self._words for e in _ENDINGS)

    def _candidates(self, norm: str) -> dict[str, str]:
        """Stem -> the KB word of that stem (the most frequent form) for every KB word within the allowed distance."""
        found: dict[str, str] = {}
        for length in range(len(norm) - 1, len(norm) + 2):
            for cand in self._by_length.get(length, ()):
                if damerau(norm, cand, 1) > 1:
                    continue
                if len(cand) <= MIN_LETTERS and len(cand) != len(norm):
                    continue  # a five-letter word reached by adding or dropping a letter is too likely another word
                s = stem(cand)
                if s not in found or self._count[cand] > self._count[found[s]]:
                    found[s] = cand
        return found

    def correct_word(self, word: str) -> str | None:
        """The KB spelling `word` is a typo of, or None (not a typo, no candidate, or more than one)."""
        norm = normalize(word)
        if (not _LATIN_WORD.fullmatch(norm) or _is_franco_digit_word(norm) or is_number(norm) or norm in self._ignore
                or stem(norm) in self._ignore_stems or self._is_kb_form(norm)):
            return None
        found = self._candidates(norm)  # every KB word one edit away, so a near word outside the pool still makes it ambiguous
        if len(found) != 1:
            return None
        cand = next(iter(found.values()))
        return self._display[cand] if cand in self._pool else None

    def correct(self, text: str, language: str = "en") -> Corrected:
        """`text` with every typo corrected. A Franco-Arabic message (`language` "mixed") is never touched."""
        if language == "mixed":
            return Corrected(text)
        out: list[str] = []
        changes: list[Change] = []
        last = 0
        for m in _WORD.finditer(text):
            new = self.correct_word(m.group())
            if new is None or new == m.group():
                continue
            out.append(text[last:m.start()])
            out.append(new)
            last = m.end()
            changes.append(Change(m.group(), new))
        if not changes:
            return Corrected(text)
        out.append(text[last:])
        corrected = Corrected("".join(out), tuple(changes))
        log.info("typos corrected: %s", ", ".join(f"{c.original} -> {c.corrected}" for c in changes))
        return corrected


# The line an answered reply starts with when a word was corrected, in the reply's language (Arabic for `ar` and `mixed`).
DID_YOU_MEAN = {"en": "Did you mean: {words}?", "ar": "هل تقصد: {words}؟"}


def did_you_mean(words: Sequence[str], language: str) -> str:
    return DID_YOU_MEAN["en" if language == "en"
                        else "ar"].format(words=", ".join(dict.fromkeys(words)))
