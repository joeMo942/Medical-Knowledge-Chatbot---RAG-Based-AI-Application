"""Trade Names (and Active Ingredients) typed in Arabic letters, mapped to the KB's English spelling before search.

The KB writes every Trade Name and Active Ingredient in Latin letters, so "بروفين" never matches "Brufen" lexically.
The mapping is derived from the KB's own names (drugs.DrugIndex), nothing is listed by hand: each name's consonant
skeleton (translit.py) is indexed, and an Arabic word or two-word phrase whose skeleton matches one is replaced by that
name ("إيه المادة الفعالة في بروفين؟" -> "إيه المادة الفعالة في Brufen؟"). Only confident matches are taken:

  - an ordinary Arabic word is never a brand: a word of the KB's own Arabic text (all of it is read once, the index is built
    from the KB at start), a frame or aspect word, an Alternatives word;
  - the Arabic article and clitics (ال, ب, بال, ل, لل, و ...) are tried off the word, the unstripped word first;
  - an exact skeleton match is accepted (four consonants or more); a skeleton one edit away only when it has six or
    more (two edits: nine), and the nearest match is not shared with a different drug; a name shared by several Trade Names of ONE Active
    Ingredient maps to the nearest of them (the search adds the others); a skeleton shared by DIFFERENT drugs (Brufen /
    Pirafene) is settled by a finer skeleton (translit.fine: b and p, f and v stay apart), and left alone when still tied;
  - the longest phrase wins ("بنادول اكسترا" is Panadol Extra, not Panadol).

A word that matches nothing is left as it is: this never invents a drug, it only re-spells one the user named.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .drugs import DrugIndex, tokens
from .models import Section
from .support import ASPECT, FRAME, INDICATION
from .textnorm import normalize
from .translit import allowed_distance, distance, fine, skeleton

_ARABIC_RUN = re.compile(  # letters and marks, not the Arabic comma, semicolon, question mark or digits
    r"[\u0621-\u063A\u0640-\u065F\u066E-\u06D3\u06FA-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+")
_ARABIC_WORD = re.compile(r"[\u0600-\u06FF]")
_CLITICS = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف", "ك")
MIN_EXACT = 4  # consonants a skeleton needs to match at all (three: "قرص" is not Eucreas, "كثير" is not Cozaar)
MAX_WORDS = 3


def arabic_common_words(sections: list[Section] | tuple[Section, ...]) -> frozenset[str]:
    """The Arabic words that are never a brand: every Arabic word of the KB's own text, and the frame, aspect,
    Alternatives and indication words of the support check."""
    words = {t for sec in sections for t in tokens(sec.render()) if _ARABIC_WORD.search(t)}
    return frozenset(words | {w for w in (*FRAME, *ASPECT, *INDICATION) if _ARABIC_WORD.search(w)})


@dataclass(frozen=True)
class Replacement:
    original: str
    name: str


class ArabicNames:
    def __init__(self, drugs: DrugIndex, common_words: frozenset[str] = frozenset()):
        """`common_words`: normalised Arabic words that are never brands (the frame, aspect and Alternatives words)."""
        self._common = common_words
        # skeleton -> [(display name, identity)]; identity is what the name IS (its Active Ingredient component
        # names), so two names of one ingredient are not ambiguous with each other
        self._index: dict[str, list[tuple[str, frozenset[str]]]] = {}
        for key in sorted(drugs.names):
            comps = frozenset(c for ing in drugs.ingredients_of(key) for c in ing)
            self._add(drugs.display(key), comps or frozenset({key}))
        seen = set()
        for ing in sorted(drugs._names_of, key=sorted):  # every Active Ingredient component, as a name and by first word
            for c in ing:
                for name in {c, c.split()[0]}:
                    if name not in seen and name not in drugs.names:
                        seen.add(name)
                        self._add(name.title(), frozenset({c}))
        self._flat = [(sk, name, ident) for sk, entries in self._index.items() for name, ident in entries]

    def _add(self, name: str, ident: frozenset[str]) -> None:
        sk = skeleton(name)
        if len(sk.replace(" ", "")) >= MIN_EXACT:
            self._index.setdefault(sk, []).append((name, ident))

    def _variants(self, word: str) -> list[str]:
        out = [word]
        for p in _CLITICS:
            if word.startswith(p) and len(word) - len(p) >= 3 and word[len(p):] not in out:
                out.append(word[len(p):])
        return out

    def _lookup(self, words: list[str]) -> str | None:
        """The KB name for these normalised Arabic words (one word, or a phrase), or None."""
        if any(w in self._common for w in words):
            return None
        found: list[tuple[int, int, str, frozenset[str]]] = []  # (skeleton distance, fine distance, name, identity)
        for variant in self._variants(words[0]):  # the clitics of the first word only
            if variant in self._common:
                continue
            phrase = " ".join([variant, *words[1:]])
            sk = skeleton(phrase)
            if len(sk.replace(" ", "")) < MIN_EXACT:
                continue
            if sk in self._index:
                near = [(0, name, ident) for name, ident in self._index[sk]]
            else:
                budget = allowed_distance(sk)
                near = []
                if budget:
                    for esk, name, ident in self._flat:
                        if abs(len(esk) - len(sk)) <= budget and (d := distance(sk, esk)) <= budget:
                            near.append((d, name, ident))
            fp = fine(phrase)
            found += [(d, distance(fp, fine(name)), name, ident) for d, name, ident in near]
        if not found:
            return None
        found.sort(key=lambda f: f[:2])
        top = [f for f in found if f[:2] == found[0][:2]]
        return top[0][2] if len({f[3] for f in top}) == 1 else None

    def map(self, text: str) -> tuple[str, list[Replacement]]:
        """`text` with each Trade Name or Active Ingredient written in Arabic letters replaced by the KB's spelling, and
        the replacements made. Nothing else in the text changes."""
        runs = list(_ARABIC_RUN.finditer(text))
        out: list[str] = []
        done: list[Replacement] = []
        pos, i = 0, 0
        while i < len(runs):
            hit = None
            for n in range(min(MAX_WORDS, len(runs) - i), 0, -1):
                group = runs[i:i + n]
                if any(text[a.end():b.start()] != " " for a, b in zip(group, group[1:])):
                    continue
                name = self._lookup([normalize(m.group()) for m in group])
                if name:
                    hit = (n, name)
                    break
            if hit:
                n, name = hit
                start, end = runs[i].start(), runs[i + n - 1].end()
                out.append(text[pos:start] + name)
                done.append(Replacement(text[start:end], name))
                pos, i = end, i + n
            else:
                i += 1
        out.append(text[pos:])
        return "".join(out), done
