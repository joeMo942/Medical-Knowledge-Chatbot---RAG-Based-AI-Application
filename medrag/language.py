"""Answer Language of a message, decided once, from the current message alone (GLOSSARY: Answer Language).

  ar     the message has Arabic script (Modern Standard or Egyptian dialect); drug names in Latin letters do not change it
  mixed  Franco-Arabic only: Arabic written in Latin letters, recognised by its digit-letters (3 = ع, 7 = ح, 5 = خ,
         2 = ء, 9 = ق, 6 = ط, 8 = غ) inside a word ("3andy", "a5od", "bta3t") or by two or more unambiguous Franco
         words ("el", "eh", "emta", "bedail" ...). A message that is English with a drug name is never mixed
  en     everything else

The label is decided here (and refined once by `refine`, see below), then handed to the answer call as the language to
write in, so the reply text and the label always agree; the routing call's own language guess never overrides it.
"""
from __future__ import annotations

import re

from .textnorm import has_arabic, normalize

# A word with one Franco digit-letter followed by letters: "3andy", "a5od", "so5oneya", "7aga", "bta3t". A digit that ends
# a word ("omega3", "type2") or a unit or ordinal ("5mcg", "2nd", "3ml") is not one.
_NOT_FRANCO = frozenset("mg mgs mcg mgm ml gm kg iu nd rd th st mmol".split())
_WORD = re.compile(r"[a-z0-9]+")
# Words that are Franco-Arabic and not English; two of them make a message mixed.
FRANCO_WORDS = frozenset("""
el ellu elly lel lil eh eih ehh ey ezay ezzay leh lih emta imta feen fein meen mesh msh
ana enta enti ahna da dah deh di dee bta3 bta3t bta3et beta3 beta3t beta3et ba2a
dawa dawaa adwya ad2ya dawaya 7aga 7aja haga shwaya kteer kitir awy awi gedan ma3a 3ashan 3lshan
bedail bedayel bdayel bdail bdayl tegareya tegarya tigareya mamnoo3 mamno3 mawane3 mawan3
a3rad a3ra2 gaanbeya ganbeya maada mada fa3ala fa3ela el3ada 3ayez 3ayza 3awez 3awza
momken lazem fe fi
""".split())
_STRONG = frozenset(FRANCO_WORDS - {"fe", "fi", "da", "di", "ana", "ma3a", "mada", "momken", "lazem", "dawa", "ey"})


_ENGLISH_WITH_DIGIT = frozenset("omega3 omega6 type2 type3 pcsk9 sglt2 sglt1 pde5 sars2 cov2 ldl3 hdl2 apob100".split())


def _is_franco_digit_word(w: str) -> bool:
    """A Franco digit-letter in the word: inside or in front of at least two letters ("3andy", "a5od", "so5oneya",
    "3aree2"), or ending a word of at least four letters ("waga3", "mamnoo3"). Not a unit, an ordinal or a
    name with a digit ("5mcg", "2nd", "omega3", "cyp3a4", "h2")."""
    if w in _ENGLISH_WITH_DIGIT or w.isdigit():
        return False
    trailing = re.fullmatch(r"[a-z]{4,12}[235-9]", w)
    if trailing:
        return True
    m = re.fullmatch(r"([a-z]{0,12})([235-9])([a-z]{2,12})[0-9]?", w)
    return bool(m) and m.group(3) not in _NOT_FRANCO


def franco_marks(message: str) -> tuple[int, int]:
    """(digit-letter words, Franco lexicon words) in a Latin-letter message."""
    t = normalize(message)
    digits = sum(1 for w in set(_WORD.findall(t)) if _is_franco_digit_word(w))
    words = sum(1 for w in set(_WORD.findall(t)) if w in _STRONG)
    return digits, words


def detect_language(message: str) -> str:
    """'ar' for Arabic script, 'mixed' for Franco-Arabic, else 'en'."""
    if has_arabic(message):
        return "ar"
    digits, words = franco_marks(message)
    return "mixed" if digits or words >= 2 else "en"


def refine(base: str, message: str, routed: str | None) -> str:
    """The Answer Language once the routing call has run: the heuristic's `en` becomes `mixed` only when the
    router also says Franco-Arabic AND the message carries some Franco mark of its own (one lexicon word). The
    router alone never decides, so an English question with a drug name is never relabelled."""
    if base == "en" and routed == "mixed":
        digits, words = franco_marks(message)
        if digits or words:
            return "mixed"
    return base
