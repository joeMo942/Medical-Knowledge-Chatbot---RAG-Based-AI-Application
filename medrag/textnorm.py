"""The one text normaliser, shared by every place that compares words: the rules, the support check, BM25, the
drug-name index, the language decision and the Arabic-brand matcher. Questions and Sections go through the same
function, so a spelling variant never decides whether two words match.

  - NFKC: Arabic presentation forms (U+FB50-FDFF, U+FE70-FEFF: what text copied from a PDF extracts as) become the
    plain letters; full-width and compatibility characters likewise;
  - lower case; quotes unified; whitespace collapsed;
  - alef variants (أ إ آ ٱ) -> ا; alef maqsura ى and Persian ی -> ي; taa marbuta ة -> ه; hamza carriers ؤ -> و, ئ -> ي;
    Persian kaf ک -> ك;
  - harakat, dagger alef and tatweel dropped;
  - Arabic-Indic (٠-٩) and Persian (۰-۹) digits -> 0-9.
"""
from __future__ import annotations

import re
import unicodedata

_AR_DIACRITICS = re.compile("[ً-ٰٟـ]")  # harakat, dagger alef, tatweel
_AR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ک": "ك",
    **{chr(0x660 + i): str(i) for i in range(10)},  # Arabic-Indic digits
    **{chr(0x6F0 + i): str(i) for i in range(10)},  # Persian digits
})
# Arabic script blocks, including the presentation forms (detected as Arabic even before NFKC).
ARABIC_LETTER = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def normalize(text: str) -> str:
    """Lower-case, NFKC, fold Arabic letter variants, drop harakat and tatweel, unify quotes and spaces."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _AR_DIACRITICS.sub("", t).translate(_AR_MAP)
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    return re.sub(r"\s+", " ", t).strip()


def has_arabic(text: str) -> bool:
    return bool(ARABIC_LETTER.search(text))
