"""Consonant skeletons: a way to compare a word written in Arabic letters, a Latin name and a Franco-Arabic word.

Arabic, English and Franco-Arabic spell the same sounds with different letters and keep or drop the vowels
differently ("بروفين" / "Brufen", "كتافلام" / "Cataflam", "سخونية" / "so5oneya"). Both sides are reduced to the same
coarse alphabet: the consonants only, each sound class one letter (b p -> b, f v -> f, k q c g ... see the tables),
vowels and the semi-vowel letters (ا و ي ع ء, a e i o u y w) dropped, repeated letters collapsed. Two words with the same
skeleton, or skeletons one edit apart when they are long enough, are taken to be one word.

The skeleton is only ever a way to compare two spellings. It carries no meaning, so it can match a name to itself
in another script (used for Arabic-letter Trade Names, arabicnames.py) and check that a translation of a word is
the same word and not another (support.SupportCheck.rewrite_ok), but it never produces a word the user did not write.
"""
from __future__ import annotations

import re

from .textnorm import normalize

_AR = {
    "ب": "b", "پ": "b", "ت": "t", "ث": "s", "ج": "g", "چ": "s", "ح": "h", "خ": "k", "د": "d", "ذ": "s", "ر": "r",
    "ز": "s", "س": "s", "ش": "s", "ص": "s", "ض": "d", "ط": "t", "ظ": "s", "غ": "g", "ف": "f", "ڤ": "f", "ق": "k",
    "ك": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ه": "h",
}
_LATIN = {"b": "b", "p": "b", "t": "t", "d": "d", "g": "g", "j": "g", "k": "k", "q": "k", "f": "f", "v": "f", "s": "s",
          "z": "s", "l": "l", "m": "m", "n": "n", "r": "r", "h": "h"}
_FRANCO_DIGITS = {"7": "h", "5": "k", "9": "k", "6": "t", "8": "g"}  # 2 and 3 (ء ع) are vowel-like and dropped
_DIGRAPHS = (("sh", "s"), ("ch", "s"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("ph", "f"), ("ck", "k"), ("qu", "k"))
_ARABIC = re.compile("[؀-ۿ]")


def _collapse(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)


def _latin(word: str, ayn: bool = False) -> str:
    w = word
    for a, b in _DIGRAPHS:
        w = w.replace(a, b.upper())  # upper case marks a finished sound
    w = w.replace("x", "KS")
    out = []
    for i, c in enumerate(w):
        if c.isupper():
            out.append(c.lower())
        elif c == "c":
            out.append("s" if w[i + 1:i + 2] in ("e", "i", "y") else "k")
        elif c in _LATIN:
            out.append(_LATIN[c])
        elif c in _FRANCO_DIGITS:
            out.append(_FRANCO_DIGITS[c])
        elif ayn and c in "23":
            out.append(c)
    return _collapse("".join(out))


def _arabic(word: str, ayn: bool = False) -> str:
    w = word[:-1] if word.endswith("ه") and len(word) > 3 else word  # a final ة / ه is a vowel ending
    table = {**_AR, "ع": "3", "ء": "2"} if ayn else _AR
    return _collapse("".join(table.get(c, "") for c in w))


def skeleton(text: str, ayn: bool = False) -> str:
    """The consonant skeleton of each word of `text`, words separated by one space (Arabic letters, Latin letters
    and Franco-Arabic digits are all understood). `ayn`: keep ع and ء (Franco 3 and 2) as letters, for comparing a
    Franco-Arabic word with its Arabic spelling ("3aly" ~ "عالي")."""
    words = []
    for w in re.findall(r"[^\W_]+", normalize(text)):
        words.append(_arabic(w, ayn) if _ARABIC.search(w) else _latin(w, ayn))
    return " ".join(w for w in words if w)


_AR_FINE = {"ث": "s", "ذ": "z", "ز": "z", "ش": "s", "ص": "s", "ض": "d", "ظ": "z", "ق": "k", "ج": "g", "غ": "g", "خ": "k", "ح": "h",
            "چ": "s", "گ": "g", "ڤ": "v", "پ": "p"}


def fine(text: str) -> str:
    """A finer skeleton for telling apart two names that share a coarse one (Brufen / Pirafene: b and p are one
    class above, and Arabic has no p): the same reduction, but b p f v z s j g stay what they are."""
    out = []
    for w in re.findall(r"[^\W_]+", normalize(text)):
        if _ARABIC.search(w):
            w = w[:-1] if w.endswith("ه") and len(w) > 3 else w
            out.append(_collapse("".join(_AR_FINE.get(c, _AR.get(c, "")) for c in w)))
        else:
            for a, b in _DIGRAPHS:
                w = w.replace(a, b.upper())
            w = w.replace("x", "KS")
            letters = []
            for i, c in enumerate(w):
                if c.isupper():
                    letters.append(c.lower())
                elif c == "c":
                    letters.append("s" if w[i + 1:i + 2] in ("e", "i", "y") else "k")
                elif c == "q":
                    letters.append("k")
                elif c in "bpdtgjkfvszlmnrh":
                    letters.append(c)
            out.append(_collapse("".join(letters)))
    return " ".join(x for x in out if x)


def distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def allowed_distance(sk: str) -> int:
    """How far apart two skeletons of this length may be and still be one word: none for short ones."""
    n = len(sk.replace(" ", ""))
    return 0 if n < 6 else 1 if n < 9 else 2


def same_word(a: str, b: str) -> bool:
    """True when two words (any script) have the same skeleton, or skeletons within `allowed_distance`."""
    for ayn in (False, True):
        sa, sb = skeleton(a, ayn), skeleton(b, ayn)
        if len(sa) >= 2 and len(sb) >= 2 and distance(sa, sb) <= min(allowed_distance(sa), allowed_distance(sb)):
            return True
    return False
