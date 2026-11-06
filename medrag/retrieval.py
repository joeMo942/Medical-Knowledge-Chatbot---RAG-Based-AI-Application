"""Placeholder retrieval: BM25 over Section.render(). Hybrid BGE-M3 search replaces it in #6."""
from __future__ import annotations

import math
import re
from collections import Counter

from .models import Section

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "the to what which who with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in _STOP]


class LexicalRetriever:
    """BM25 (k1=1.5, b=0.75) over each Section's rendered text."""

    def __init__(self, sections: list[Section] | tuple[Section, ...], k1: float = 1.5, b: float = 0.75):
        self.sections = tuple(sections)
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(s.render())) for s in self.sections]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.sections)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int) -> list[Section]:
        """Top-k Sections with a non-zero score, best first (empty if no term matches)."""
        q = [t for t in tokenize(query) if t in self._idf]
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    norm = 1 - self._b + self._b * self._len[i] / (self._avg or 1)
                    score += self._idf[t] * f * (self._k1 + 1) / (f + self._k1 * norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.sections[i] for _, i in scored[:k]]
