"""Retrieval interface and the lexical half of it: BM25 over Section.render() (hybrid.py fuses it with BGE-M3)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .models import Section
from .support import FRAME, stem, tokens as folded_tokens

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "the to what which who with you your".split()
)
STOP_WORDS = _STOP


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in _STOP]


def index_tokens(text: str) -> list[str]:
    """BM25's words: Arabic folded as everywhere else (alef, yaa, taa marbuta, harakat, tatweel) and stemmed (the article
    ال and the clitics, plural endings), so "الحموضة" and "حموضة" are one word; English lightly stemmed likewise."""
    return [stem(t) for t in folded_tokens(text) if t not in _STOP]


def query_tokens(text: str) -> list[str]:
    """The same words of a question, minus the frame words of the support check (support.FRAME: "which", "drugs",
    "ايه", "عندي", "bedail" ...): they say how to ask, not what is asked, and a word like انا or اخد must not
    outvote the topic."""
    return [t for t in index_tokens(text) if t not in _FRAME_STEMS]


_FRAME_STEMS = frozenset(stem(w) for w in FRAME)


class Retriever(Protocol):
    """What the pipeline needs from retrieval: the Sections for a question, best first, at most k.
    The call blocks (an embedding model runs on the CPU): callers run it off the event loop. A retriever
    that needs loading also has `ready` (bool), `model_id` and `load()` (see app.py)."""

    def search(self, query: str, k: int) -> list[Section]: ...


class LexicalRetriever:
    """BM25 (k1=1.5, b=0.75) over each Section's rendered text."""

    def __init__(self, sections: list[Section] | tuple[Section, ...], k1: float = 1.5, b: float = 0.75,
                 normalized: bool = True):
        """`normalized=False` is the plain tokenizer of the walking skeleton (a baseline for the recall script)."""
        self.sections = tuple(sections)
        self._k1, self._b = k1, b
        self._doc_tokens = index_tokens if normalized else tokenize
        self._query_tokens = query_tokens if normalized else tokenize
        self._tf = [Counter(self._doc_tokens(s.render())) for s in self.sections]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.sections)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        """BM25 score of every Section (0.0 when no query term occurs in it), in Section order."""
        q = [t for t in self._query_tokens(query) if t in self._idf]
        out = []
        for tf, length in zip(self._tf, self._len):
            score = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    norm = 1 - self._b + self._b * length / (self._avg or 1)
                    score += self._idf[t] * f * (self._k1 + 1) / (f + self._k1 * norm)
            out.append(score)
        return out

    def search(self, query: str, k: int) -> list[Section]:
        """Top-k Sections with a non-zero score, best first (empty if no term matches)."""
        scored = [(sc, i) for i, sc in enumerate(self.scores(query)) if sc > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.sections[i] for _, i in scored[:k]]
