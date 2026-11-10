"""Hybrid search: BGE-M3 dense + BGE-M3 sparse + BM25, fused by reciprocal rank, over whole Sections.

Search units are Sections (never rows or chunks): a Citation needs the whole Section, and the model reads it whole.

Ranking. Each of the three scorers ranks every Section; a Section's score is the sum over scorers of
w / (RRF_K + rank) (reciprocal rank fusion: it needs no score calibration between a cosine, a token-weight sum and
BM25). A question that names a Trade Name is searched a second time with the Active Ingredient(s) added, and one
that names an Active Ingredient with its Trade Names added (drugs.py: derived from the products tables), so a
question written with the brand finds the Section that only says the generic and the other way round; the two
variants are separate lists in the fusion.

Thinning. The KB repeats the same table in several Documents (the NSAID products appear in KB-03, 04, 06 and
09). Walking down the fused list, a Section whose lines and rows are mostly already in the chosen ones is passed
over for the next one; passed-over Sections only fill what is left, so nothing is ever lost to thinning alone.

Alternatives. "Alternatives of Brufen" means every Trade Name with the same Active Ingredient. Those names are
spread over several Sections (some in one only), so plain top-k cannot reach them. Such a question is recognised
from its words and the drug it names, and gets the Sections from `DrugIndex.alternative_views`: the fewest Sections
that between them list every Trade Name of that Active Ingredient, each cut down to those product rows (rows
verbatim, same section_id), ahead of anything else. Answers and Citations still come only from retrieved text.

No match. When no query term occurs in any Section (BM25 is empty) and even the best dense similarity is below
`min_dense`, nothing is returned and the pipeline answers Out of Scope without an LLM call, as it did with BM25 alone.

Loading is separate from construction (`load()`): the model and the index take seconds to minutes, and /health
stays 503 until `ready`. The index (one dense and one sparse vector per Section) is cached on disk under a key
made of the KB's content hash, the model ID and revision, and a format version, so a changed Document, or a
different model, rebuilds it and an unchanged start reuses it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
from array import array
from collections.abc import Callable
from pathlib import Path

from .alternatives import ALT_WORDS, BRAND_WORDS
from .drugs import DrugIndex, Mentions, tokens
from .embedding import Embedder, Encoded, dense_score, sparse_score
from .models import Line, Section
from .retrieval import LexicalRetriever
from .textnorm import normalize
from .support import ABSENT, ASPECT, FRAME

log = logging.getLogger("medrag")

FORMAT_VERSION = 1
RRF_K = 60
WEIGHTS = (3.0, 1.0, 1.0)  # fusion weights of the dense, sparse and BM25 rankings: the meaning (dense) outvotes either lexical list alone but not both together, so a name the embedding cannot read (Arabic "فلاجيل" next to "الكحول") is still found lexically; chosen on the recall grid, see README Search
DOC_MAX_LENGTH = 1024  # tokens; the longest Section of the KB is about 600
QUERY_MAX_LENGTH = 512
CANDIDATES = 12  # fused Sections considered for thinning
THIN_OVERLAP = 0.6  # a Section is passed over when this share of its lines/rows is already in the chosen ones
MAX_EXPANSION_NAMES = 10
# Aspects the KB states in a Section's lines, not in its product rows (the cut-down views lack them).
LINE_ASPECTS = {normalize(w) for w in """side effect effects adverse contraindication contraindications contraindicated
warning warnings avoid interact interaction interactions otc prescription
عرض اعراض الاعراض جانبي جانبيه الجانبيه مضاعفات اثار مانع موانع ممنوع ممنوعه تحذير تحذيرات تداخل تداخلات
a3rad gaanbeya ganbeya mamnoo3 mamno3 mawane3""".split()}


def _units(section: Section) -> set[str]:
    out = {ln.text.strip() for ln in section.items if isinstance(ln, Line) and ln.text.strip()}
    out.update(r.text for t in section.tables for r in t.rows)
    return out


def _pack(v: tuple[float, ...]) -> str:
    return base64.b64encode(array("f", v).tobytes()).decode("ascii")


def _unpack(s: str) -> tuple[float, ...]:
    a = array("f")
    a.frombytes(base64.b64decode(s))
    return tuple(a)


def kb_fingerprint(sections: list[Section] | tuple[Section, ...], files_hash: str = "") -> str:
    h = hashlib.sha256(files_hash.encode())
    for s in sections:
        h.update(s.section_id.encode())
        h.update(b"\0")
        h.update(s.render().encode())
        h.update(b"\0")
    return h.hexdigest()


class HybridRetriever:
    def __init__(
        self,
        sections: list[Section] | tuple[Section, ...],
        embedder_factory: Callable[[], Embedder],
        model_id: str,
        revision: str = "",
        cache_dir: str | Path | None = None,
        files_hash: str = "",
        min_dense: float = 0.0,
        weights: tuple[float, float, float] = WEIGHTS,
        drugs: DrugIndex | None = None,
    ):
        self.sections = tuple(sections)
        self._factory = embedder_factory
        self.model_id, self.revision = model_id, revision
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._files_hash = files_hash
        self.min_dense = min_dense
        self.weights = weights
        self.drugs = drugs or DrugIndex(self.sections)
        self._bm25 = LexicalRetriever(self.sections)
        self._units = [_units(s) for s in self.sections]
        self._by_id = {s.section_id: s for s in self.sections}
        self._embedder: Embedder | None = None
        self._docs: list[Encoded] = []
        self.ready = False
        self.index_source = ""  # "cache" or "built", once ready
        self.last_embed_ms = 0.0
        self.last_best_dense = 0.0  # highest dense similarity of the last search (diagnostics)

    # ------------------------------------------------------------------ loading
    def cache_key(self) -> str:
        h = hashlib.sha256(f"v{FORMAT_VERSION}|{self.model_id}|{self.revision}|{DOC_MAX_LENGTH}|".encode())
        h.update(kb_fingerprint(self.sections, self._files_hash).encode())
        return h.hexdigest()

    def load(self) -> None:
        """Load the model and the index (from the cache when its key matches, else built and cached). Blocking."""
        t0 = time.monotonic()
        self._embedder = self._factory()
        t1 = time.monotonic()
        docs = self._read_cache()
        if docs is None:
            docs = self._embedder.encode([s.render() for s in self.sections], max_length=DOC_MAX_LENGTH)
            self._write_cache(docs)
            self.index_source = "built"
        else:
            self.index_source = "cache"
        self._docs = list(docs)
        self.model_id = getattr(self._embedder, "model_id", self.model_id)
        self.revision = getattr(self._embedder, "revision", self.revision)
        self.ready = True
        log.info("embedder loaded in %.1fs, index %s in %.1fs (%d Sections)", t1 - t0, self.index_source,
                 time.monotonic() - t1, len(self._docs))

    def _cache_file(self) -> Path | None:
        return self._cache_dir / f"index-{self.cache_key()[:24]}.json" if self._cache_dir else None

    def _read_cache(self) -> list[Encoded] | None:
        path = self._cache_file()
        if not path or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["key"] != self.cache_key() or [d["id"] for d in data["docs"]] != [s.section_id for s in self.sections]:
                return None
            return [Encoded(_unpack(d["dense"]), {int(t): w for t, w in d["sparse"].items()}) for d in data["docs"]]
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("index cache %s is unreadable; rebuilding", path)
            return None

    def _write_cache(self, docs: list[Encoded]) -> None:
        path = self._cache_file()
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"key": self.cache_key(), "model": self.model_id, "revision": self.revision, "docs": [
                {"id": s.section_id, "dense": _pack(d.dense), "sparse": {str(t): w for t, w in d.sparse.items()}}
                for s, d in zip(self.sections, docs)]}
            tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, path)
            for old in path.parent.glob("index-*.json"):  # an index of another KB or model is of no use any more
                if old != path:
                    old.unlink(missing_ok=True)
        except OSError:
            log.warning("could not write the index cache %s; it will be rebuilt at the next start", path, exc_info=True)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int) -> list[Section]:
        if not self.ready or self._embedder is None:
            raise RuntimeError("the retriever is not loaded yet")
        mentions = self.drugs.mentions(query)
        expanded = self._expand(query, mentions)
        variants = [query] + ([expanded] if expanded != query else [])
        t0 = time.monotonic()
        enc = self._embedder.encode(variants, max_length=QUERY_MAX_LENGTH)
        self.last_embed_ms = (time.monotonic() - t0) * 1000
        n = len(self.sections)
        rankings: list[tuple[float, list[int]]] = []
        best_dense = 0.0
        lexical_hit = False
        for text, e in zip(variants, enc):
            dense = [dense_score(e.dense, d.dense) for d in self._docs]
            sparse = [sparse_score(e.sparse, d.sparse) for d in self._docs]
            bm25 = self._bm25.scores(text)
            best_dense = max(best_dense, max(dense, default=0.0))
            lexical_hit = lexical_hit or any(x > 0 for x in bm25)
            w_dense, w_sparse, w_bm25 = self.weights
            rankings.append((w_dense, sorted(range(n), key=lambda i: (-dense[i], i))))
            rankings.append((w_sparse, [i for i in sorted(range(n), key=lambda i: (-sparse[i], i)) if sparse[i] > 0]))
            rankings.append((w_bm25, [i for i in sorted(range(n), key=lambda i: (-bm25[i], i)) if bm25[i] > 0]))
        self.last_best_dense = best_dense
        if not lexical_hit and best_dense < self.min_dense:
            return []
        fused = self._fuse(rankings)
        if mentions.ingredients:
            # The question names a drug: the Sections that list it are the likeliest units, so they get one more
            # list in the fusion (in the order they already have), which keeps "OTC" or "class" noise from crowding them out.
            listing = self.drugs.sections_with(mentions.ingredients)
            rankings.append((1.0, [i for i in fused if self.sections[i].section_id in listing]))
            fused = self._fuse(rankings)
        alt = self._alternative_views(query, mentions, fused)
        if alt:
            more = self._asks_more(query)
            # Asked only for the names: the cut-down views (just the product rows). Asked more as well (side
            # effects, pregnancy ...): the whole Sections, whose lines hold it, and then the best other Sections
            # that list the same Active Ingredient. Never a Section of another drug: an Alternative has the
            # same Active Ingredient, and a different drug of its Drug Class is not one.
            chosen = [self._by_id[v.section_id] if more else v for v in alt[:k]]
            if more and len(chosen) < k:
                same = self.drugs.sections_with(mentions.ingredients) - {s.section_id for s in chosen}
                rest = [i for i in fused if self.sections[i].section_id in same]
                chosen += [self.sections[i] for i in self._thin(rest)[:k - len(chosen)]]
            return chosen
        return [self.sections[i] for i in self._thin(fused)[:k]]

    def _expand(self, query: str, m: Mentions) -> str:
        extra: list[str] = []
        for key in m.trade:
            for ing in self.drugs.ingredients_of(key):
                extra.append(self.drugs.ingredient_text(ing).replace(" + ", " "))
        if not m.trade:  # named by ingredient: add its Trade Names
            for ing in m.ingredients:
                extra.extend(self.drugs.display(t) for t in self.drugs.trade_names_of(ing)[:MAX_EXPANSION_NAMES])
        return f"{query} {' '.join(dict.fromkeys(extra))}".strip() if extra else query

    def _fuse(self, rankings: list[tuple[float, list[int]]]) -> list[int]:
        score: dict[int, float] = {}
        for weight, ranking in rankings:
            for rank, i in enumerate(ranking):
                score[i] = score.get(i, 0.0) + weight / (RRF_K + rank + 1)
        return sorted(score, key=lambda i: (-score[i], i))

    def _thin(self, order: list[int]) -> list[int]:
        """`order` with near-duplicate Sections moved behind the others (all of them are kept, in order)."""
        chosen: list[int] = []
        seen: set[str] = set()
        passed: list[int] = []
        for i in order[:CANDIDATES]:
            u = self._units[i]
            if u and chosen and len(u & seen) / len(u) >= THIN_OVERLAP:
                passed.append(i)
                continue
            chosen.append(i)
            seen |= u
        return chosen + passed + order[CANDIDATES:]

    # ------------------------------------------------------------------ Alternatives
    def _wants_alternatives(self, query: str, m: Mentions) -> bool:
        toks = set(tokens(query))
        if toks & ALT_WORDS:
            return True
        return bool(toks & BRAND_WORDS) and bool(m.ingredients) and not m.trade  # "trade names of Ibuprofen"

    def _alternative_views(self, query: str, m: Mentions, fused: list[int]) -> list[Section]:
        if not m.ingredients or not self._wants_alternatives(query, m):
            return []
        prefer = [self.sections[i].section_id for i in fused]
        return self.drugs.alternative_views(m.ingredients, prefer)

    def _asks_more(self, query: str) -> bool:
        """True when the question asks something besides the Alternatives: an aspect the Section's lines answer
        (side effects, contraindications ...) or any other topic word. Drug names, the Alternative frame and
        words the KB answers by its tables (ingredient, class, trade names) do not count."""
        for t in tokens(query):
            if t in LINE_ASPECTS:
                return True
            if len(t) < 3 or t in FRAME or t in ASPECT or t in ABSENT or t in ALT_WORDS or t in self.drugs.words:
                continue
            if t in self.drugs.names or re.fullmatch(r"\d+\w*", t):
                continue
            return True
        return False
