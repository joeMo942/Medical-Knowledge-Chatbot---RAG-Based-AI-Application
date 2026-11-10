"""Offline search recall: runs a retriever in-process over the test set, no stack and no LLM needed.

    python -m evaluation.recall                       # BM25 and hybrid, k=1 2 4, the case files + recall_queries.json
    python -m evaluation.recall --retrievers bm25 --k 4 8
    python -m evaluation.recall --details queries

For every Answered turn it asks the retriever the turn's message, with the k the API uses, and reports
  kb       an expected KB ID is among the retrieved Sections (loose: ANY Section of that Document counts);
  relevant a retrieved Section is relevant: it is in an expected KB Document AND is an expected Section, or holds an
           expected anchor or answer term (the honest KB metric: a Section that merely shares a Document does not count);
  section  where the turn names expected Sections, one of them is retrieved;
  mention  where the turn lists terms the answer must mention (Trade Names, mostly), one occurs in the retrieved text;
  prec     the share of the retrieved Sections that are relevant.
`recall_queries.json` holds everyday questions with no drug name (English, Arabic, dialect, Franco-Arabic), each with
the KB IDs and Sections that answer it, written before the retrievers were measured on them.
Turns after the first of a multi-turn case are left out: retrieval there works on the router's standalone rewrite,
which only the running stack produces. Not-answered turns are not recall cases; the script prints how often they retrieve
nothing (the "no match" shortcut). The hybrid retriever runs with the shipped settings (MEDRAG_MIN_DENSE, default
0.5) and needs the model: MEDRAG_EMBED_PATH (default ~/.cache/medrag/bge-m3) and torch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from .scoring import PATH_SEPARATOR, Case, load_cases, norm

HERE = Path(__file__).parent
DEFAULT_CASES = (HERE / "cases.json", HERE / "cases_issue5.json")
DEFAULT_QUERIES = HERE / "recall_queries.json"


def build(kind: str, sections, cache_dir: str | None):
    from medrag.retrieval import LexicalRetriever

    if kind == "bm25":
        return LexicalRetriever(sections)
    if kind == "bm25-plain":  # the walking skeleton's tokenizer, for the before/after
        return LexicalRetriever(sections, normalized=False)
    from medrag.config import Settings
    from medrag.embedding import BGEM3Embedder
    from medrag.hybrid import HybridRetriever
    from medrag.modelstore import MODEL_ID, REVISION

    path = os.environ.get("MEDRAG_EMBED_PATH") or str(Path.home() / ".cache" / "medrag" / "bge-m3")
    r = HybridRetriever(sections, lambda: BGEM3Embedder(path), MODEL_ID, REVISION, cache_dir=cache_dir,
                        min_dense=float(os.environ.get("MEDRAG_MIN_DENSE") or Settings().min_dense))
    r.load()
    return r


def build_names(sections):
    from medrag.arabicnames import ArabicNames, arabic_common_words
    from medrag.drugs import DrugIndex

    return ArabicNames(DrugIndex(sections), arabic_common_words(sections))


def section_hit(got, want: set[str]) -> bool:
    return any(norm(s.section) in want or norm(f"{s.title}{PATH_SEPARATOR}{s.section}") in want for s in got)


class Item:
    """One thing to retrieve for: the message and what makes a Section right for it."""

    def __init__(self, category, id_, message, kb_ids, sections=(), anchors=(), mentions=(), rewrite=None):
        self.category, self.id, self.message = category, id_, message
        self.rewrite = rewrite  # what the routing call's rewrite would be (recall_queries.json only): see query_of
        self.kb_ids, self.sections = set(kb_ids), {norm(x) for x in sections}
        self.anchors, self.mentions = tuple(anchors), tuple(mentions)

    def relevant(self, s) -> bool:
        if s.kb_id not in self.kb_ids:
            return False
        if self.sections and section_hit([s], self.sections):
            return True
        terms = (*self.anchors, *self.mentions)
        if terms:
            text = norm(s.render())
            return any(norm(t) in text for t in terms)
        return not self.sections


def items_from(cases: list[Case], queries: list[dict]) -> list[Item]:
    out = []
    for case in cases:
        for n, turn in enumerate(case.turns, 1):
            if (case.multi_turn and n > 1) or turn.expected_status != "answered":
                continue
            out.append(Item(case.category, case.id, turn.message, turn.expected_kb_ids, turn.expected_sections,
                            turn.anchors, turn.answer_must_mention))
    out += [Item("queries", q["id"], q["message"], q["expected_kb_ids"], q.get("expected_sections", ()),
                 rewrite=q.get("rewrite")) for q in queries]
    return out


def make_query_of(names, use_rewrites: bool):
    """What the pipeline searches for a message: the routing call's rewrite for a Franco-Arabic question (recall has no LLM,
    so `recall_queries.json` carries the rewrite a router would give: Arabic letters, word for word), then the Arabic-letter
    Trade Names mapped to the KB's spelling (arabicnames.py). Without `names` the raw message, as before #7."""
    def query_of(message: str, rewrite: str | None = None) -> str:
        text = rewrite if (use_rewrites and rewrite) else message
        return names.map(text)[0] if names is not None else text
    return query_of


def evaluate(retriever, items: list[Item], cases: list[Case], k: int, query_of=lambda m, r=None: m):
    rows = []  # (category, id, kb, relevant, section|None, mention|None, precision, retrieved ids)
    for it in items:
        got = retriever.search(query_of(it.message, it.rewrite), k)
        kb = bool({s.kb_id for s in got} & it.kb_ids)
        rel = [s for s in got if it.relevant(s)]
        sec = section_hit([s for s in got if s.kb_id in it.kb_ids], it.sections) if it.sections else None
        text = norm("\n".join(s.render() for s in got))
        men = any(norm(t) in text for t in it.mentions) if it.mentions else None
        rows.append((it.category, it.id, kb, bool(rel), sec, men, len(rel) / len(got) if got else 0.0,
                     [s.section_id for s in got]))
    quiet = {"n": 0, "empty": 0}
    for case in cases:
        for n, turn in enumerate(case.turns, 1):
            if (case.multi_turn and n > 1) or turn.expected_status == "answered":
                continue
            quiet["n"] += 1
            quiet["empty"] += not retriever.search(query_of(turn.message), k)
    return rows, quiet


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "-"


def report(name: str, k: int, rows, quiet, details: str | None) -> list[str]:
    out = [f"### {name}, k={k}", "", "| category | turns | kb (loose) | relevant | section | mention | precision |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
        by["ALL"].append(r)
    for cat in [*sorted(c for c in by if c != "ALL"), "ALL"]:
        rs = by[cat]
        sec = [r for r in rs if r[4] is not None]
        men = [r for r in rs if r[5] is not None]
        out.append(f"| {cat} | {len(rs)} | {pct(sum(r[2] for r in rs), len(rs))} | {pct(sum(r[3] for r in rs), len(rs))} | "
                   f"{pct(sum(bool(r[4]) for r in sec), len(sec))} | {pct(sum(bool(r[5]) for r in men), len(men))} | "
                   f"{100 * sum(r[6] for r in rs) / len(rs):.0f}% |")
    out.append("")
    misses = [r for r in rows if not r[3] or r[4] is False or r[5] is False]
    out.append(f"Misses ({len(misses)}): " + ", ".join(
        f"{r[1]}[{'relevant ' if not r[3] else ''}{'section ' if r[4] is False else ''}{'mention' if r[5] is False else ''}]"
        .replace(" ]", "]") for r in misses))
    if quiet["n"]:
        out.append(f"Not-answered turns: {quiet['n']}, retrieving nothing: {quiet['empty']}")
    if details:
        out += ["", f"Details of {details}:"]
        for r in rows:
            if r[0] == details:
                out.append(f"- {r[1]}: kb={r[2]} relevant={r[3]} section={r[4]} mention={r[5]} -> " + "; ".join(r[7]))
    return out + [""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=[str(p) for p in DEFAULT_CASES])
    ap.add_argument("--queries", default=str(DEFAULT_QUERIES), help="the everyday-question file (empty string to skip)")
    ap.add_argument("--retrievers", nargs="*", default=["bm25", "hybrid"], choices=["bm25", "bm25-plain", "hybrid"])
    ap.add_argument("--k", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--details", help="print each turn of this category")
    ap.add_argument("--cache-dir", default=None, help="where the hybrid index is cached (default: not cached)")
    ap.add_argument("--raw", action="store_true", help="search the message as typed (no Arabic-letter name mapping, no rewrites): the numbers before #7")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(HERE.parent))
    from medrag.ingest import kb_dir, load_knowledge_base

    sections = [s for d in load_knowledge_base(kb_dir()) for s in d.sections]
    cases = [c for p in args.cases for c in load_cases(p)]
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8")) if args.queries else []
    items = items_from(cases, queries)
    query_of = make_query_of(None if args.raw else build_names(sections), not args.raw)
    for kind in args.retrievers:
        t0 = time.time()
        retriever = build(kind, sections, args.cache_dir)
        print(f"# {kind}: ready in {time.time() - t0:.1f}s", flush=True)
        for k in args.k:
            t0 = time.time()
            rows, quiet = evaluate(retriever, items, cases, k, query_of)
            print("\n".join(report(kind, k, rows, quiet, args.details)))
            print(f"(search time over {len(items) + quiet['n']} turns: {time.time() - t0:.1f}s)\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
