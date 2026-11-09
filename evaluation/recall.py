"""Offline search recall: runs a retriever in-process over the test set, no stack and no LLM needed.

    python -m evaluation.recall                       # BM25 and hybrid, k=4, cases.json + cases_issue5.json
    python -m evaluation.recall --retrievers bm25 --k 4 8
    python -m evaluation.recall --details arabic_trade_names

For every Answered turn it asks the retriever the turn's message, with the k the API uses, and checks
  kb       an expected KB ID is among the retrieved Sections (the Citation could point at a right Document);
  section  where the turn names expected Sections, one of them is retrieved (name or "title > name");
  mention  where the turn lists terms the answer must mention (Trade Names, mostly), one of them occurs in the text
           of the retrieved Sections, i.e. the model could have written it from what it was given.
Turns after the first of a multi-turn case are left out: retrieval there works on the router's standalone
rewrite, which only the running stack produces. Out-of-scope and escalated turns are not recall cases; the
script prints how often they retrieve nothing (the "no match" shortcut) and the best dense similarity they reach.
The hybrid retriever needs the model: MEDRAG_EMBED_PATH (default ~/.cache/medrag/bge-m3) and torch.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from .scoring import PATH_SEPARATOR, Case, load_cases, norm

HERE = Path(__file__).parent
DEFAULT_CASES = (HERE / "cases.json", HERE / "cases_issue5.json")


def build(kind: str, sections, cache_dir: str | None):
    from medrag.retrieval import LexicalRetriever

    if kind == "bm25":
        return LexicalRetriever(sections)
    from medrag.embedding import BGEM3Embedder
    from medrag.hybrid import HybridRetriever
    from medrag.modelstore import MODEL_ID, REVISION

    path = os.environ.get("MEDRAG_EMBED_PATH") or str(Path.home() / ".cache" / "medrag" / "bge-m3")
    r = HybridRetriever(sections, lambda: BGEM3Embedder(path), MODEL_ID, REVISION, cache_dir=cache_dir,
                        min_dense=float(os.environ.get("MEDRAG_MIN_DENSE") or 0.0))
    r.load()
    return r


def section_hit(got, want: set[str]) -> bool:
    return any(norm(s.section) in want or norm(f"{s.title}{PATH_SEPARATOR}{s.section}") in want for s in got)


def evaluate(retriever, cases: list[Case], k: int, details: str | None = None):
    rows = []  # (category, id, kb, section|None, mention|None, retrieved ids)
    quiet = {"n": 0, "empty": 0, "dense": []}
    for case in cases:
        for n, turn in enumerate(case.turns, 1):
            if case.multi_turn and n > 1:
                continue
            got = retriever.search(turn.message, k)
            if turn.expected_status != "answered":
                quiet["n"] += 1
                quiet["empty"] += not got
                quiet["dense"].append(getattr(retriever, "last_best_dense", None))
                continue
            kb = bool({s.kb_id for s in got} & set(turn.expected_kb_ids))
            sec = section_hit(got, {norm(x) for x in turn.expected_sections}) if turn.expected_sections else None
            text = norm("\n".join(s.render() for s in got))
            men = any(norm(t) in text for t in turn.answer_must_mention) if turn.answer_must_mention else None
            rows.append((case.category, case.id, kb, sec, men, [s.section_id for s in got]))
    return rows, quiet


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "-"


def report(name: str, k: int, rows, quiet, details: str | None) -> list[str]:
    out = [f"### {name}, k={k}", "", "| category | turns | kb | section | mention |", "| --- | --- | --- | --- | --- |"]
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
        by["ALL"].append(r)
    for cat in [*sorted(c for c in by if c != "ALL"), "ALL"]:
        rs = by[cat]
        sec = [r for r in rs if r[3] is not None]
        men = [r for r in rs if r[4] is not None]
        out.append(f"| {cat} | {len(rs)} | {pct(sum(r[2] for r in rs), len(rs))} | "
                   f"{pct(sum(bool(r[3]) for r in sec), len(sec))} | {pct(sum(bool(r[4]) for r in men), len(men))} |")
    out.append("")
    misses = [r for r in rows if not r[2] or r[3] is False or r[4] is False]
    out.append(f"Misses ({len(misses)}): " + ", ".join(
        f"{r[1]}[{'kb ' if not r[2] else ''}{'section ' if r[3] is False else ''}{'mention' if r[4] is False else ''}]".replace(" ]", "]")
        for r in misses))
    if quiet["n"]:
        dense = [d for d in quiet["dense"] if d is not None]
        extra = f"; best dense similarity of those that retrieved something: max {max(dense):.2f}" if dense else ""
        out.append(f"Not-answered turns: {quiet['n']}, retrieving nothing: {quiet['empty']}{extra}")
    if details:
        out += ["", f"Details of {details}:"]
        for r in rows:
            if r[0] == details:
                out.append(f"- {r[1]}: kb={r[2]} section={r[3]} mention={r[4]} -> " + "; ".join(r[5]))
    return out + [""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=[str(p) for p in DEFAULT_CASES])
    ap.add_argument("--retrievers", nargs="*", default=["bm25", "hybrid"], choices=["bm25", "hybrid"])
    ap.add_argument("--k", nargs="*", type=int, default=[4])
    ap.add_argument("--details", help="print each turn of this category")
    ap.add_argument("--cache-dir", default=None, help="where the hybrid index is cached (default: not cached)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(HERE.parent))
    from medrag.ingest import kb_dir, load_knowledge_base

    sections = [s for d in load_knowledge_base(kb_dir()) for s in d.sections]
    cases = [c for p in args.cases for c in load_cases(p)]
    for kind in args.retrievers:
        t0 = time.time()
        retriever = build(kind, sections, args.cache_dir)
        print(f"# {kind}: ready in {time.time() - t0:.1f}s", flush=True)
        for k in args.k:
            t0 = time.time()
            rows, quiet = evaluate(retriever, cases, k, args.details)
            print("\n".join(report(kind, k, rows, quiet, args.details)))
            print(f"(search time over {len(rows) + quiet['n']} turns: {time.time() - t0:.1f}s)\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
