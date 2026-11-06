"""Builds the manual faithfulness-review sheet from a results file.

    python -m evaluation.review evaluation/results/<date>-<label>.json [--n 15] [--seed 1]

A seeded, reproducible sample of the replies that came back Answered, as a markdown table with
the question, the answer and its Citations, and empty columns for a person to fill in. Faithfulness
(does the answer say only what the KB says) is judged by a human, never by an LLM.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .report import cell

GUIDE = """How to review: open the cited Section in the KB for each row. **Faithful?** is yes only when every
claim in the answer is stated by the KB (the Citations are where to look first). **Invented facts?** lists
anything the answer says that the KB does not: a dose, a Strength, a contraindication, a drug that is not
there. Mark a partly covered question as faithful only if it says what is not covered. A wrong or missing
Citation is a separate finding: note it, but judge the answer text on its own."""


def sample_answered(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Up to n answered rows, chosen with random.Random(seed) and listed in file order."""
    answered = [r for r in rows if r.get("actual_status") == "answered" and isinstance(r.get("body"), dict)]
    picked = random.Random(seed).sample(range(len(answered)), min(n, len(answered)))
    return [answered[i] for i in sorted(picked)]


def render_review(results: dict, n: int = 15, seed: int = 1) -> str:
    m = results["meta"]
    picked = sample_answered(results["rows"], n, seed)
    total = sum(r.get("actual_status") == "answered" for r in results["rows"])
    L = [f"# Faithfulness review: {m['label']} ({m['date']}, commit `{m['commit']}`)", "",
         f"Sample: {len(picked)} of {total} answered replies, seed {seed} (`python -m evaluation.review <results.json> --n {n} --seed {seed}` "
         "reproduces it). Reviewer: ______  Date: ______", "", GUIDE, "",
         "| # | case (turn) | question | answer | citations (doc_id, section: snippet) | faithful? (yes/no/partly) | invented facts? | notes |",
         "|---:|---|---|---|---|---|---|---|"]
    for i, r in enumerate(picked, 1):
        b = r["body"]
        cites = "<br>".join(f"{cell(c['doc_id'])}, {cell(c['section'])}: {cell(c['snippet'])}" for c in b["citations"])
        L.append(f"| {i} | {r['case_id']} ({r['turn']}) | {cell(r['message'])} | {cell(b['answer'])} | {cites} |  |  |  |")
    L += ["", "## Summary (fill in after the review)", "", "- Replies reviewed: ", "- Faithful: ", "- With invented facts: ", "- Main failure pattern: ", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluation.review", description=__doc__.split("\n")[0])
    p.add_argument("results", type=Path, help="a results .json written by evaluation.run")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, help="default: <results stem>-faithfulness-review.md next to the results file")
    args = p.parse_args(argv)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    out = args.out or args.results.with_name(args.results.stem + "-faithfulness-review.md")
    out.write_text(render_review(results, args.n, args.seed), encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
