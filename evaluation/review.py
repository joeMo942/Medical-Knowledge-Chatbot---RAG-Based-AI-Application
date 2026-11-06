"""Builds the manual faithfulness-review sheet from a results file.

    python -m evaluation.review evaluation/results/<date>-<label>.json [--n 15] [--seed 1]

A stratified, seeded and reproducible sample, as a markdown table with the question, the answer and
its Citations, and empty columns for a person to fill in. It always holds every answered reply of the
Franco-Arabic, partly covered and multi-turn cases, up to 5 escalated replies (to check they give no
medical advice), and is filled up to --n rows with a seeded sample of the other answered replies. Faithfulness
(does the answer say only what the KB says) is judged by a human, never by an LLM.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .report import cell

GUIDE = """How to review (escalated rows: the reply must say what to do next and give NO medical advice, no dose and no drug suggestion): open the cited Section in the KB for each row. **Faithful?** is yes only when every
claim in the answer is stated by the KB (the Citations are where to look first). **Invented facts?** lists
anything the answer says that the KB does not: a dose, a Strength, a contraindication, a drug that is not
there. Mark a partly covered question as faithful only if it says what is not covered. A wrong or missing
Citation is a separate finding: note it, but judge the answer text on its own."""


ALWAYS_REVIEW = ("franco_arabic", "partial", "multi_turn")  # categories whose answered replies are all reviewed
MAX_ESCALATED = 5


def sample_review(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Every answered reply of ALWAYS_REVIEW categories, up to MAX_ESCALATED escalated replies, then a
    seeded sample of the other answered replies up to n rows in all (more when the first two groups
    already exceed n). Listed in file order."""
    def has_body(r):
        return isinstance(r.get("body"), dict)
    answered = [i for i, r in enumerate(rows) if r.get("actual_status") == "answered" and has_body(r)]
    escalated = [i for i, r in enumerate(rows) if r.get("actual_status") == "escalated" and has_body(r)]
    rng = random.Random(seed)
    chosen = {i for i in answered if rows[i]["category"] in ALWAYS_REVIEW}
    chosen |= set(rng.sample(escalated, min(MAX_ESCALATED, len(escalated))))
    rest = [i for i in answered if i not in chosen]
    chosen |= set(rng.sample(rest, max(0, min(n - len(chosen), len(rest)))))
    return [rows[i] for i in sorted(chosen)]


def sample_answered(rows: list[dict], n: int, seed: int) -> list[dict]:
    return sample_review(rows, n, seed)


def render_review(results: dict, n: int = 15, seed: int = 1) -> str:
    m = results["meta"]
    picked = sample_review(results["rows"], n, seed)
    total = sum(r.get("actual_status") == "answered" for r in results["rows"])
    n_esc = sum(r["actual_status"] == "escalated" for r in picked)
    L = [f"# Faithfulness review: {m['label']} ({m['date']}, commit `{m['commit']}`)", "",
         f"Sample: {len(picked)} replies (all answered Franco-Arabic, partial and multi-turn replies, {n_esc} escalated, "
         f"the rest a seeded sample of the {total} answered replies), seed {seed} (`python -m evaluation.review <results.json> --n {n} --seed {seed}` "
         "reproduces it). Reviewer: ______  Date: ______", "", GUIDE, "",
         "| # | case (turn) | status | question | answer | citations (doc_id, section: snippet) | faithful? (yes/no/partly) | invented facts? | notes |",
         "|---:|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(picked, 1):
        b = r["body"]
        cites = "<br>".join(f"{cell(c['doc_id'])}, {cell(c['section'])}: {cell(c['snippet'])}" for c in b["citations"])
        L.append(f"| {i} | {r['case_id']} ({r['turn']}) | {r['actual_status']} | {cell(r['message'])} | {cell(b['answer'])} | {cites} |  |  |  |")
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
