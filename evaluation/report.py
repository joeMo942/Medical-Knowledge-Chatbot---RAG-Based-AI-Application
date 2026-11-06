"""Markdown rendering of a results file (pure functions, unit-tested)."""
from __future__ import annotations

from typing import Any


def pct(rate: dict | None) -> str:
    """'ok/n (xx.x%)' for a {ok, n, rate} dict; 'n/a' when nothing was measured."""
    if not rate or not rate.get("n"):
        return "n/a"
    return f"{rate['ok']}/{rate['n']} ({100 * rate['rate']:.1f}%)"


def num(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def cell(text: Any, limit: int | None = None) -> str:
    """Text safe inside a markdown table cell: no pipes, no line breaks."""
    s = " ".join(str(text).split()).replace("|", "\\|")
    return s if limit is None or len(s) <= limit else s[: limit - 1] + "…"


def confusion_table(conf: dict) -> list[str]:
    cols = conf["actual"]
    out = ["| expected \\ actual | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for e in conf["expected"]:
        out.append(f"| **{e}** | " + " | ".join(str(conf["counts"][e][a]) for a in cols) + " |")
    return out


def render_markdown(results: dict) -> str:
    m, s, rows = results["meta"], results["summary"], results["rows"]
    L: list[str] = []
    L.append(f"# Evaluation results: {m['label']} ({m['date']})")
    L.append("")
    L.append(f"- Git commit: `{m['commit']}`" + (" (working tree had uncommitted changes)" if m.get("dirty") else ""))
    L.append(f"- LLM: `{m['llm']}`, embedding: `{m['embedding']}` (from `/health`)")
    L.append(f"- API: {m['base_url']}, case file: `{m['cases_file']}` ({s['n_cases']} cases, {s['n_turns']} turns)")
    if m.get("notes"):
        L.append(f"- Notes: {m['notes']}")
    if not m.get("kb_checked"):
        L.append("- **The KB files were not available to the runner: Section and snippet checks were skipped.**")
    L.append("")
    st = s["status"]
    L.append("## Status accuracy")
    L.append("")
    L.append(f"Overall (per turn): **{pct(st['overall'])}**. Cases with every turn right: {pct(st['cases_all_turns_right'])}.")
    L.append("")
    L.append("| category | cases | turns | status accuracy | language accuracy |")
    L.append("|---|---:|---:|---:|---:|")
    for cat, r in st["per_category"].items():
        L.append(f"| {cat} | {s['cases_per_category'].get(cat, 0)} | {s['turns_per_category'][cat]} | {pct(r)} | "
                 f"{pct(s['language']['per_category'][cat])} |")
    L.append("")
    L.append("### Confusion matrix (rows: expected, columns: actual)")
    L.append("")
    L += confusion_table(st["confusion"])
    L.append("")
    c, sn = s["citation"], s["snippet"]
    L.append("## Citations and snippets")
    L.append("")
    L.append(f"- Turns expecting an Answered reply that came back answered with a cited `doc_id` in the expected set: **{pct(c['expected_doc_hit'])}**")
    L.append(f"- The same, among the replies that did come back answered: {pct(c['expected_doc_hit_when_answered'])}")
    L.append(f"- Replies whose cited `doc_id`s are all real KB IDs (KB-01..KB-09): {pct(c['doc_ids_real'])}")
    L.append(f"- Replies whose citations all name a real (`title`, `section`) of that KB Document: {pct(c['sections_found'])}")
    L.append(f"- Citations whose snippet lies within ONE body line or data row of the cited Section (checked against the real KB, not via the API): **{pct(sn['citations_valid'])}**")
    L.append(f"- Replies where every snippet is valid: {pct(sn['responses_all_valid'])}")
    L.append("")
    L.append("## Language label")
    L.append("")
    L.append(f"Matches the expected Answer Language (glossary rule): **{pct(s['language']['overall'])}**")
    L.append("")
    lat, tk, ct, ht = s["latency_s"], s["tokens"], s["contract"], s["http"]
    L.append("## Latency, tokens, contract")
    L.append("")
    L.append(f"- Latency (s): p50 **{num(lat['p50'], 2)}**, p95 **{num(lat['p95'], 2)}**, max {num(lat['max'], 2)}, mean {num(lat['mean'], 2)} ({lat['n']} requests)")
    if tk["turns_with_usage"]:
        L.append(f"- Tokens per question (from the `X-Medrag-Usage` header, {tk['turns_with_usage']} turns): "
                 f"prompt mean {num(tk['prompt_mean'], 0)}, completion mean {num(tk['completion_mean'], 0)}, "
                 f"LLM calls mean {num(tk['llm_calls_mean'], 2)} (totals: {tk['prompt_total']} prompt, {tk['completion_total']} completion, {tk['llm_calls_total']} calls)")
    else:
        L.append("- Tokens per question: not available (no `X-Medrag-Usage` header in any response)")
    L.append(f"- Contract violations: **{ct['violations']}** in {ct['turns_with_violations']} turns (exact fields and types, answered implies at least one Citation, else an empty list)")
    L.append(f"- Non-200 responses: {ht['non_200']} (no response at all: {ht['no_response']})")
    L.append("")
    bad = [r for r in rows if not r["status_ok"]]
    L.append(f"## Status misses ({len(bad)})")
    L.append("")
    if bad:
        L.append("| case | turn | category | expected | actual | question |")
        L.append("|---|---:|---|---|---|---|")
        for r in bad:
            L.append(f"| {r['case_id']} | {r['turn']} | {r['category']} | {r['expected_status']} | {r['actual_status']} | {cell(r['message'], 70)} |")
    else:
        L.append("None.")
    L.append("")
    flagged = [r for r in rows if r["contract_violations"] or r["citation_snippets_valid"] is False
               or r["citation_ids_real"] is False or (r["citation_doc_ok"] is False and r["actual_status"] == "answered")]
    L.append(f"## Citation, snippet or contract problems ({len(flagged)} turns)")
    L.append("")
    if flagged:
        for r in flagged:
            probs = list(r["contract_violations"])
            if r["citation_ids_real"] is False:
                probs.append("a cited doc_id is not a real KB ID")
            if r["citation_snippets_valid"] is False:
                probs.append("a snippet is not within one line/row of its Section")
            if r["citation_doc_ok"] is False and r["actual_status"] == "answered":
                probs.append(f"cited {r['cited_doc_ids']}, expected one of {r['expected_kb_ids']}")
            L.append(f"- `{r['case_id']}` turn {r['turn']}: " + "; ".join(probs))
    else:
        L.append("None.")
    L.append("")
    L.append("Per-case rows (full response bodies, latency, session IDs, per-metric verdicts) are in the JSON file next to this one.")
    L.append("")
    return "\n".join(L)
