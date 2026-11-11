"""Cross-check of GET /admin/analytics against the numbers of an evaluation run (run.py --analytics-check).

The run's own rows (status, client-side latency, the X-Medrag-Usage header) are compared with what the server
logged for the same window. Counts and token totals must match exactly; latency is measured at two ends (the
client includes the network, the server only the request) so it is reported side by side, never a failure."""
from __future__ import annotations

import datetime as dt
from collections import Counter

import httpx

from medrag.analytics import UNANSWERED, percentile

LLM_KINDS = ("routing", "answer")


def iso_utc(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def fetch(client: httpx.Client, since: float) -> dict:
    r = client.get("/admin/analytics", params={"since": iso_utc(since), "limit": 1000},
                   timeout=30)
    r.raise_for_status()
    return r.json()


def run_status(row: dict) -> str:
    """The status the server logs for this turn: the body's, or `error` for any non-200 answer."""
    return row["actual_status"] if row.get("http_status") == 200 else "error"


def compare(rows: list[dict], report: dict) -> tuple[list[str], bool]:
    """(report lines, all exact checks matched). Rows without a usable status count as errors."""
    lines, ok = [], True

    def check(name: str, run, server) -> None:
        nonlocal ok
        same = run == server
        ok = ok and same
        lines.append(f"{'ok  ' if same else 'DIFF'} {name}: run {run} | analytics {server}")

    statuses = Counter(run_status(r) for r in rows)
    check("turns", len(rows), report["summary"]["turns"])
    check("unanswered", sum(statuses[s] for s in UNANSWERED), report["summary"]["unanswered"])
    check("status counts", dict(sorted(statuses.items())), report["summary"]["by_status"])
    # Two copies of the same question are one line in the top lists.
    check("top-question turns", len(rows), sum(q["count"] for q in report["top_questions"]))
    lat = [r["latency_s"] * 1000 for r in rows if r.get("latency_s") is not None]
    srv = report["latency_ms"]
    for p in (50, 95):
        client_ms, server_ms = percentile(lat, p), srv[f"p{p}"]
        lines.append(f"info latency p{p} (ms, nearest-rank): client {client_ms:.0f} | server {server_ms:.0f}"
                     if client_ms is not None and server_ms is not None else f"info latency p{p}: n/a")
    usage = [r["usage"] for r in rows if r.get("usage")]
    llm = report["tokens_by_call_type"]["llm_total"]
    if len(usage) == len(rows) and usage:
        for key, name in (("prompt_tokens", "prompt tokens"), ("completion_tokens", "completion tokens"), ("llm_calls", "LLM calls")):
            check(f"{name} (X-Medrag-Usage vs routing+answer)", sum(u[key] for u in usage),
                  llm["calls" if key == "llm_calls" else key])
    else:
        lines.append("skip tokens: not every response carried X-Medrag-Usage (start the stack with MEDRAG_USAGE_HEADER=1)")
    for k in report["tokens_by_call_type"]["by_kind"]:
        lines.append(f"info {k['kind']}: {k['calls']} calls, {k['prompt_tokens']} prompt + {k['completion_tokens']} completion tokens, "
                     f"mean {k['mean_latency_ms']:.0f} ms")
    pt = report["tokens_per_turn"]
    lines.append(f"info per turn: mean {pt['mean_prompt_tokens']} prompt + {pt['mean_completion_tokens']} completion = "
                 f"{pt['mean_total_tokens']} tokens (all call types, {pt['turns']} turns)")
    return lines, ok
