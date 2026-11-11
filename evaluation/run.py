"""Runs every case of the test set against a running API and writes the results.

    python -m evaluation.run --label baseline [--base-url http://localhost:8000]

Cases run one after another (the API serves one LLM call at a time and refuses concurrent
requests as busy). A single-turn case sends no session_id; a multi-turn case is one Session:
the X-Session-Id of its first reply is sent back as session_id in the body of each next turn.
Output: <out-dir>/<date>-<label>.json (a row per turn, full bodies) and .md (the summary).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from .analytics_check import compare, fetch
from .report import render_markdown
from .scoring import Case, build_kb_index, load_cases, parse_usage, score_turn, summarize

HERE = Path(__file__).parent
DEFAULT_CASES = HERE / "cases.json"
DEFAULT_OUT = HERE / "results"
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_LABEL_CHARS = 80
CLIENT_TIMEOUT_S = 45.0  # above the API's own 30 s limit, so its 500s are recorded, not cut off


def git_info() -> tuple[str, bool]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=HERE.parent, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except Exception:
        return "unknown", False


def post_chat(client: httpx.Client, message: str, session_id: str | None) -> dict[str, Any]:
    """One /chat call; never raises. A transport failure gives http_status None."""
    payload: dict[str, Any] = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    t0 = time.perf_counter()
    try:
        r = client.post("/chat", json=payload)
    except httpx.HTTPError as exc:
        return {"http_status": None, "body": None, "latency_s": time.perf_counter() - t0,
                "session_id": None, "usage": None, "error": f"{type(exc).__name__}: {exc}"}
    latency = time.perf_counter() - t0
    try:
        body = r.json()
    except ValueError:
        body = None
    return {"http_status": r.status_code, "body": body, "latency_s": latency,
            "session_id": r.headers.get("X-Session-Id"), "usage": parse_usage(r.headers.get("X-Medrag-Usage")),
            "error": None if body is not None else "response body is not JSON"}


def run_case(client: httpx.Client, case: Case, kb_index: dict | None) -> list[dict]:
    """Runs the turns of one case in order. A multi-turn case is one Session: the X-Session-Id the
    first reply returned is sent as session_id on each later turn. Each row records the session ID
    SENT (`sent_session_id`) and the one returned (`session_id`); `session_problem` says what went
    wrong when a later turn could not be sent in the Session or the server switched Sessions."""
    rows, carry = [], None
    for n, turn in enumerate(case.turns, 1):
        sent = carry if case.multi_turn else None
        obs = post_chat(client, turn.message, sent)
        problem = None
        if case.multi_turn:
            if n > 1 and sent is None:
                problem = "no X-Session-Id was returned earlier in this case, so this turn started a new Session"
            elif sent and obs["session_id"] and obs["session_id"] != sent:
                problem = f"the server answered in Session {obs['session_id']}, not the one sent ({sent})"
            carry = obs["session_id"] or carry
        row = score_turn(turn, {**obs, "message": turn.message, "sent_session_id": sent, "session_problem": problem}, kb_index)
        rows.append({"case_id": case.id, "category": case.category, "turn": n, **row})
    return rows


def results_paths(stem: Path) -> tuple[Path, Path]:
    """(<stem>.json, <stem>.md). Built by appending, never with Path.with_suffix: a label with a dot
    ("qwen2.5-7b") would otherwise lose everything after the dot."""
    return stem.parent / f"{stem.name}.json", stem.parent / f"{stem.name}.md"


def unique_stem(out_dir: Path, name: str) -> Path:
    """<out-dir>/<name>, or <name>-2, -3, ... when a results file of that name exists: never overwrites."""
    stem, n = out_dir / name, 1
    while any(p.exists() for p in results_paths(stem)):
        n += 1
        stem = out_dir / f"{name}-{n}"
    return stem


def write_results(out_dir: Path, name: str, results: dict) -> tuple[Path, Path]:
    """Writes <name>.json and <name>.md (or <name>-2, ... when taken) into out_dir; returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = results_paths(unique_stem(out_dir, name))
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(results), encoding="utf-8", newline="\n")
    return json_path, md_path


def load_kb_index(kb_dir: Path | None) -> dict | None:
    """The real KB from medrag's own ingestion, or None (with a warning) when it is not there."""
    from medrag.ingest import kb_dir as default_kb_dir, load_knowledge_base, missing_files
    from medrag.manifest import load_manifest
    directory = kb_dir or default_kb_dir()
    if missing_files(load_manifest(), directory):
        print(f"warning: KB files not found in {directory}; snippet checks are skipped", file=sys.stderr)
        return None
    return build_kb_index(load_knowledge_base(directory))


def run_label(value: str) -> str:
    """argparse type for --label: it becomes part of a file name, so only letters, digits, '.', '_' and '-'.
    Checked before the first request: on Windows "qwen3.5:4b" would write into an NTFS alternate data
    stream, and '/', '?' or '*' would fail only after the whole run."""
    if LABEL_RE.match(value) and len(value) <= MAX_LABEL_CHARS:
        return value
    suggestion = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")[:MAX_LABEL_CHARS] or "run"
    raise argparse.ArgumentTypeError(
        f"invalid label {value!r}: use letters, digits, '.', '_' and '-' only, starting with a letter or "
        f"digit, at most {MAX_LABEL_CHARS} characters (for example {suggestion!r})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluation.run", description=__doc__.split("\n")[0])
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--label", required=True, type=run_label,
                   help="run label, e.g. baseline or qwen3.5-4b; part of the output file name "
                        "(letters, digits, '.', '_', '-')")
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--kb-dir", type=Path, help="KB .docx folder for the snippet check (default: $MEDRAG_KB_DIR or ./data)")
    p.add_argument("--notes", default="", help="free text recorded in the results (what the pipeline was at this run)")
    p.add_argument("--only", help="run only cases whose id contains this text (a smoke run)")
    p.add_argument("--analytics-check", action="store_true",
                   help="after the run, compare GET /admin/analytics for the run's window with the run's own numbers")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cases = [c for c in load_cases(args.cases) if not args.only or args.only in c.id]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2
    kb_index = load_kb_index(args.kb_dir)
    with httpx.Client(base_url=args.base_url, timeout=CLIENT_TIMEOUT_S) as client:
        try:
            h = client.get("/health", timeout=10)
            health = h.json() if h.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            health = {}
        if not health:
            print(f"the API at {args.base_url} is not healthy (GET /health); start it with: docker compose up -d --wait", file=sys.stderr)
            return 2
        rows: list[dict] = []
        run_start = time.time()  # the analytics window starts here (server and client share a clock on one machine)
        for i, case in enumerate(cases, 1):
            got = run_case(client, case, kb_index)
            rows += got
            print(f"[{i}/{len(cases)}] {case.id}: " + ", ".join(
                f"{r['actual_status']}{'' if r['status_ok'] else ' (expected ' + r['expected_status'] + ')'} {r['latency_s']:.1f}s"
                for r in got), flush=True)
        report = None
        if args.analytics_check:
            try:
                report = fetch(client, run_start)
            except (httpx.HTTPError, ValueError) as exc:
                print(f"WARNING: analytics check failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    warnings = []
    if not any(r.get("usage") for r in rows):
        warnings.append("No response carried the X-Medrag-Usage header, so the token metrics are absent. "
                        "Start the stack with MEDRAG_USAGE_HEADER=1 to measure tokens.")
    if any(r.get("session_problem") for r in rows):
        warnings.append("Some multi-turn turns were not sent in their Session (see session_problem in the rows).")
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if report is not None:
        lines, matched = compare(rows, report)
        print(chr(10) + "Analytics check (GET /admin/analytics since the run start):")
        print(chr(10).join(lines))
        print("analytics check: " + ("all exact numbers match" if matched else "MISMATCH (see DIFF lines)"))
    commit, dirty = git_info()
    today = dt.date.today().isoformat()
    results = {
        "meta": {"date": today, "label": args.label, "commit": commit, "dirty": dirty,
                 "llm": health.get("llm"), "embedding": health.get("embedding"), "base_url": args.base_url,
                 "cases_file": str(args.cases.name), "kb_checked": kb_index is not None, "notes": args.notes,
                 "usage_header": any(r.get("usage") for r in rows), "warnings": warnings},
        "summary": summarize(rows),
        "rows": rows,
    }
    json_path, md_path = write_results(args.out_dir, f"{today}-{args.label}", results)
    s = results["summary"]
    print(f"\nstatus accuracy {s['status']['overall']['ok']}/{s['status']['overall']['n']}; wrote {json_path} and {md_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
