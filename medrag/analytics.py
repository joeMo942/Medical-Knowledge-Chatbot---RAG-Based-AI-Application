"""The conversation and model-call log, and the aggregates behind GET /admin/analytics.

Two tables in the Sessions database file (separate from `sessions` / `turns`, so the pruning of Sessions never
touches them; created if missing, so an existing volume keeps working):
  turn_log  one row per /chat request that reached the pipeline: Session, message, status, language, cited KB IDs,
            total latency, and whether it went unanswered (Out of Scope or Escalated). A request that failed
            (500, or 499 when the client left) is a turn with status `error`: counted in volume and latency,
            never as unanswered.
  call_log  one row per model call of that turn: kind (routing, answer, embedding), model, prompt and completion
            tokens as the model reported them, and latency.
No real patient data is stored: the project has none.

Writing never raises into /chat: the caller (app.py) wraps `record` and swallows any failure."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .sessions import SessionStore
from .textnorm import normalize
from .usage import LLM_KINDS, CallRecord

MAX_TEXT_CHARS = 4000
UNANSWERED = ("out_of_scope", "escalated")
ERROR = "error"

SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    message TEXT NOT NULL,
    question_key TEXT NOT NULL,
    status TEXT NOT NULL,
    language TEXT,
    kb_ids TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    unanswered INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS turn_log_by_ts ON turn_log (ts);
CREATE TABLE IF NOT EXISTS call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL REFERENCES turn_log(id) ON DELETE CASCADE,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS call_log_by_turn ON call_log (turn_id);
"""


def question_key(message: str) -> str:
    """What makes two questions the same for the top lists: the project's text normaliser (case, Arabic letter
    variants, harakat, digits) and collapsed whitespace."""
    return " ".join(normalize(message).split())


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the value at rank ceil(p/100 * n) of the sorted values. None when there are none."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


@dataclass(frozen=True)
class Window:
    since: float | None = None  # inclusive, epoch seconds
    until: float | None = None  # exclusive

    def where(self, column: str = "ts") -> tuple[str, list[float]]:
        parts, args = [], []
        if self.since is not None:
            parts.append(f"{column} >= ?")
            args.append(self.since)
        if self.until is not None:
            parts.append(f"{column} < ?")
            args.append(self.until)
        return (" AND ".join(parts) or "1=1"), args


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AnalyticsLog:
    def __init__(self, store: SessionStore):
        self._store = store
        with store.connection() as db:
            db.executescript(SCHEMA)

    def record(self, session_id: str, message: str, status: str, language: str | None, kb_ids: list[str],
               latency_s: float, calls: list[CallRecord], ts: float | None = None) -> None:
        ts = time.time() if ts is None else ts
        unanswered = int(status in UNANSWERED)
        with self._store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                turn_id = db.execute(
                    "INSERT INTO turn_log (ts, session_id, message, question_key, status, language, kb_ids, latency_ms, unanswered)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, session_id, message[:MAX_TEXT_CHARS], question_key(message)[:MAX_TEXT_CHARS], status, language,
                     json.dumps(kb_ids), latency_s * 1000, unanswered)).lastrowid
                db.executemany(
                    "INSERT INTO call_log (turn_id, ts, kind, model, prompt_tokens, completion_tokens, latency_ms)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(turn_id, ts, c.kind, c.model, c.prompt_tokens, c.completion_tokens, c.latency_s * 1000) for c in calls])
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def report(self, window: Window = Window(), bucket: str = "day", limit: int = 10) -> dict:
        fmt = "%Y-%m-%dT%H:00:00Z" if bucket == "hour" else "%Y-%m-%dT00:00:00Z"
        where, args = window.where("ts")
        call_where, call_args = window.where("t.ts")
        with self._store.connection() as db:
            summary = db.execute(
                f"SELECT COUNT(*), COALESCE(SUM(unanswered), 0) FROM turn_log WHERE {where}", args).fetchone()
            by_status = dict(db.execute(
                f"SELECT status, COUNT(*) FROM turn_log WHERE {where} GROUP BY status ORDER BY status", args).fetchall())
            latencies = [r[0] for r in db.execute(f"SELECT latency_ms FROM turn_log WHERE {where}", args)]
            volume = [{"bucket": b, "turns": n, "unanswered": u, "errors": e} for b, n, u, e in db.execute(
                f"SELECT strftime(?, ts, 'unixepoch') b, COUNT(*), SUM(unanswered), SUM(status = ?) "
                f"FROM turn_log WHERE {where} GROUP BY b ORDER BY b", [fmt, ERROR, *args])]
            top = self._top(db, where, args, limit, unanswered=False)
            unanswered = self._top(db, where, args, limit, unanswered=True)
            kinds = db.execute(
                f"SELECT c.kind, COUNT(*), SUM(c.prompt_tokens), SUM(c.completion_tokens), AVG(c.latency_ms) "
                f"FROM call_log c JOIN turn_log t ON t.id = c.turn_id WHERE {call_where} "
                f"GROUP BY c.kind ORDER BY c.kind", call_args).fetchall()
        n = summary[0]
        by_kind = [{"kind": k, "calls": c, "prompt_tokens": p, "completion_tokens": o, "total_tokens": p + o,
                    "mean_latency_ms": round(lat, 1)} for k, c, p, o, lat in kinds]
        keys = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        total = {key: sum(r[key] for r in by_kind) for key in keys}
        llm = {key: sum(r[key] for r in by_kind if r["kind"] in LLM_KINDS) for key in keys}
        per_turn = {"turns": n, **{key: total[key] for key in keys[1:]}}
        per_turn.update({f"mean_{key}": round(total[key] / n, 2) if n else None for key in keys[1:]})
        return {
            "window": {"since": _iso(window.since) if window.since is not None else None,
                       "until": _iso(window.until) if window.until is not None else None, "bucket": bucket},
            "summary": {"turns": n, "unanswered": summary[1], "by_status": by_status},
            "volume": volume,
            "top_questions": top,
            "unanswered_questions": unanswered,
            "tokens_by_call_type": {"by_kind": by_kind, "total": {"kind": "total", **total},
                                    "llm_total": {"kind": "llm (routing + answer)", **llm}},
            "tokens_per_turn": per_turn,
            "latency_ms": {"method": "nearest-rank over the turns in the window", "n": n,
                           "p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        }

    @staticmethod
    def _top(db: sqlite3.Connection, where: str, args: list[float], limit: int, unanswered: bool) -> list[dict]:
        cond = f"{where} AND unanswered = 1" if unanswered else where
        rows = db.execute(
            f"SELECT g.n, t.message, t.status, t.ts FROM (SELECT COUNT(*) n, MAX(id) mid FROM turn_log WHERE {cond} "
            f"GROUP BY question_key) g JOIN turn_log t ON t.id = g.mid ORDER BY g.n DESC, g.mid DESC LIMIT ?",
            [*args, limit]).fetchall()
        out = []
        for n, message, status, ts in rows:
            row = {"question": message, "count": n, "last_asked": _iso(ts)}
            if unanswered:
                row["status"] = status  # of the most recent time it was asked
            out.append(row)
        return out
