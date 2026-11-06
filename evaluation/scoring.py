"""Scoring logic of the evaluation harness. Pure functions: no network, no KB files, no
imports from `medrag`, so the unit tests run anywhere. The snippet check here is
deliberately separate from the system's own `is_valid_snippet` (the system must not grade itself)."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

STATUSES = ("answered", "out_of_scope", "escalated")
LANGUAGES = ("ar", "en", "mixed")
KB_IDS = tuple(f"KB-{n:02d}" for n in range(1, 10))
CATEGORIES = (
    "dev", "paraphrase", "dialect", "franco_arabic", "arabic_trade_names", "trap_out_of_scope",
    "off_topic", "dose", "strength", "emergency", "group_level", "partial", "multi_turn",
    "prompt_injection", "diagnosis", "treatment_decision",
)
TURN_FIELDS = {"message", "expected_status", "acceptable_statuses", "language", "expected_kb_ids",
               "anchors", "absent_terms"}
CASE_FIELDS = {"id", "category", "note", "source", "turns"} | TURN_FIELDS
NO_RESPONSE = "no_response"  # actual "status" when there was no usable 200 reply
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# Arabic presentation forms (what PDF text extraction yields, in visual order) and bidi controls:
# a message with these is not the text a user would type, and would poison language detection.
BAD_TEXT = re.compile("[ﭐ-﷿ﹰ-﻿‎‏‪-‮⁦-⁩]")


# --------------------------------------------------------------------------- cases

@dataclass(frozen=True)
class Turn:
    message: str
    expected_status: str
    language: str  # expected Answer Language (GLOSSARY): ar | en | mixed (mixed only for Franco-Arabic)
    expected_kb_ids: tuple[str, ...] = ()
    acceptable_statuses: tuple[str, ...] = ()  # extra statuses that also pass (rare; explained in the note)
    anchors: tuple[str, ...] = ()  # answered: each expected KB Document must contain one of these terms
    absent_terms: tuple[str, ...] = ()  # out-of-scope traps: none of these may occur anywhere in the KB

    @property
    def passing_statuses(self) -> tuple[str, ...]:
        return (self.expected_status, *self.acceptable_statuses)


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    note: str
    turns: tuple[Turn, ...]
    source: str = ""

    @property
    def multi_turn(self) -> bool:
        return len(self.turns) > 1


def _text_list_errors(where: str, key: str, v: Any) -> list[str]:
    if not isinstance(v, list) or any(not isinstance(x, str) or not x.strip() for x in v):
        return [f"{where}: {key!r} must be a list of non-empty strings"]
    return []


def _turn_errors(where: str, t: Any) -> list[str]:
    if not isinstance(t, dict):
        return [f"{where}: must be an object"]
    errs = []
    msg = t.get("message")
    if not isinstance(msg, str) or not msg.strip():
        errs.append(f"{where}: 'message' must be non-empty text")
    elif BAD_TEXT.search(msg):
        errs.append(f"{where}: 'message' has Arabic presentation-form or bidi-control characters "
                    "(copy-pasted from a PDF?); retype it in logical order with plain letters")
    status = t.get("expected_status")
    if status not in STATUSES:
        errs.append(f"{where}: 'expected_status' must be one of {STATUSES}")
    if t.get("language") not in LANGUAGES:
        errs.append(f"{where}: 'language' must be one of {LANGUAGES}")
    acceptable = t.get("acceptable_statuses", [])
    if not isinstance(acceptable, list) or any(a not in STATUSES for a in acceptable):
        errs.append(f"{where}: 'acceptable_statuses' must be a list of statuses")
    ids = t.get("expected_kb_ids", [])
    if not isinstance(ids, list) or any(i not in KB_IDS for i in ids):
        errs.append(f"{where}: 'expected_kb_ids' must be a list of IDs in KB-01..KB-09")
        ids = []
    if status == "answered" and not ids:
        errs.append(f"{where}: an answered turn needs expected_kb_ids")
    if status != "answered" and ids:
        errs.append(f"{where}: only an answered turn has expected_kb_ids")
    if status == "answered" and not t.get("anchors"):
        errs.append(f"{where}: an answered turn needs 'anchors' (terms to check its KB IDs against the real KB)")
    for key in ("anchors", "absent_terms"):
        errs += _text_list_errors(where, key, t.get(key, []))
    if status == "answered" and t.get("absent_terms"):
        errs.append(f"{where}: 'absent_terms' belongs on out_of_scope turns")
    return errs


def validate_cases(data: Any) -> list[str]:
    """Every problem found in the raw case file; an empty list means it is valid."""
    if not isinstance(data, list) or not data:
        return ["the case file must be a non-empty JSON list"]
    errs: list[str] = []
    seen: set[str] = set()
    for n, c in enumerate(data):
        if not isinstance(c, dict):
            errs.append(f"case #{n}: must be an object")
            continue
        cid = c.get("id")
        where = f"case {cid!r}" if isinstance(cid, str) else f"case #{n}"
        errs += [f"{where}: unknown field {k!r}" for k in c if k not in CASE_FIELDS]
        if not isinstance(cid, str) or not ID_RE.match(cid):
            errs.append(f"{where}: 'id' must be lowercase words joined by '-'")
        elif cid in seen:
            errs.append(f"{where}: duplicate id")
        else:
            seen.add(cid)
        if c.get("category") not in CATEGORIES:
            errs.append(f"{where}: 'category' must be one of {CATEGORIES}")
        if not isinstance(c.get("note"), str) or not c["note"].strip():
            errs.append(f"{where}: 'note' (why this status) is required")
        if "turns" in c:
            if TURN_FIELDS & set(c):
                errs.append(f"{where}: a multi-turn case keeps its fields inside 'turns'")
            turns = c["turns"]
            if not isinstance(turns, list) or len(turns) < 2:
                errs.append(f"{where}: 'turns' must be a list of at least 2 turns")
            else:
                for i, t in enumerate(turns):
                    if isinstance(t, dict):
                        errs += [f"{where} turn {i + 1}: unknown field {k!r}" for k in t if k not in TURN_FIELDS]
                    errs += _turn_errors(f"{where} turn {i + 1}", t)
        else:
            errs += _turn_errors(where, c)
    return errs


def _as_turn(t: dict) -> Turn:
    return Turn(
        message=t["message"], expected_status=t["expected_status"], language=t["language"],
        expected_kb_ids=tuple(t.get("expected_kb_ids", ())),
        acceptable_statuses=tuple(t.get("acceptable_statuses", ())),
        anchors=tuple(t.get("anchors", ())), absent_terms=tuple(t.get("absent_terms", ())))


def parse_cases(data: Any) -> list[Case]:
    errs = validate_cases(data)
    if errs:
        raise ValueError("invalid case file:\n  " + "\n  ".join(errs))
    return [Case(id=c["id"], category=c["category"], note=c["note"], source=c.get("source", ""),
                 turns=tuple(_as_turn(t) for t in c["turns"]) if "turns" in c else (_as_turn(c),))
            for c in data]


def load_cases(path: str | Path) -> list[Case]:
    return parse_cases(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------- contract

def check_contract(http_status: int | None, body: Any) -> list[str]:
    """Violations of the API contract in one response (PDF page 3); empty when conforming.
    200: exactly {answer, citations, status, language}, with the right types; answered
    means at least one Citation, anything else means an empty list. 500: {"error": text}."""
    if http_status is None:
        return []  # no response at all: reported as a transport failure, not a contract one
    if http_status == 500:
        if not (isinstance(body, dict) and set(body) == {"error"} and isinstance(body["error"], str)):
            return ['HTTP 500 body is not exactly {"error": <text>}']
        return []
    if http_status != 200:
        return [f"unexpected HTTP status {http_status}"]
    if not isinstance(body, dict):
        return ["HTTP 200 body is not a JSON object"]
    v: list[str] = []
    expected = {"answer", "citations", "status", "language"}
    if set(body) != expected:
        v.append(f"fields differ from the contract: extra {sorted(set(body) - expected)}, "
                 f"missing {sorted(expected - set(body))}")
    if "answer" in body and not isinstance(body["answer"], str):
        v.append("answer is not a string")
    if "status" in body and body["status"] not in STATUSES:
        v.append(f"status {body['status']!r} is not one of {STATUSES}")
    if "language" in body and body["language"] not in LANGUAGES:
        v.append(f"language {body['language']!r} is not one of {LANGUAGES}")
    if "citations" in body:
        cites = body["citations"]
        if not isinstance(cites, list):
            v.append("citations is not a list")
        else:
            for i, c in enumerate(cites):
                if not (isinstance(c, dict) and set(c) == {"doc_id", "title", "section", "snippet"}
                        and all(isinstance(x, str) for x in c.values())):
                    v.append(f"citation {i} is not exactly {{doc_id, title, section, snippet}} as strings")
            if body.get("status") == "answered" and not cites:
                v.append("answered without a Citation")
            if body.get("status") in ("out_of_scope", "escalated") and cites:
                v.append(f"{body['status']} with citations (must be an empty list)")
    return v


# ------------------------------------------------------------------ snippet check

def _units(section: Any) -> list[str]:
    """Body lines and data rows of a Section (duck-typed: .lines and .rows[].text); never the
    heading and never a table header row."""
    return [u for u in [*section.lines, *(r.text for r in section.rows)] if u.strip()]


def snippet_in_one_unit(snippet: str, section: Any) -> bool:
    """True when the snippet is non-empty and lies within ONE body line or data row of the Section."""
    s = snippet.strip()
    return bool(s) and any(s in u for u in _units(section))


def build_kb_index(documents: Iterable[Any]) -> dict[str, list[Any]]:
    """kb_id -> Sections, from medrag's ingestion (KBDocument.kb_id and .sections)."""
    return {d.kb_id: list(d.sections) for d in documents}


def check_citation(citation: dict, kb_index: dict[str, list[Any]]) -> dict[str, bool]:
    """Checks one Citation against the real KB. `section_found`: some Section of that KB Document
    has exactly this title and section (blocks with no heading of their own cite the disease
    title, so several Sections can match). `snippet_valid`: the snippet lies within one line or
    row of any such Section."""
    doc_id = citation.get("doc_id")
    matches = [s for s in kb_index.get(doc_id, [])
               if s.title == citation.get("title") and s.section == citation.get("section")]
    return {
        "real_id": doc_id in KB_IDS,
        "section_found": bool(matches),
        "snippet_valid": any(snippet_in_one_unit(str(citation.get("snippet", "")), s) for s in matches),
    }


# ---------------------------------------------------------------------- per turn

def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (the numpy default), p in 0..100."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def parse_usage(header: str | None) -> dict[str, int] | None:
    """The X-Medrag-Usage header (JSON), or None when absent or unreadable."""
    if not header:
        return None
    try:
        d = json.loads(header)
        return {k: int(d[k]) for k in ("prompt_tokens", "completion_tokens", "llm_calls")}
    except (ValueError, KeyError, TypeError):
        return None


def score_turn(turn: Turn, obs: dict, kb_index: dict[str, list[Any]] | None = None) -> dict:
    """The observation (http_status, body, ...) plus every per-metric verdict for one turn.
    `kb_index` None means no KB was available: the section and snippet checks are then None."""
    body = obs.get("body")
    ok200 = obs.get("http_status") == 200 and isinstance(body, dict)
    actual = body.get("status") if ok200 and body.get("status") in STATUSES else NO_RESPONSE
    raw_cites = body.get("citations") if ok200 else None
    cites = [c for c in raw_cites if isinstance(c, dict)] if isinstance(raw_cites, list) else []
    cited_ids = [c.get("doc_id") for c in cites]
    checks = [check_citation(c, kb_index) for c in cites] if kb_index is not None else []
    return {
        **obs,
        "expected_status": turn.expected_status,
        "passing_statuses": list(turn.passing_statuses),
        "expected_language": turn.language,
        "expected_kb_ids": list(turn.expected_kb_ids),
        "actual_status": actual,
        "status_ok": actual in turn.passing_statuses,
        "language_ok": bool(ok200 and body.get("language") == turn.language),
        "contract_violations": check_contract(obs.get("http_status"), body),
        "cited_doc_ids": cited_ids,
        # Only a turn that expects an Answered reply has expected KB IDs. Right when the reply is
        # answered and at least one cited doc_id is one of them.
        "citation_doc_ok": (actual == "answered" and bool(set(cited_ids) & set(turn.expected_kb_ids)))
        if turn.expected_kb_ids else None,
        "citation_ids_real": all(i in KB_IDS for i in cited_ids) if cites else None,
        "citation_sections_found": all(c["section_found"] for c in checks) if checks else None,
        "citation_snippets_valid": all(c["snippet_valid"] for c in checks) if checks else None,
        "citation_count": len(cites),
        "snippets_valid_count": sum(c["snippet_valid"] for c in checks) if checks else None,
    }


# ----------------------------------------------------------------------- summary

def _rate(ok: int, n: int) -> dict:
    return {"ok": ok, "n": n, "rate": (ok / n) if n else None}


def confusion_matrix(pairs: Iterable[tuple[str, str]]) -> dict:
    """Expected vs actual status. Rows are the expected status, columns what the API returned,
    plus 'no_response' when some turn had no usable 200 reply."""
    pairs = list(pairs)
    actual_labels = [*STATUSES, *([NO_RESPONSE] if any(a == NO_RESPONSE for _, a in pairs) else [])]
    counts = Counter(pairs)
    return {"expected": list(STATUSES), "actual": actual_labels,
            "counts": {e: {a: counts.get((e, a), 0) for a in actual_labels} for e in STATUSES}}


def summarize(rows: list[dict]) -> dict:
    """All metrics over scored turn rows (score_turn output plus case_id, category, turn)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    cases: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
        cases[r["case_id"]].append(r)
    case_cat = Counter(rs[0]["category"] for rs in cases.values())

    with_cites = [r for r in rows if r["citation_count"]]
    expect_ans = [r for r in rows if r["citation_doc_ok"] is not None]
    answered = [r for r in expect_ans if r["actual_status"] == "answered"]
    snip_rows = [r for r in with_cites if r["citation_snippets_valid"] is not None]
    lat = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float))]
    usage = [r["usage"] for r in rows if r.get("usage")]

    def mean(key: str) -> float | None:
        return sum(u[key] for u in usage) / len(usage) if usage else None

    return {
        "n_cases": len(cases),
        "n_turns": len(rows),
        "cases_per_category": dict(sorted(case_cat.items())),
        "turns_per_category": {c: len(rs) for c, rs in sorted(by_cat.items())},
        "status": {
            "overall": _rate(sum(r["status_ok"] for r in rows), len(rows)),
            "per_category": {c: _rate(sum(r["status_ok"] for r in rs), len(rs)) for c, rs in sorted(by_cat.items())},
            "cases_all_turns_right": _rate(sum(all(t["status_ok"] for t in ts) for ts in cases.values()), len(cases)),
            "confusion": confusion_matrix((r["expected_status"], r["actual_status"]) for r in rows),
        },
        "citation": {
            # of the turns that should be answered: right status and a cited doc_id in the expected set
            "expected_doc_hit": _rate(sum(r["citation_doc_ok"] for r in expect_ans), len(expect_ans)),
            # same, counting only the replies that did come back answered
            "expected_doc_hit_when_answered": _rate(sum(r["citation_doc_ok"] for r in answered), len(answered)),
            "doc_ids_real": _rate(sum(bool(r["citation_ids_real"]) for r in with_cites), len(with_cites)),
            "sections_found": _rate(sum(bool(r["citation_sections_found"]) for r in snip_rows), len(snip_rows)),
        },
        "snippet": {
            "citations_valid": _rate(sum(r["snippets_valid_count"] for r in snip_rows),
                                     sum(r["citation_count"] for r in snip_rows)),
            "responses_all_valid": _rate(sum(bool(r["citation_snippets_valid"]) for r in snip_rows), len(snip_rows)),
        },
        "language": {
            "overall": _rate(sum(r["language_ok"] for r in rows), len(rows)),
            "per_category": {c: _rate(sum(r["language_ok"] for r in rs), len(rs)) for c, rs in sorted(by_cat.items())},
        },
        "latency_s": {"n": len(lat), "p50": percentile(lat, 50), "p95": percentile(lat, 95),
                      "max": max(lat) if lat else None, "mean": (sum(lat) / len(lat)) if lat else None},
        "tokens": {
            "turns_with_usage": len(usage),
            "prompt_mean": mean("prompt_tokens"), "completion_mean": mean("completion_tokens"),
            "llm_calls_mean": mean("llm_calls"),
            "prompt_total": sum(u["prompt_tokens"] for u in usage),
            "completion_total": sum(u["completion_tokens"] for u in usage),
            "llm_calls_total": sum(u["llm_calls"] for u in usage),
        },
        "contract": {
            "violations": sum(len(r["contract_violations"]) for r in rows),
            "turns_with_violations": sum(bool(r["contract_violations"]) for r in rows),
        },
        "http": {"non_200": sum(r.get("http_status") != 200 for r in rows),
                 "no_response": sum(r.get("http_status") is None for r in rows)},
    }
