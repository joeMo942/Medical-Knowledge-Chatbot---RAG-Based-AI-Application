"""Scoring logic of the evaluation harness. Pure functions: no network, no KB files, no
imports from `medrag`, so the unit tests run anywhere. The snippet check here is
deliberately separate from the system's own `is_valid_snippet` (the system must not grade itself)."""
from __future__ import annotations

import json
import re
import unicodedata
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
RELEVANCE_FIELDS = ("expected_sections", "answer_must_mention", "answer_must_mention_all")
TURN_FIELDS = {"message", "expected_status", "acceptable_statuses", "language", "expected_kb_ids",
               "anchors", "absent_terms", *RELEVANCE_FIELDS}
PATH_SEPARATOR = " › "  # joins a Section's disease title and drug class: "Asthma › SABA"
MIN_SNIPPET_CHARS = 8  # a shorter snippet is only valid when it is a whole line or row
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
    expected_sections: tuple[str, ...] = ()  # relevance: a cited Section must be one of these (name, or "title › name")
    answer_must_mention: tuple[str, ...] = ()  # relevance: the answer text must contain one of these terms
    # relevance: the answer text must match EVERY entry; an entry is a term, or a tuple of alternative terms
    # (any one of which matches it): ("500", "65") or (("30%", "30 %"), ("70%", "70 %"))
    answer_must_mention_all: tuple[str | tuple[str, ...], ...] = ()

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


def _group_list_errors(where: str, key: str, v: Any) -> list[str]:
    """A list whose entries are a non-empty string or a non-empty list of non-empty strings."""
    def ok(x: Any) -> bool:
        if isinstance(x, str):
            return bool(x.strip())
        return isinstance(x, list) and bool(x) and all(isinstance(y, str) and y.strip() for y in x)
    if not isinstance(v, list) or not all(ok(x) for x in v):
        return [f"{where}: {key!r} must be a list whose entries are a non-empty string or a non-empty "
                "list of non-empty strings (alternatives)"]
    return []


def term_groups(entries: Iterable[str | Iterable[str]]) -> list[tuple[str, ...]]:
    """answer_must_mention_all entries as groups of alternatives: "500" -> ("500",)."""
    return [(e,) if isinstance(e, str) else tuple(e) for e in entries]


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
    for key in ("anchors", "absent_terms", "expected_sections", "answer_must_mention"):
        errs += _text_list_errors(where, key, t.get(key, []))
    errs += _group_list_errors(where, "answer_must_mention_all", t.get("answer_must_mention_all", []))
    if status == "answered" and t.get("absent_terms"):
        errs.append(f"{where}: 'absent_terms' belongs on out_of_scope turns")
    if status == "answered" and not any(t.get(k) for k in RELEVANCE_FIELDS):
        errs.append(f"{where}: an answered turn needs 'expected_sections', 'answer_must_mention' or "
                    "'answer_must_mention_all' (relevance)")
    if status != "answered" and any(t.get(k) for k in RELEVANCE_FIELDS):
        errs.append(f"{where}: relevance fields belong on answered turns")
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
        anchors=tuple(t.get("anchors", ())), absent_terms=tuple(t.get("absent_terms", ())),
        expected_sections=tuple(t.get("expected_sections", ())),
        answer_must_mention=tuple(t.get("answer_must_mention", ())),
        answer_must_mention_all=tuple(term_groups(t.get("answer_must_mention_all", ()))))


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

_DIACRITICS = re.compile("[\u064b-\u065f\u0670\u0640]")
_ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه",
                              # Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits -> 0-9
                              **{chr(0x0660 + i): str(i) for i in range(10)},
                              **{chr(0x06F0 + i): str(i) for i in range(10)},
                              "٪": "%"})  # Arabic percent sign


def norm(text: str) -> str:
    """Case-insensitive text with Arabic diacritics, tatweel and alef/yeh/teh-marbuta variants folded,
    Arabic-Indic digits mapped to 0-9 ("٥٠٠" matches "500") and the Arabic percent sign "٪" to "%"."""
    return _DIACRITICS.sub("", text).translate(_ARABIC_FOLD).casefold()


# Body lines that only introduce what follows and support no claim by themselves, derived from the
# real KB (every short body line without sentence content): mostly the per-class headings of KB-07 and
# KB-08. Content lines that are just as short ("Menthol", "OTC: None", "Depends on formulation", and
# "Prescription Only", which states the prescription status) are NOT labels.
LABEL_LINES = frozenset(norm(x) for x in (
    "Side Effects", "Side Effects (الأعراض الجانبية)", "Contraindications",
    "Contraindications (موانع الاستعمال)", "الأدوية", "الأدوية الموجودة في السوق المصري",
    "الأدوية والمستحضرات الموجودة في السوق المصري", "مستحضر خاص بالـMigraine",
))
# Lettered sub-headings ("C. Nitroimidazoles") and Arabic ordinal ones ("أولًا: Aspirin – ...").
_SUB_HEADING = re.compile(r"^(?:[A-Z]\.\s+\S|(?:أولًا|ثانيًا|ثالثًا|رابعًا|أولا|ثانيا|ثالثا|رابعا)\s*:)")
LABEL_MAX_WORDS = 7  # a label that ends in ':' or '?' is short ...
_CLAUSE = re.compile("[,،;؛]")  # ... and has no clause break, which marks sentence content


def is_label_line(text: str) -> bool:
    """A body line that labels what follows and states nothing by itself:
    - a short line (at most LABEL_MAX_WORDS words, no comma) ending in ':' or a question mark
      ("Common side effects:", "Trade Names في مصر:", "OTC ولا Prescription؟"). A sentence that happens to
      end in ':' ("في المجموعة دي فيه 3 مواد فعالة أساسية، وبعضها Thiazide-like:") is content;
    - a lettered or Arabic ordinal sub-heading that is not a sentence ("C. Nitroimidazoles", "أولًا: Aspirin");
    - one of LABEL_LINES ("Side Effects", "Contraindications", "الأدوية الموجودة في السوق المصري")."""
    t = text.strip()
    if not t:
        return False
    if norm(t) in LABEL_LINES:
        return True
    if t.endswith((":", "؟", "?")):
        return len(t.split()) <= LABEL_MAX_WORDS and not _CLAUSE.search(t)
    # A sentence ends in '.': "H. pylori is ..." would match the lettered pattern but is content.
    return bool(_SUB_HEADING.match(t)) and not t.endswith(".")


_EDGE_NON_WORD = re.compile(r"^\W+|\W+$")


def word_text_len(s: str) -> int:
    """Length of `s` without leading and trailing punctuation or spaces (": ممنوعة" counts 6)."""
    return len(_EDGE_NON_WORD.sub("", s))


def _is_word_char(ch: str) -> bool:
    return ch == "_" or unicodedata.category(ch)[0] in "LNM"


def _on_word_boundaries(s: str, text: str) -> bool:
    """True when some occurrence of `s` in `text` neither starts nor ends inside a word."""
    i = text.find(s)
    while i != -1:
        j = i + len(s)
        starts_ok = i == 0 or not (_is_word_char(text[i - 1]) and _is_word_char(s[0]))
        ends_ok = j == len(text) or not (_is_word_char(text[j]) and _is_word_char(s[-1]))
        if starts_ok and ends_ok:
            return True
        i = text.find(s, i + 1)
    return False


def _units(section: Any) -> list[tuple[str, bool]]:
    """(text, is_label_line) for every body line and data row of a Section, split on line breaks
    (a soft break inside a paragraph makes two units); duck-typed: .lines and .rows[].text. Never the
    heading and never a table header row. Label lines: see is_label_line (data rows never are)."""
    out: list[tuple[str, bool]] = []
    for text, is_line in [*((ln, True) for ln in section.lines), *((r.text, False) for r in section.rows)]:
        for part in text.split("\n"):
            part = part.strip()
            if part:
                out.append((part, is_line and is_label_line(part)))
    return out


def snippet_check(snippet: str, section: Any) -> dict[str, bool]:
    """`valid`: the snippet is a whole body line or data row of the Section, or lies inside ONE of them,
    has at least MIN_SNIPPET_CHARS characters of word text (leading/trailing punctuation does not count,
    so not ": ممنوعة") and starts and ends on word boundaries (so not "a" or "oloc – Pant").
    `label`: it is valid but supports no claim by itself: the snippet is itself label-shaped
    (is_label_line: "Common side effects:", "Side Effects", "C. Nitroimidazoles"), or every line holding it
    is a label line."""
    snip = snippet.strip()
    holders = [(u, lab) for u, lab in _units(section)
               if snip and (snip == u or (word_text_len(snip) >= MIN_SNIPPET_CHARS and _on_word_boundaries(snip, u)))]
    return {"valid": bool(holders),
            "label": bool(holders) and (is_label_line(snip) or all(lab for _, lab in holders))}


def snippet_in_one_unit(snippet: str, section: Any) -> bool:
    return snippet_check(snippet, section)["valid"]


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
    checks = [snippet_check(str(citation.get("snippet", "")), s) for s in matches]
    valid = [c for c in checks if c["valid"]]
    return {
        "real_id": doc_id in KB_IDS,
        "section_found": bool(matches),
        "snippet_valid": bool(valid),
        "snippet_label": bool(valid) and all(c["label"] for c in valid),
    }


def check_relevance(turn: "Turn", body: Any, cites: list[dict]) -> bool | None:
    """Does the reply answer the question asked? None when the turn defines no relevance check.
    expected_sections: some cited Section is one of them (matched by name or by "title › name").
    answer_must_mention: the answer contains one of the terms (Arabic normalised).
    answer_must_mention_all: the answer matches every entry; an entry is a term or a group of alternative
    terms, matched by any one of them (Arabic normalised).
    Every check that is set must hold."""
    verdicts = []
    if turn.expected_sections:
        want = {norm(x) for x in turn.expected_sections}
        verdicts.append(any(norm(str(c.get("section", ""))) in want
                            or norm(f"{c.get('title', '')}{PATH_SEPARATOR}{c.get('section', '')}") in want
                            for c in cites))
    if turn.answer_must_mention:
        answer = norm(str(body.get("answer", ""))) if isinstance(body, dict) else ""
        verdicts.append(any(norm(t) in answer for t in turn.answer_must_mention))
    if turn.answer_must_mention_all:
        answer = norm(str(body.get("answer", ""))) if isinstance(body, dict) else ""
        verdicts.append(all(any(norm(t) in answer for t in group) for group in term_groups(turn.answer_must_mention_all)))
    return all(verdicts) if verdicts else None


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
        "label_snippet_count": sum(c["snippet_label"] for c in checks) if checks else None,
        # Relevance needs an answered reply: a reply that is not answered cannot answer the question.
        "relevance_ok": (actual == "answered" and check_relevance(turn, body, cites))
        if (turn.expected_sections or turn.answer_must_mention or turn.answer_must_mention_all) else None,
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
    rel_rows = [r for r in rows if r["relevance_ok"] is not None]
    rel_by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rel_rows:
        rel_by_cat[r["category"]].append(r)
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
            # valid snippets that are only a label line ("Common side effects:"): verbatim, but no support
            "label_line_citations": {"n": sum(r["label_snippet_count"] for r in snip_rows),
                                     "of": sum(r["citation_count"] for r in snip_rows)},
        },
        "relevance": {
            # does the reply answer the question asked (expected section cited / key terms in the answer)
            "overall": _rate(sum(bool(r["relevance_ok"]) for r in rel_rows), len(rel_rows)),
            "per_category": {c: _rate(sum(bool(r["relevance_ok"]) for r in rs), len(rs))
                             for c, rs in sorted(rel_by_cat.items())},
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
