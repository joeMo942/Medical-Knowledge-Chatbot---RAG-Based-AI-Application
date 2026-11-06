"""Unit tests of the evaluation harness (evaluation/): scoring, metrics, case-file schema.\nNo stack and no KB files needed; the shipped case file is validated too."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import httpx
from synthetic import NSAIDS, OVERVIEW, PPI

import medrag
from evaluation import kb_check, report, review, run, scoring
from medrag.models import Line, Row, Section
from evaluation.scoring import (KB_IDS, NO_RESPONSE, Turn, build_kb_index, check_citation, check_contract,
                                confusion_matrix, parse_cases, parse_usage, percentile, score_turn,
                                snippet_check, snippet_in_one_unit, summarize, validate_cases)

CASES_FILE = Path(scoring.__file__).with_name("cases.json")


# ----------------------------------------------------------------- case-file schema

def good_case(**kw):
    c = {"id": "x-one", "category": "dose", "language": "en", "message": "How many mg?",
         "expected_status": "escalated", "note": "A Dose question."}
    c.update(kw)
    return c


def answered_case(**kw):
    c = good_case(expected_status="answered", expected_kb_ids=["KB-02"], anchors=["Controloc"],
                  answer_must_mention=["Pantoprazole"])
    c.update(kw)
    return c


def errors_of(*cases):
    return validate_cases(list(cases))


def test_a_minimal_valid_file():
    assert errors_of(good_case(), answered_case(id="x-two")) == []


def test_every_answered_case_needs_expected_kb_ids():
    c = answered_case()
    del c["expected_kb_ids"]
    assert any("needs expected_kb_ids" in e for e in errors_of(c))
    assert any("needs expected_kb_ids" in e for e in errors_of(answered_case(expected_kb_ids=[])))


@pytest.mark.parametrize("bad", ["KB-00", "KB-10", "kb-01", "KB-1", "KB-011", "02"])
def test_kb_ids_must_be_in_kb_01_to_09(bad):
    assert any("KB-01..KB-09" in e for e in errors_of(answered_case(expected_kb_ids=[bad])))


def test_all_nine_ids_are_accepted():
    assert errors_of(answered_case(expected_kb_ids=list(KB_IDS))) == []


def test_only_answered_cases_carry_expected_kb_ids():
    assert any("only an answered" in e for e in errors_of(good_case(expected_kb_ids=["KB-01"])))


def test_answered_needs_anchors_and_absent_terms_belong_on_other_statuses():
    c = answered_case()
    del c["anchors"]
    assert any("anchors" in e for e in errors_of(c))
    assert any("absent_terms" in e for e in errors_of(answered_case(absent_terms=["malaria"])))


def test_answered_needs_a_relevance_check_and_only_answered_may_have_one():
    c = answered_case()
    del c["answer_must_mention"]
    assert any("relevance" in e for e in errors_of(c))
    assert errors_of(answered_case(answer_must_mention=[], expected_sections=["PPIs"])) == []
    assert any("relevance" in e for e in errors_of(good_case(answer_must_mention=["x"])))
    assert errors_of(answered_case(answer_must_mention=[], answer_must_mention_all=["500", "65"])) == []
    assert any("relevance" in e for e in errors_of(good_case(answer_must_mention_all=["x"])))
    assert any("answer_must_mention_all" in e for e in errors_of(answered_case(answer_must_mention_all=["", "x"])))
    assert any("expected_sections" in e for e in errors_of(answered_case(expected_sections="PPIs")))


@pytest.mark.parametrize("field,value", [
    ("expected_status", "refused"), ("language", "fr"), ("category", "nonsense"), ("message", "  "),
    ("note", ""), ("id", "Bad_Id"), ("acceptable_statuses", ["nope"]),
])
def test_invalid_field_values_are_reported(field, value):
    assert errors_of(good_case(**{field: value}))


def test_missing_fields_unknown_fields_and_duplicate_ids():
    c = good_case()
    del c["language"]
    assert any("language" in e for e in errors_of(c))
    assert any("unknown field 'colour'" in e for e in errors_of(good_case(colour="red")))
    assert any("duplicate id" in e for e in errors_of(good_case(), good_case()))
    assert validate_cases([]) and validate_cases({"a": 1}) and validate_cases(["x"])


def test_arabic_presentation_forms_and_bidi_controls_are_rejected():
    pdf_text = "ﻣﺎ ﻳﻲ ARBs"      # what PDF text extraction yields
    assert any("presentation-form" in e for e in errors_of(good_case(message=pdf_text)))
    assert any("presentation-form" in e for e in errors_of(good_case(message="what ‮is this")))
    assert errors_of(good_case(message="ما هي موانع استعمال ARBs؟")) == []


def test_multi_turn_cases():
    t1 = {"message": "a?", "language": "en", "expected_status": "escalated"}
    t2 = {"message": "b?", "language": "en", "expected_status": "answered", "expected_kb_ids": ["KB-01"],
          "anchors": ["x"], "answer_must_mention": ["x"]}
    ok = {"id": "m-one", "category": "multi_turn", "note": "n", "turns": [t1, t2]}
    assert validate_cases([ok]) == []
    assert any("at least 2" in e for e in validate_cases([{**ok, "turns": [t1]}]))
    assert any("inside 'turns'" in e for e in validate_cases([{**ok, "message": "top level"}]))
    bad_turn = {**t2, "expected_kb_ids": []}
    assert any("turn 2" in e and "needs expected_kb_ids" in e for e in validate_cases([{**ok, "turns": [t1, bad_turn]}]))
    case = parse_cases([ok])[0]
    assert case.multi_turn and len(case.turns) == 2 and case.turns[1].expected_kb_ids == ("KB-01",)


def test_parse_cases_raises_on_invalid_data():
    with pytest.raises(ValueError, match="invalid case file"):
        parse_cases([good_case(language="xx")])


# ----------------------------------------------------------- the shipped test set

@pytest.fixture(scope="module")
def shipped():
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def test_shipped_case_file_is_valid(shipped):
    assert validate_cases(shipped) == []


def test_shipped_set_has_at_least_four_cases_in_every_category(shipped):
    per = Counter(c["category"] for c in shipped)
    assert set(per) == set(scoring.CATEGORIES)
    assert min(per.values()) >= 4
    assert len(shipped) >= 60


def test_shipped_set_has_the_ten_dev_questions_in_order(shipped):
    dev = [c for c in shipped if c["category"] == "dev"]
    assert len(dev) == 10
    assert [c["expected_status"] for c in dev] == ["answered"] * 6 + ["out_of_scope", "out_of_scope", "escalated", "escalated"]


def test_shipped_answered_cases_cover_all_nine_kb_documents(shipped):
    ids = set()
    for case in parse_cases(shipped):
        for t in case.turns:
            if t.expected_status == "answered":
                assert t.expected_kb_ids and set(t.expected_kb_ids) <= set(KB_IDS)
                ids |= set(t.expected_kb_ids)
    assert ids == set(KB_IDS)


def test_shipped_set_has_every_status_and_label(shipped):
    cases = parse_cases(shipped)
    turns = [t for c in cases for t in c.turns]
    assert {t.expected_status for t in turns} == set(scoring.STATUSES)
    assert {t.language for t in turns} == set(scoring.LANGUAGES)
    assert any(c.multi_turn for c in cases)


def test_the_system_never_imports_the_test_set():
    package = Path(medrag.__file__).parent
    for py in package.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not re.search(r"(import|from)\s+evaluation", text), py
        assert "cases.json" not in text, py


# ------------------------------------------------------------- the contract check

def good_body(**kw):
    b = {"answer": "x", "status": "answered", "language": "en",
         "citations": [{"doc_id": "KB-02", "title": "t", "section": "s", "snippet": "n"}]}
    b.update(kw)
    return b


def test_contract_accepts_conforming_responses():
    assert check_contract(200, good_body()) == []
    assert check_contract(200, good_body(status="out_of_scope", citations=[])) == []
    assert check_contract(500, {"error": "the request timed out"}) == []
    assert check_contract(None, None) == []


@pytest.mark.parametrize("body", [
    good_body(extra=1),                                       # extra field
    {k: v for k, v in good_body().items() if k != "answer"},  # missing field
    good_body(answer=5), good_body(status="done"), good_body(language="fr"),
    good_body(citations="x"), good_body(citations=[]),        # answered, no Citation
    good_body(status="out_of_scope"),                         # not answered, but with citations
    good_body(status="escalated"),
    good_body(citations=[{"doc_id": "KB-02", "title": "t", "section": "s"}]),
    good_body(citations=[{"doc_id": "KB-02", "title": "t", "section": "s", "snippet": 1}]),
    good_body(citations=[{"doc_id": "KB-02", "title": "t", "section": "s", "snippet": "n", "x": "y"}]),
    ["not", "an", "object"],
])
def test_contract_violations_are_reported(body):
    assert check_contract(200, body)


def test_contract_for_error_responses():
    assert check_contract(500, {"error": "x", "detail": "y"})
    assert check_contract(500, {"error": 1})
    assert check_contract(500, "boom")
    assert check_contract(400, {"error": "x"})  # only 200 and 500 are in the contract


# --------------------------------------------------------------- snippet checking

def test_snippet_must_lie_within_one_body_line_or_row():
    assert snippet_in_one_unit("Used for pain and inflammation.", NSAIDS)
    assert snippet_in_one_unit("Cataflam – Diclofenac potassium", NSAIDS)         # a whole row
    assert snippet_in_one_unit("Diclofenac potassium", NSAIDS)                    # inside one row, on word boundaries
    assert snippet_in_one_unit("  pain and inflammation  ", NSAIDS)               # whitespace trimmed
    # the rows in document order: Cataflam, then Voltaren
    assert not snippet_in_one_unit("Cataflam – Diclofenac potassium\nVoltaren – Diclofenac sodium", NSAIDS)  # two rows
    assert not snippet_in_one_unit("Diclofenac potassium Voltaren", NSAIDS)       # runs across two rows
    assert not snippet_in_one_unit("Trade Name – Active Ingredient", NSAIDS)      # the table header row
    assert not snippet_in_one_unit("Pain & Arthritis › NSAIDs", NSAIDS)           # heading text
    assert not snippet_in_one_unit("Cataflam – Ibuprofen", NSAIDS)                # invented
    assert not snippet_in_one_unit("", NSAIDS) and not snippet_in_one_unit("   ", NSAIDS)
    assert snippet_in_one_unit("Asthma is a chronic airway disease.", OVERVIEW)


LABELLED = Section("KB-03", "Pain", "NSAIDs", "class", (
    Line("OTC ولا Prescription؟\nغالبًا OTC في المستحضرات المخصصة للـself-care."),   # a soft line break in one paragraph
    Line("Common side effects:"),
    Line("Tremors."),
    Row and NSAIDS.items[1],
))


@pytest.mark.parametrize("snippet", [
    "a", "o", "Ta", "oloc – Pant", "flam – Diclo", "ataflam", "Cataflam – Diclofenac potass",   # fragments, mid-word edges
    "Tremor",                                                  # shorter than the minimum and not a whole line
])
def test_fragments_and_tiny_snippets_are_not_valid(snippet):
    assert not snippet_in_one_unit(snippet, LABELLED)


def test_short_whole_lines_are_valid_but_other_short_pieces_are_not():
    assert snippet_in_one_unit("Tremors.", LABELLED)          # a whole (short) line
    assert not snippet_in_one_unit("Tremors", LABELLED)       # a piece, under 8 characters


def test_a_soft_line_break_splits_a_paragraph_into_two_units():
    assert snippet_in_one_unit("OTC ولا Prescription؟", LABELLED)
    assert snippet_in_one_unit("غالبًا OTC في المستحضرات المخصصة للـself-care.", LABELLED)
    assert not snippet_in_one_unit("OTC ولا Prescription؟\nغالبًا OTC", LABELLED)       # spans the break
    assert not snippet_in_one_unit("Prescription؟ غالبًا OTC في المستحضرات", LABELLED)   # spans it, break as a space
    assert not snippet_in_one_unit("Prescription؟\nغالبًا", LABELLED)


def test_label_lines_are_valid_but_counted_as_labels():
    assert snippet_check("Common side effects:", LABELLED) == {"valid": True, "label": True}
    assert snippet_check("Tremors.", LABELLED) == {"valid": True, "label": False}
    assert snippet_check("Cataflam – Diclofenac potassium", LABELLED) == {"valid": True, "label": False}
    assert snippet_check("made up", LABELLED) == {"valid": False, "label": False}


def _doc(section):
    return type("Doc", (), {"kb_id": section.kb_id, "sections": [section]})()


KB = build_kb_index([_doc(s) for s in (NSAIDS, PPI, OVERVIEW)])


def cite(doc_id="KB-04", title="Pain & Arthritis", section="NSAIDs", snippet="Cataflam – Diclofenac potassium"):
    return {"doc_id": doc_id, "title": title, "section": section, "snippet": snippet}


def test_check_citation_against_the_kb_index():
    assert check_citation(cite(), KB) == {"real_id": True, "section_found": True, "snippet_valid": True,
                                          "snippet_label": False}
    assert check_citation(cite(snippet="made up"), KB)["snippet_valid"] is False
    assert check_citation(cite(section="Other"), KB)["section_found"] is False
    assert check_citation(cite(doc_id="KB-09"), KB)["snippet_valid"] is False   # a real ID, but not this text there
    assert check_citation(cite(doc_id="KB-99"), KB)["real_id"] is False
    # the snippet must come from the cited Section, not from another Section of the KB
    other = cite(doc_id="KB-02", title="Peptic Ulcer & GERD", section="Proton Pump Inhibitors (PPIs)")
    assert check_citation(other, KB)["snippet_valid"] is False


# ------------------------------------------------------------------ metric helpers

def test_percentile_is_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7
    xs = [1, 2, 3, 4, 5]
    assert percentile(xs, 50) == 3
    assert percentile(xs, 0) == 1 and percentile(xs, 100) == 5
    assert percentile([5, 1, 4, 2, 3], 50) == 3     # unsorted input
    assert percentile([10, 20], 50) == 15
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_confusion_matrix_counts_expected_vs_actual():
    pairs = [("answered", "answered")] * 3 + [("answered", "out_of_scope"), ("escalated", "answered"),
                                              ("out_of_scope", "out_of_scope"), ("escalated", "escalated")]
    m = confusion_matrix(pairs)
    assert m["actual"] == list(scoring.STATUSES)
    assert m["counts"]["answered"] == {"answered": 3, "out_of_scope": 1, "escalated": 0}
    assert m["counts"]["escalated"] == {"answered": 1, "out_of_scope": 0, "escalated": 1}
    assert m["counts"]["out_of_scope"] == {"answered": 0, "out_of_scope": 1, "escalated": 0}
    assert sum(sum(r.values()) for r in m["counts"].values()) == len(pairs)


def test_confusion_matrix_has_a_column_for_missing_responses():
    m = confusion_matrix([("answered", NO_RESPONSE), ("escalated", "escalated")])
    assert m["actual"][-1] == NO_RESPONSE and m["counts"]["answered"][NO_RESPONSE] == 1
    assert NO_RESPONSE not in confusion_matrix([("answered", "answered")])["actual"]


def test_parse_usage():
    assert parse_usage('{"prompt_tokens":3,"completion_tokens":4,"llm_calls":1}') == {
        "prompt_tokens": 3, "completion_tokens": 4, "llm_calls": 1}
    for bad in (None, "", "nope", '{"prompt_tokens":3}', '{"prompt_tokens":"x","completion_tokens":1,"llm_calls":1}'):
        assert parse_usage(bad) is None


# --------------------------------------------------------------- per-turn scoring

def obs(body, http=200, latency=1.0, usage=None):
    return {"http_status": http, "body": body, "latency_s": latency, "session_id": "s", "usage": usage}


ANS = Turn("q", "answered", "en", expected_kb_ids=("KB-04", "KB-03"), anchors=("x",))
REL_ANS = Turn("q", "answered", "en", expected_kb_ids=("KB-04",), anchors=("x",),
               expected_sections=("NSAIDs",), answer_must_mention=("diclofenac", "ibuprofen"))


def test_score_turn_all_correct():
    row = score_turn(ANS, obs(good_body(citations=[cite()])), KB)
    assert row["status_ok"] and row["language_ok"] and row["citation_doc_ok"] and row["citation_ids_real"]
    assert row["citation_snippets_valid"] and row["citation_sections_found"] and row["contract_violations"] == []
    assert row["actual_status"] == "answered" and row["cited_doc_ids"] == ["KB-04"]


def test_score_turn_wrong_status_language_doc_and_snippet():
    row = score_turn(ANS, obs(good_body(status="out_of_scope", citations=[], language="ar")), KB)
    assert not row["status_ok"] and not row["language_ok"] and row["citation_doc_ok"] is False
    assert row["citation_ids_real"] is None and row["citation_snippets_valid"] is None
    row = score_turn(ANS, obs(good_body(citations=[cite(doc_id="KB-01")])), KB)
    assert row["status_ok"] and row["citation_doc_ok"] is False and row["citation_ids_real"] is True
    row = score_turn(ANS, obs(good_body(citations=[cite(), cite(snippet="invented")])), KB)
    assert row["citation_snippets_valid"] is False and row["snippets_valid_count"] == 1 and row["citation_count"] == 2
    row = score_turn(ANS, obs(good_body(citations=[cite(doc_id="KB-77")])), KB)
    assert row["citation_ids_real"] is False


def test_score_turn_one_expected_doc_among_several_is_enough():
    row = score_turn(ANS, obs(good_body(citations=[cite(doc_id="KB-01"), cite()])), KB)
    assert row["citation_doc_ok"] is True


def test_score_turn_acceptable_statuses():
    t = Turn("q", "out_of_scope", "en", acceptable_statuses=("answered",))
    assert score_turn(t, obs(good_body()), KB)["status_ok"]
    assert score_turn(t, obs(good_body(status="out_of_scope", citations=[])), KB)["status_ok"]
    assert not score_turn(t, obs(good_body(status="escalated", citations=[])), KB)["status_ok"]
    assert score_turn(t, obs(good_body()), KB)["citation_doc_ok"] is None   # no expected IDs on this turn


def test_score_turn_without_a_usable_response():
    for o in (obs(None, http=None), obs({"error": "busy"}, http=500), obs({"x": 1}, http=200)):
        row = score_turn(ANS, o, KB)
        assert row["actual_status"] == NO_RESPONSE and not row["status_ok"] and not row["language_ok"]
        assert row["citation_count"] == 0


def test_score_turn_without_a_kb_skips_snippet_checks():
    row = score_turn(ANS, obs(good_body(citations=[cite(snippet="anything")])), None)
    assert row["citation_snippets_valid"] is None and row["citation_ids_real"] is True


def test_score_turn_tolerates_malformed_citations():
    row = score_turn(ANS, obs(good_body(citations=["str", {"doc_id": 5}])), KB)
    assert row["contract_violations"] and row["citation_count"] == 1


# ------------------------------------------------------------------------ summary

def mk_rows():
    t_dose = Turn("d", "escalated", "en")
    t_oos = Turn("o", "out_of_scope", "ar")
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "llm_calls": 1}
    rows = []
    for cid, cat, turn, o in [
        ("a1", "dev", ANS, obs(good_body(citations=[cite()]), latency=1.0, usage=usage)),
        ("a2", "dev", ANS, obs(good_body(citations=[cite(snippet="bad")]), latency=2.0,
                               usage={"prompt_tokens": 300, "completion_tokens": 40, "llm_calls": 1})),
        ("d1", "dose", t_dose, obs(good_body(status="escalated", citations=[]), latency=3.0)),
        # expected escalated, answered: a status miss; its citation (title "t") is in no real Section
        ("d2", "dose", t_dose, obs(good_body(), latency=4.0, usage={"prompt_tokens": 200, "completion_tokens": 30, "llm_calls": 1})),
        ("o1", "off_topic", t_oos, obs(good_body(status="out_of_scope", citations=[], language="ar"), latency=5.0)),
        ("o2", "off_topic", t_oos, obs({"error": "busy"}, http=500, latency=30.0)),
    ]:
        rows.append({"case_id": cid, "category": cat, "turn": 1, "message": "q?", **score_turn(turn, o, KB)})
    return rows


def test_summarize_status_accuracy_overall_and_per_category():
    s = summarize(mk_rows())
    st = s["status"]
    assert (st["overall"]["ok"], st["overall"]["n"]) == (4, 6)
    assert (st["per_category"]["dev"]["ok"], st["per_category"]["dev"]["n"]) == (2, 2)
    assert (st["per_category"]["dose"]["ok"], st["per_category"]["dose"]["n"]) == (1, 2)
    assert (st["per_category"]["off_topic"]["ok"], st["per_category"]["off_topic"]["n"]) == (1, 2)
    assert st["confusion"]["counts"]["escalated"]["answered"] == 1
    assert st["confusion"]["counts"]["out_of_scope"][NO_RESPONSE] == 1
    assert s["cases_per_category"] == {"dev": 2, "dose": 2, "off_topic": 2}


def test_summarize_citation_snippet_language_latency_tokens_contract():
    s = summarize(mk_rows())
    assert (s["citation"]["expected_doc_hit"]["ok"], s["citation"]["expected_doc_hit"]["n"]) == (2, 2)
    assert (s["citation"]["doc_ids_real"]["ok"], s["citation"]["doc_ids_real"]["n"]) == (3, 3)  # a1, a2, d2 cite
    # a1 valid; a2's snippet is not in the Section; d2 cites a Section that does not exist
    assert (s["snippet"]["citations_valid"]["ok"], s["snippet"]["citations_valid"]["n"]) == (1, 3)
    assert (s["snippet"]["responses_all_valid"]["ok"], s["snippet"]["responses_all_valid"]["n"]) == (1, 3)
    assert (s["citation"]["sections_found"]["ok"], s["citation"]["sections_found"]["n"]) == (2, 3)
    assert (s["language"]["overall"]["ok"], s["language"]["overall"]["n"]) == (5, 6)   # only the 500 has no label
    assert s["latency_s"]["p50"] == pytest.approx(3.5) and s["latency_s"]["max"] == 30.0
    assert s["latency_s"]["p95"] == pytest.approx(percentile([1, 2, 3, 4, 5, 30], 95))
    assert s["tokens"]["turns_with_usage"] == 3 and s["tokens"]["prompt_mean"] == 200
    assert s["tokens"]["completion_total"] == 90 and s["tokens"]["llm_calls_total"] == 3
    assert s["contract"] == {"violations": 0, "turns_with_violations": 0}
    assert s["http"] == {"non_200": 1, "no_response": 0}


def test_summarize_counts_contract_violations():
    bad = obs(good_body(extra=1, citations=[]))   # extra field, and answered without a Citation
    row = {"case_id": "c", "category": "dose", "turn": 1, **score_turn(Turn("d", "answered", "en", ("KB-01",), anchors=("x",)), bad, KB)}
    assert summarize([row])["contract"] == {"violations": 2, "turns_with_violations": 1}


def test_summarize_empty_metrics_are_none_not_errors():
    row = {"case_id": "c", "category": "dose", "turn": 1,
           **score_turn(Turn("d", "escalated", "en"), obs(None, http=None, latency=None), KB)}
    s = summarize([row])
    assert s["latency_s"]["p50"] is None and s["tokens"]["prompt_mean"] is None
    assert s["snippet"]["citations_valid"]["rate"] is None and report.pct(s["snippet"]["citations_valid"]) == "n/a"
    assert s["http"]["no_response"] == 1


def test_multi_turn_cases_count_all_turns_in_the_case_rate():
    esc = Turn("a", "escalated", "en")
    good = {"case_id": "m", "category": "multi_turn", "turn": 1,
            **score_turn(esc, obs(good_body(status="escalated", citations=[])), KB)}
    bad = {"case_id": "m", "category": "multi_turn", "turn": 2,
           **score_turn(esc, obs(good_body(status="out_of_scope", citations=[])), KB)}
    s = summarize([good, bad])
    assert s["n_cases"] == 1 and s["n_turns"] == 2
    assert (s["status"]["overall"]["ok"], s["status"]["overall"]["n"]) == (1, 2)
    assert (s["status"]["cases_all_turns_right"]["ok"], s["status"]["cases_all_turns_right"]["n"]) == (0, 1)


# ------------------------------------------------------- report and review sheets

def results_of(rows, summary=True):
    return {"meta": {"label": "t", "date": "2026-01-01", "commit": "abc1234", "dirty": False, "llm": "m",
                     "embedding": "e", "base_url": "http://x", "cases_file": "cases.json", "kb_checked": True,
                     "notes": ""},
            "summary": summarize(rows) if summary else {}, "rows": rows}


def test_markdown_summary_has_the_required_sections():
    md = report.render_markdown(results_of(mk_rows()))
    for needle in ("abc1234", "Status accuracy", "Confusion matrix", "| dose |", "p50", "p95", "Contract violations",
                   "Tokens per question", "Status misses", "Language label"):
        assert needle in md
    assert "\n" not in report.cell("a\nb") and "\\|" in report.cell("a|b")


def answered_rows(n):
    return [{"case_id": f"c{i}", "category": "dev", "turn": 1, "message": f"q{i}|x", "actual_status": "answered",
             "body": good_body(answer=f"a{i}\nline2")} for i in range(n)]


def test_review_template_has_empty_columns_for_the_human():
    md = review.render_review(results_of(answered_rows(5), summary=False), n=3, seed=1)
    header = next(line for line in md.splitlines() if line.startswith("| #"))
    assert "faithful?" in header and "invented facts?" in header and "notes" in header
    body_rows = [line for line in md.splitlines() if line.startswith("| ") and line.split("|")[1].strip().isdigit()]
    assert len(body_rows) == 3 and all(line.endswith("|  |  |  |") for line in body_rows)
    assert "KB-02" in md and "\\|" in md
