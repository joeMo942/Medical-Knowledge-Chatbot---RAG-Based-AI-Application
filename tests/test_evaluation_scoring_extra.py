"""More evaluation-harness tests: relevance, the metric edge cases that mutation testing found
unprotected, the runner's Session handling (against a mock transport), the KB checks on synthetic
text, the stratified review sample, and the rule that results are never tracked by git."""
import argparse
import json
import subprocess
import threading
from pathlib import Path

import httpx
import pytest
from synthetic import NSAIDS, OVERVIEW, PPI

from evaluation import kb_check, report, review, run, scoring
from evaluation.scoring import (NO_RESPONSE, Turn, build_kb_index, check_citation, check_relevance, parse_cases,
                                score_turn, summarize)
from medrag.models import Line, Row, Section, Table


def body(status="answered", citations=None, answer="x", language="en"):
    if citations is None:
        citations = [{"doc_id": "KB-04", "title": "Pain & Arthritis", "section": "NSAIDs",
                      "snippet": "Cataflam – Diclofenac potassium"}] if status == "answered" else []
    return {"answer": answer, "citations": citations, "status": status, "language": language}


def cite(doc_id="KB-04", title="Pain & Arthritis", section="NSAIDs", snippet="Cataflam – Diclofenac potassium"):
    return {"doc_id": doc_id, "title": title, "section": section, "snippet": snippet}


def obs(b, http=200, latency=1.0, usage=None):
    return {"http_status": http, "body": b, "latency_s": latency, "session_id": "s", "usage": usage}


def doc(*sections):
    return type("Doc", (), {"kb_id": sections[0].kb_id, "sections": list(sections)})()


KB = build_kb_index([doc(NSAIDS), doc(PPI), doc(OVERVIEW)])
ANS = Turn("q", "answered", "en", expected_kb_ids=("KB-04",), anchors=("x",), answer_must_mention=("diclofenac",))


def row(case_id, category, turn, o, n=1):
    return {"case_id": case_id, "category": category, "turn": n, "message": "q", **score_turn(turn, o, KB)}


# ------------------------------------------------------------------- relevance

def test_relevance_by_expected_section_and_by_answer_terms():
    t = Turn("q", "answered", "en", ("KB-04",), anchors=("x",), expected_sections=("NSAIDs",))
    assert check_relevance(t, body(), [cite()]) is True
    assert check_relevance(t, body(), [cite(section="Other")]) is False
    path = Turn("q", "answered", "en", ("KB-04",), anchors=("x",), expected_sections=("Pain & Arthritis › NSAIDs",))
    assert check_relevance(path, body(), [cite()]) is True and check_relevance(path, body(), [cite(title="Z")]) is False
    t = Turn("q", "answered", "en", ("KB-04",), anchors=("x",), answer_must_mention=("Tremor", "رعشة"))
    assert check_relevance(t, body(answer="Common effects: tremors and headache"), []) is True
    assert check_relevance(t, body(answer="It is available as an inhaler"), []) is False
    assert check_relevance(t, body(answer="في رَعْشَة"), []) is True          # Arabic diacritics are normalised
    both = Turn("q", "answered", "en", ("KB-04",), anchors=("x",), expected_sections=("NSAIDs",),
                answer_must_mention=("diclofenac",))
    assert check_relevance(both, body(answer="Diclofenac"), [cite()]) is True
    assert check_relevance(both, body(answer="Ibuprofen"), [cite()]) is False       # right section, wrong answer
    every = Turn("q", "answered", "en", ("KB-03",), anchors=("x",), answer_must_mention_all=("500", "65"))
    assert check_relevance(every, body(answer="Paracetamol 500 mg and Caffeine 65 mg"), []) is True
    assert check_relevance(every, body(answer="Paracetamol 500 mg and Caffeine"), []) is False   # one strength missing
    assert check_relevance(every, body(answer="Paracetamol and Caffeine"), []) is False
    # an entry may be a group of alternatives: EVERY entry must match, a group by ANY one of its terms
    grouped = Turn("q", "answered", "en", ("KB-08",), anchors=("x",), answer_must_mention_all=(
        ("breastfeeding", "الرضاعة"), ("not recommended", "avoid", "لا ينصح")))
    assert check_relevance(grouped, body(answer="Statins are not recommended during breastfeeding."), []) is True
    assert check_relevance(grouped, body(answer="لا يُنصح باستخدامها أثناء الرضاعة"), []) is True     # Arabic, diacritics
    assert check_relevance(grouped, body(answer="Statins should be avoided in pregnancy."), []) is False  # topic missing
    assert check_relevance(grouped, body(answer="The KB covers statins and breastfeeding."), []) is False  # no verdict
    pct = Turn("q", "answered", "en", ("KB-07",), anchors=("x",), answer_must_mention_all=(("30%", "30 %"), ("70%",)))
    assert check_relevance(pct, body(answer="NovoMix 30: ٣٠٪ Insulin Aspart + ٧٠٪ Protamine Aspart"), []) is True
    assert check_relevance(pct, body(answer="NovoMix 30 is a premixed insulin"), []) is False
    assert check_relevance(both, body(answer="Diclofenac"), [cite(section="x")]) is False
    assert check_relevance(Turn("q", "answered", "en", ("KB-04",), anchors=("x",)), body(), [cite()]) is None


def test_relevance_is_a_separate_metric_from_status_and_doc_hit():
    right_topic_wrong_answer = row("a", "franco_arabic", ANS, obs(body(answer="It is sold in pharmacies")))
    assert right_topic_wrong_answer["status_ok"] and right_topic_wrong_answer["citation_doc_ok"]
    assert right_topic_wrong_answer["relevance_ok"] is False
    good = row("b", "franco_arabic", ANS, obs(body(answer="Contains Diclofenac")))
    assert good["relevance_ok"] is True
    not_answered = row("c", "franco_arabic", ANS, obs(body("out_of_scope", answer="diclofenac")))
    assert not_answered["relevance_ok"] is False                                  # a non-answer cannot be relevant
    no_check = row("d", "dose", Turn("q", "escalated", "en"), obs(body("escalated")))
    assert no_check["relevance_ok"] is None
    s = summarize([right_topic_wrong_answer, good, not_answered, no_check])
    assert (s["relevance"]["overall"]["ok"], s["relevance"]["overall"]["n"]) == (1, 3)
    assert (s["relevance"]["per_category"]["franco_arabic"]["ok"], s["relevance"]["per_category"]["franco_arabic"]["n"]) == (1, 3)
    assert "dose" not in s["relevance"]["per_category"]


# ------------------------------------------- metrics that mutation testing found unprotected

def test_citation_doc_ok_needs_an_answered_status():
    # an escalated reply that (against the contract) cites an expected Document is still not a hit
    r = row("a", "x", ANS, obs(body("escalated", citations=[cite()])))
    assert r["citation_doc_ok"] is False and r["actual_status"] == "escalated"


def test_section_match_uses_the_title_not_just_the_section_name():
    # KB-01 repeats Section names under Asthma and COPD: the same name, different Sections
    t1 = Table(Row.from_cells(["Trade name", "Active ingredient"]), (Row.from_cells(["Ventolin Evohaler", "Salbutamol"]),), "products", 0, 1)
    t2 = Table(Row.from_cells(["Trade name", "Active ingredient"]), (Row.from_cells(["Salamol", "Salbutamol"]),), "products", 0, 1)
    asthma = Section("KB-01", "Asthma", "SABA", "class", (Line("Asthma use of SABA."), t1))
    copd = Section("KB-01", "COPD", "SABA", "class", (Line("COPD use of SABA."), t2))
    kb = build_kb_index([doc(asthma, copd)])
    asthma_row = cite("KB-01", "Asthma", "SABA", "Ventolin Evohaler – Salbutamol")
    assert check_citation(asthma_row, kb)["snippet_valid"] is True
    # the snippet of the Asthma Section, cited under the COPD title: invalid
    wrong = cite("KB-01", "COPD", "SABA", "Ventolin Evohaler – Salbutamol")
    assert check_citation(wrong, kb)["section_found"] is True and check_citation(wrong, kb)["snippet_valid"] is False
    assert check_citation(cite("KB-01", "Nowhere", "SABA", "Salamol – Salbutamol"), kb)["section_found"] is False


def test_http_status_other_than_200_is_never_a_scored_reply():
    for http in (500, 400, 404, 503):
        r = score_turn(ANS, obs(body(), http=http), KB)      # a body that looks perfect
        assert r["actual_status"] == NO_RESPONSE and not r["status_ok"] and not r["language_ok"]
        assert r["citation_count"] == 0 and r["citation_doc_ok"] is False


def test_expected_doc_hit_counts_every_turn_that_expects_an_answer():
    hit = row("a", "x", ANS, obs(body()))
    miss_status = row("b", "x", ANS, obs(body("out_of_scope")))               # expected answered, got out_of_scope
    miss_doc = row("c", "x", ANS, obs(body(citations=[cite(doc_id="KB-01")])))
    s = summarize([hit, miss_status, miss_doc])["citation"]
    assert (s["expected_doc_hit"]["ok"], s["expected_doc_hit"]["n"]) == (1, 3)
    assert (s["expected_doc_hit_when_answered"]["ok"], s["expected_doc_hit_when_answered"]["n"]) == (1, 2)


def test_citation_validity_counts_every_citation_not_every_reply():
    two_ok = row("a", "x", ANS, obs(body(citations=[cite(), cite(snippet="Used for pain and inflammation.")])))
    one_bad = row("b", "x", ANS, obs(body(citations=[cite(), cite(snippet="made up text here")])))
    none = row("c", "x", Turn("q", "escalated", "en"), obs(body("escalated")))
    s = summarize([two_ok, one_bad, none])["snippet"]
    assert (s["citations_valid"]["ok"], s["citations_valid"]["n"]) == (3, 4)    # 4 citations, not 2 replies
    assert (s["responses_all_valid"]["ok"], s["responses_all_valid"]["n"]) == (1, 2)


def test_a_citation_matching_two_headingless_sections_is_valid_when_either_holds_the_snippet():
    # Blocks without a heading of their own cite the disease title, so an overview and an OTC table of
    # the same disease share (title, section). The snippet only has to be in one of them.
    overview = Section("KB-02", "Peptic Ulcer & GERD", "Peptic Ulcer & GERD", "overview",
                       (Line("Acid-related disorders of the stomach."),), heading_present=False)
    otc = Section("KB-02", "Peptic Ulcer & GERD", "Peptic Ulcer & GERD", "otc",
                  (Table(Row.from_cells(["Drug class", "Classification"]), (Row.from_cells(["PPIs", "Some OTC"]),)),),
                  heading_present=False)
    kb = build_kb_index([doc(overview, otc)])
    c = cite(doc_id="KB-02", title="Peptic Ulcer & GERD", section="Peptic Ulcer & GERD", snippet="PPIs – Some OTC")
    assert check_citation(c, kb)["snippet_valid"] is True
    assert check_citation({**c, "snippet": "Acid-related disorders of the stomach."}, kb)["snippet_valid"] is True


def test_label_line_citations_are_counted():
    sec = Section("KB-04", "Pain & Arthritis", "NSAIDs", "class", (Line("Common side effects:"), Line("Tremors and cramps.")))
    kb = build_kb_index([doc(sec)])
    labelled = score_turn(ANS, obs(body(citations=[cite(snippet="Common side effects:")])), kb)
    proper = score_turn(ANS, obs(body(citations=[cite(snippet="Tremors and cramps.")])), kb)
    assert labelled["citation_snippets_valid"] is True and labelled["label_snippet_count"] == 1
    assert proper["label_snippet_count"] == 0
    rows = [{"case_id": "a", "category": "x", "turn": 1, "message": "q", **labelled},
            {"case_id": "b", "category": "x", "turn": 1, "message": "q", **proper}]
    assert summarize(rows)["snippet"]["label_line_citations"] == {"n": 1, "of": 2}


# ---------------------------------------------- the runner, against a mock transport

def fake_api(log, session_header=True):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        log.append(payload)
        sid = payload.get("session_id", "new-session-1")
        headers = {"X-Medrag-Usage": '{"prompt_tokens":10,"completion_tokens":2,"llm_calls":1}'}
        if session_header:
            headers["X-Session-Id"] = sid
        return httpx.Response(200, json=body("out_of_scope"), headers=headers)
    return handler


def multi_case():
    t = {"message": "first?", "language": "en", "expected_status": "out_of_scope"}
    return parse_cases([{"id": "m-one", "category": "multi_turn", "note": "n",
                         "turns": [t, {**t, "message": "second?"}, {**t, "message": "third?"}]}])[0]


def single_case():
    return parse_cases([{"id": "s-one", "category": "dose", "note": "n", "language": "en", "message": "one?",
                         "expected_status": "out_of_scope"}])[0]


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api")


def test_run_case_carries_the_session_through_a_multi_turn_case():
    log = []
    rows = run.run_case(client_for(fake_api(log)), multi_case(), None)
    assert [p.get("session_id") for p in log] == [None, "new-session-1", "new-session-1"]   # sent in the body
    assert [p["message"] for p in log] == ["first?", "second?", "third?"]
    assert [r["sent_session_id"] for r in rows] == [None, "new-session-1", "new-session-1"]
    assert [r["session_id"] for r in rows] == ["new-session-1"] * 3
    assert [r["turn"] for r in rows] == [1, 2, 3] and all(r["session_problem"] is None for r in rows)
    assert rows[0]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "llm_calls": 1}


def test_run_case_single_turn_sends_no_session_id():
    log = []
    rows = run.run_case(client_for(fake_api(log)), single_case(), None)
    assert log == [{"message": "one?"}] and rows[0]["sent_session_id"] is None


def test_run_case_flags_a_missing_session_header_on_later_turns():
    log = []
    rows = run.run_case(client_for(fake_api(log, session_header=False)), multi_case(), None)
    assert [p.get("session_id") for p in log] == [None, None, None]      # nothing to carry
    assert rows[0]["session_problem"] is None
    assert all("X-Session-Id" in r["session_problem"] for r in rows[1:])


def test_run_case_flags_a_server_that_switches_sessions():
    calls = []

    def switching(request):
        calls.append(1)
        return httpx.Response(200, json=body("out_of_scope"), headers={"X-Session-Id": f"s{len(calls)}"})
    rows = run.run_case(client_for(switching), multi_case(), None)
    assert rows[1]["sent_session_id"] == "s1" and rows[1]["session_id"] == "s2"
    assert "not the one sent" in rows[1]["session_problem"]


def test_post_chat_never_raises_on_transport_failure():
    def boom(request):
        raise httpx.ConnectError("down")
    obs_ = run.post_chat(client_for(boom), "q", None)
    assert obs_["http_status"] is None and obs_["body"] is None and "ConnectError" in obs_["error"]


def test_results_files_are_never_overwritten(tmp_path):
    first = run.unique_stem(tmp_path, "2026-01-01-baseline")
    assert first.name == "2026-01-01-baseline"
    run.results_paths(first)[0].write_text("{}")
    second = run.unique_stem(tmp_path, "2026-01-01-baseline")
    assert second.name == "2026-01-01-baseline-2"
    run.results_paths(second)[1].write_text("x")
    assert run.unique_stem(tmp_path, "2026-01-01-baseline").name == "2026-01-01-baseline-3"


def minimal_results():
    rows = [row("a", "dose", Turn("q", "escalated", "en"), obs(body("escalated")))]
    return {"meta": {"label": "qwen2.5-7b", "date": "2026-01-01", "commit": "abc", "dirty": False, "llm": "m",
                     "embedding": "e", "base_url": "http://x", "cases_file": "c.json", "kb_checked": True, "notes": ""},
            "summary": summarize(rows), "rows": rows}


def write_within(seconds, out_dir, name):
    """write_results in a thread: a regression to the old with_suffix loop fails the test instead of hanging it."""
    done = {}
    t = threading.Thread(target=lambda: done.setdefault("paths", run.write_results(out_dir, name, minimal_results())),
                         daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"write_results({name!r}) did not return within {seconds}s"
    return done["paths"]


def test_a_dotted_label_keeps_its_full_name_and_a_rerun_gets_a_suffix(tmp_path):
    # Path.with_suffix would turn "...-qwen2.5-7b" into "...-qwen2.json" and, on a rerun, loop forever.
    name = "2026-01-01-qwen2.5-7b"
    assert [p.name for p in run.results_paths(tmp_path / name)] == [f"{name}.json", f"{name}.md"]
    first = write_within(5, tmp_path, name)
    assert [p.name for p in first] == [f"{name}.json", f"{name}.md"] and all(p.is_file() for p in first)
    second = write_within(5, tmp_path, name)
    assert [p.name for p in second] == [f"{name}-2.json", f"{name}-2.md"]
    other = write_within(5, tmp_path, "2026-01-01-qwen2.5-14b")   # same text before the dot
    assert other[0].name == "2026-01-01-qwen2.5-14b.json"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"{name}.json", f"{name}.md", f"{name}-2.json", f"{name}-2.md",
         "2026-01-01-qwen2.5-14b.json", "2026-01-01-qwen2.5-14b.md"])
    assert json.loads(first[0].read_text(encoding="utf-8"))["meta"]["label"] == "qwen2.5-7b"


@pytest.mark.parametrize("label", ["baseline", "baseline-r5", "qwen3.5-4b", "qwen2.5_7b", "A1", "x" * 80])
def test_valid_run_labels_are_accepted(label):
    assert run.run_label(label) == label


@pytest.mark.parametrize("label,suggestion", [
    ("qwen3.5:4b", "qwen3.5-4b"),        # ':' would write into an NTFS alternate data stream on Windows
    ("a/b", "a-b"), ("what?", "what"), ("all*", "all"), ("two words", "two-words"), ("-lead", "lead"),
    (".hidden", "hidden"), ("", "run"), ("x" * 81, "x" * 80), ("مقارنة", "run"),
])
def test_invalid_run_labels_are_rejected_with_a_suggestion(label, suggestion):
    with pytest.raises(argparse.ArgumentTypeError) as e:
        run.run_label(label)
    assert repr(suggestion) in str(e.value)


def test_an_invalid_label_stops_the_run_before_any_request(tmp_path, monkeypatch, capsys):
    def no_network(*a, **kw):
        raise AssertionError("the runner opened an HTTP client before validating the label")
    monkeypatch.setattr(run.httpx, "Client", no_network)
    monkeypatch.setattr(run, "load_kb_index", lambda *_: pytest.fail("the KB was loaded before validating the label"))
    with pytest.raises(SystemExit) as e:
        run.main(["--label", "qwen3.5:4b", "--out-dir", str(tmp_path)])
    assert e.value.code == 2
    assert "invalid label 'qwen3.5:4b'" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


def test_markdown_warns_when_the_usage_header_was_missing():
    rows = [row("a", "dose", Turn("q", "escalated", "en"), obs(body("escalated")))]
    results = {"meta": {"label": "t", "date": "2026-01-01", "commit": "abc", "dirty": False, "llm": "m", "embedding": "e",
                        "base_url": "http://x", "cases_file": "c.json", "kb_checked": True, "notes": "",
                        "warnings": ["No response carried the X-Medrag-Usage header"]},
               "summary": summarize(rows), "rows": rows}
    md = report.render_markdown(results)
    assert "WARNING" in md and "absent" in md and "Relevance" in md and "label line" in md


# ---------------------------------------------------- the KB checks on synthetic text

def cases_with(**turn):
    t = {"id": "c-one", "category": "dose", "note": "n", "language": "en", "message": "m", **turn}
    return parse_cases([t])


TEXTS = {"KB-01": kb_check.norm("Ventolin – Salbutamol\nكحة جافة"), "KB-02": kb_check.norm("Controloc – Pantoprazole")}


def test_anchor_check_flags_an_expected_document_without_the_anchor_and_lists_omissions():
    answered = dict(expected_status="answered", answer_must_mention=["x"])
    ok = cases_with(**answered, expected_kb_ids=["KB-01"], anchors=["VENTOLIN"])
    assert kb_check.anchor_problems(ok, TEXTS) == ([], [])
    bad = cases_with(**answered, expected_kb_ids=["KB-02"], anchors=["Ventolin"])
    errors, notes = kb_check.anchor_problems(bad, TEXTS)
    assert len(errors) == 1 and "KB-02" in errors[0]
    broader = cases_with(**answered, expected_kb_ids=["KB-01"], anchors=["Salbutamol", "Pantoprazole"])
    errors, notes = kb_check.anchor_problems(broader, TEXTS)
    assert errors == [] and len(notes) == 1 and "KB-02" in notes[0]            # KB-02 has an anchor, is not expected
    arabic = cases_with(**answered, expected_kb_ids=["KB-01"], anchors=["كَحَّة"])
    assert kb_check.anchor_problems(arabic, TEXTS) == ([], [])             # diacritics are ignored


def test_absent_check_flags_a_term_that_exists_in_the_kb():
    trap = dict(expected_status="out_of_scope")
    assert kb_check.absent_problems(cases_with(**trap, absent_terms=["malaria"]), TEXTS) == []
    errors = kb_check.absent_problems(cases_with(**trap, absent_terms=["malaria", "SALBUTAMOL"]), TEXTS)
    assert len(errors) == 1 and "salbutamol" in errors[0].lower() and "KB-01" in errors[0]


def test_relevance_check_against_synthetic_sections():
    secs = {"KB-01": {kb_check.norm("SABA"), kb_check.norm("Asthma › SABA")}}
    base = dict(expected_status="answered", expected_kb_ids=["KB-01"], anchors=["x"])
    assert kb_check.relevance_problems(cases_with(**base, expected_sections=["saba"]), TEXTS, secs) == []
    assert kb_check.relevance_problems(cases_with(**base, expected_sections=["Asthma › SABA"]), TEXTS, secs) == []
    assert kb_check.relevance_problems(cases_with(**base, expected_sections=["LABA"]), TEXTS, secs)
    assert kb_check.relevance_problems(cases_with(**base, answer_must_mention=["nothing", "salbutamol"]), TEXTS, secs) == []
    assert kb_check.relevance_problems(cases_with(**base, answer_must_mention=["nothing"]), TEXTS, secs)
    # answer_must_mention_all: EVERY term must be in an expected Document
    assert kb_check.relevance_problems(cases_with(**base, answer_must_mention_all=["ventolin", "salbutamol"]), TEXTS, secs) == []
    errors = kb_check.relevance_problems(cases_with(**base, answer_must_mention_all=["salbutamol", "nothing"]), TEXTS, secs)
    assert len(errors) == 1 and "'nothing'" in errors[0]
    # a group of alternatives needs one of its terms in an expected Document
    grouped = cases_with(**base, answer_must_mention_all=[["nothing", "salbutamol"], "ventolin"])
    assert kb_check.relevance_problems(grouped, TEXTS, secs) == []
    errors = kb_check.relevance_problems(cases_with(**base, answer_must_mention_all=[["nothing", "nada"]]), TEXTS, secs)
    assert len(errors) == 1 and "'nada'" in errors[0]


def test_echo_check_flags_answer_terms_the_question_already_contains():
    base = dict(expected_status="answered", expected_kb_ids=["KB-01"], anchors=["x"])
    ok = cases_with(**base, message="What are the side effects of Ventolin?", answer_must_mention=["Tremor"])
    assert kb_check.echo_problems(ok) == []
    echo = cases_with(**base, message="Is there a warning about alcohol with Flagyl?", answer_must_mention=["ALCOHOL", "48"])
    errors = kb_check.echo_problems(echo)
    assert len(errors) == 1 and "'ALCOHOL'" in errors[0]
    arabic = cases_with(**base, message="هل في تحذير من الكُحول؟", answer_must_mention=["الكحول"])   # normalised
    assert len(kb_check.echo_problems(arabic)) == 1
    # answer_must_mention_all: one entry may echo the topic as long as another (the verdict) does not
    topic_and_verdict = cases_with(**base, message="Can statins be used during breastfeeding?",
                                   answer_must_mention_all=[["breastfeeding", "الرضاعة"], ["not recommended", "avoid"]])
    assert kb_check.echo_problems(topic_and_verdict) == []
    all_echo = cases_with(**base, message="Can statins be used during breastfeeding?",
                          answer_must_mention_all=[["breastfeeding"], ["statins"]])
    assert len(kb_check.echo_problems(all_echo)) == 1


def test_answer_must_mention_all_schema_accepts_strings_and_groups_only():
    base = dict(expected_status="answered", expected_kb_ids=["KB-01"], anchors=["x"])
    assert cases_with(**base, answer_must_mention_all=["500", ["65 mg", "65mg"]])[0].turns[0].answer_must_mention_all == (
        ("500",), ("65 mg", "65mg"))
    for bad in ([[]], [["ok", ""]], [5], ["ok", [""]], "500", [{"a": 1}]):
        errors = scoring.validate_cases([{"id": "c-one", "category": "dose", "note": "n", "language": "en",
                                          "message": "m", **base, "answer_must_mention_all": bad}])
        assert any("answer_must_mention_all" in e for e in errors), bad


# --------------------------------------------------------- stratified review sample

def rr(i, category, status="answered"):
    return {"case_id": f"c{i}", "category": category, "turn": 1, "message": "q", "actual_status": status,
            "body": body(status)}


def test_review_sample_always_has_franco_partial_multi_turn_and_up_to_five_escalated():
    rows = ([rr(i, "dev") for i in range(40)] + [rr(100 + i, "franco_arabic") for i in range(5)]
            + [rr(200 + i, "partial") for i in range(4)] + [rr(300 + i, "multi_turn") for i in range(6)]
            + [rr(400 + i, "dose", "escalated") for i in range(9)] + [rr(500, "franco_arabic", "out_of_scope")])
    picked = review.sample_review(rows, 30, seed=1)
    ids = {r["case_id"] for r in picked}
    assert all(f"c{100 + i}" in ids for i in range(5)) and all(f"c{200 + i}" in ids for i in range(4))
    assert all(f"c{300 + i}" in ids for i in range(6))
    assert sum(r["actual_status"] == "escalated" for r in picked) == 5
    assert "c500" not in ids                                                  # not answered, not escalated
    assert len(picked) == 30
    assert picked == review.sample_review(rows, 30, seed=1) and picked != review.sample_review(rows, 30, seed=2)
    mandatory = review.sample_review(rows, 1, seed=1)                         # n smaller than the mandatory rows
    assert len(mandatory) == 5 + 4 + 6 + 5 + review.MIN_RANDOM          # fixed groups plus the random minimum
    order = [rows.index(r) for r in picked]
    assert order == sorted(order)


def test_review_sample_always_has_every_answered_reply_that_should_have_been_escalated():
    unsafe = [{**rr(600 + i, cat), "expected_status": "escalated"}
              for i, cat in enumerate(["emergency", "dose", "diagnosis", "treatment_decision", "emergency", "dose"])]
    rows = [{**rr(i, "dev"), "expected_status": "answered"} for i in range(40)] + unsafe
    picked = review.sample_review(rows, 3, seed=1)                            # n far below the unsafe count
    ids = {r["case_id"] for r in picked}
    assert {f"c{600 + i}" for i in range(6)} <= ids and len(picked) == 6 + review.MIN_RANDOM
    for seed in range(5):                                                     # whatever the seed
        assert {f"c{600 + i}" for i in range(6)} <= {r["case_id"] for r in review.sample_review(rows, 15, seed)}
    md = review.render_review({"meta": {"label": "t", "date": "d", "commit": "c"}, "rows": rows}, n=3, seed=1)
    assert md.count("| answered (expected escalated) |") == 6
    assert "6 answered although they should have been escalated" in md and "unsafe" in md


def test_random_rows_are_added_even_when_the_fixed_groups_exceed_n():
    # The r4 situation: 22 fixed rows against --n 15 left the seeded sample empty.
    unsafe = [{**rr(600 + i, "emergency"), "expected_status": "escalated"} for i in range(11)]
    rows = [{**rr(i, "dev"), "expected_status": "answered"} for i in range(30)] + unsafe + [
        {**rr(700 + i, "franco_arabic"), "expected_status": "answered"} for i in range(11)]
    g = review.review_groups(rows, 15, seed=1)
    assert len(g["unsafe"]) == 11 and len(g["always"]) == 11 and len(g["random"]) == review.MIN_RANDOM
    assert all(rows[i]["category"] == "dev" for i in g["random"])
    md = review.render_review({"meta": {"label": "t", "date": "d", "commit": "c"}, "rows": rows}, n=15, seed=1)
    assert (f"Sample: {22 + review.MIN_RANDOM} replies: 11 answered although they should have been escalated, "
            f"11 other answered Franco-Arabic, partial and multi-turn replies, 0 escalated replies, and "
            f"{review.MIN_RANDOM} chosen at random (seed 1) from the other 30 answered replies") in md
    # with room left under --n, the random sample fills it up
    assert len(review.review_groups(rows, 40, seed=1)["random"]) == 40 - 22


def test_the_review_command_never_overwrites_an_existing_sheet(tmp_path):
    results = {"meta": {"label": "qwen2.5-7b", "date": "d", "commit": "c"},
               "rows": [{**rr(i, "dev"), "expected_status": "answered"} for i in range(3)]}
    src = tmp_path / "2026-01-01-qwen2.5-7b.json"
    src.write_text(json.dumps(results), encoding="utf-8")
    review.main([str(src)])
    review.main([str(src)])
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["2026-01-01-qwen2.5-7b-faithfulness-review-2.md", "2026-01-01-qwen2.5-7b-faithfulness-review.md",
                     "2026-01-01-qwen2.5-7b.json"]
    filled = tmp_path / "mine.md"
    filled.write_text("my filled-in review", encoding="utf-8")
    review.main([str(src), "--out", str(filled)])
    assert filled.read_text(encoding="utf-8") == "my filled-in review" and (tmp_path / "mine-2.md").is_file()


def test_review_sheet_has_a_status_column_and_empty_review_columns():
    rows = [rr(1, "franco_arabic"), rr(2, "dose", "escalated")]
    results = {"meta": {"label": "t", "date": "d", "commit": "c"}, "rows": rows}
    md = review.render_review(results, n=2, seed=1)
    assert "| status |" in md and "| escalated |" in md and "NO medical advice" in md
    assert all(line.endswith("|  |  |  |") for line in md.splitlines() if line.startswith("| ") and line.split("|")[1].strip().isdigit())


# ------------------------------------------------------------- results are never tracked

def test_nothing_under_evaluation_results_is_tracked_by_git():
    root = Path(scoring.__file__).parent.parent
    try:
        out = subprocess.run(["git", "ls-files", "evaluation/results"], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert out.stdout.strip() == ""
    ignored = subprocess.run(["git", "check-ignore", "evaluation/results/x.json"], cwd=root, capture_output=True, text=True)
    assert ignored.returncode == 0
