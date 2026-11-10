"""Conversation and model-call log (#9) and GET /admin/analytics. Fake LLM, a SQLite file per test; no Ollama, no KB."""
import asyncio
import json
import sqlite3
import time

import httpx
import pytest
from synthetic import SECTIONS, FakeLLM, answered
from test_hybrid import FakeEmbedder

from medrag.analytics import AnalyticsLog, Window, percentile, question_key
from medrag.app import create_app
from medrag.config import Settings
from medrag.hybrid import HybridRetriever
from medrag.sessions import SessionStore
from medrag.usage import CallRecord

Q = {"message": "What is the active ingredient in Cataflam?"}
TOKEN = "s3cret-token"


class Api:
    def __init__(self, llm, tmp_path, retriever=None, **settings):
        self.db = tmp_path / "sessions.db"
        settings.setdefault("admin_token", TOKEN)
        self.app = create_app(SECTIONS, llm, Settings(sessions_db=str(self.db), **settings), retriever=retriever)

    def request(self, method, url, **kw):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t") as c:
                return await c.request(method, url, **kw)
        return asyncio.run(go())

    def chat(self, message=Q["message"], **kw):
        return self.request("POST", "/chat", json={"message": message}, **kw)

    def analytics(self, token=TOKEN, **params):
        headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
        return self.request("GET", "/admin/analytics", params=params, headers=headers)

    def rows(self, sql):
        con = sqlite3.connect(self.db)
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()


# --- what one /chat turn writes ---

def test_one_turn_writes_one_turn_row_and_one_row_per_model_call(tmp_path):
    api = Api(FakeLLM(answered(), tokens=(321, 45)), tmp_path)
    r = api.chat()
    assert r.status_code == 200
    (turn,) = api.rows("SELECT session_id, message, status, language, kb_ids, latency_ms, unanswered FROM turn_log")
    assert turn[0] == r.headers["X-Session-Id"] and turn[1] == Q["message"]
    assert turn[2:5] == ("answered", "en", '["KB-04"]') and turn[5] > 0 and turn[6] == 0
    calls = api.rows("SELECT kind, model, prompt_tokens, completion_tokens, latency_ms FROM call_log ORDER BY id")
    assert [c[:4] for c in calls] == [("routing", "fake-model", 321, 45), ("answer", "fake-model", 321, 45)]
    assert all(c[4] >= 0 for c in calls)


def test_kb_ids_are_unique_and_in_order(tmp_path):
    reply = {"answer": "x", "status": "answered", "citations": [
        {"section_id": SECTIONS[0].section_id, "snippet": "Cataflam – Diclofenac potassium"},
        {"section_id": SECTIONS[0].section_id, "snippet": "Voltaren – Diclofenac sodium"}]}
    api = Api(FakeLLM(reply), tmp_path)
    assert api.chat().json()["status"] == "answered"
    assert api.rows("SELECT kb_ids FROM turn_log") == [('["KB-04"]',)]


def test_the_response_and_headers_are_unchanged_by_logging(tmp_path):
    r = Api(FakeLLM(answered()), tmp_path).chat()
    assert set(r.json()) == {"answer", "citations", "status", "language"}
    assert "X-Session-Id" in r.headers and "X-Medrag-Usage" not in r.headers


def test_out_of_scope_and_escalated_turns_are_unanswered(tmp_path):
    api = Api(FakeLLM({"answer": "", "status": "out_of_scope", "citations": []}), tmp_path)
    assert api.chat("what is the capital of France").json()["status"] == "out_of_scope"
    assert api.chat("He is unconscious and not breathing").json()["status"] == "escalated"
    assert api.chat().json()["status"] in ("answered", "out_of_scope")  # whichever; not checked here
    rows = api.rows("SELECT status, unanswered FROM turn_log ORDER BY id")
    assert rows[0] == ("out_of_scope", 1) and rows[1] == ("escalated", 1)
    assert api.rows("SELECT COUNT(*) FROM call_log WHERE turn_id = 2") == [(0,)]  # an emergency rule needs no model call


def test_a_failed_request_is_logged_as_an_error_and_is_not_unanswered(tmp_path):
    api = Api(FakeLLM(answered(), exc=RuntimeError("boom"), tokens=(10, 2)), tmp_path)
    assert api.chat().status_code == 500
    assert api.rows("SELECT status, language, kb_ids, unanswered FROM turn_log") == [("error", None, "[]", 0)]
    assert api.rows("SELECT kind, prompt_tokens, completion_tokens FROM call_log ORDER BY id") == [
        ("routing", 10, 2), ("answer", 0, 0)]  # the answer call failed: no usage reported


def test_an_embedding_call_is_logged_with_the_tokenizer_count_and_the_model_revision(tmp_path):
    class Counting(FakeEmbedder):
        def encode(self, texts, max_length=512):
            return [type(e)(e.dense, e.sparse, len(t.split())) for e, t in zip(super().encode(texts, max_length), texts)]

    emb = Counting()
    retriever = HybridRetriever(SECTIONS, lambda: emb, emb.model_id, emb.revision)
    retriever.load()
    api = Api(FakeLLM(answered(), tokens=(5, 1)), tmp_path, retriever=retriever)
    assert api.chat("Cataflam active ingredient").status_code == 200
    kinds = api.rows("SELECT kind, model, prompt_tokens, completion_tokens FROM call_log ORDER BY id")
    embedding = [k for k in kinds if k[0] == "embedding"]
    assert len(embedding) == 1 and embedding[0][1:] == (
        "fake/bge-m3@r1", sum(len(t.split()) for t in emb.calls[-1]), 0)  # every variant of the question is one pass
    assert [k[0] for k in kinds] == ["routing", "embedding", "answer"]


def test_the_usage_header_still_counts_only_llm_calls(tmp_path):
    emb = FakeEmbedder()
    retriever = HybridRetriever(SECTIONS, lambda: emb, emb.model_id, emb.revision)
    retriever.load()
    api = Api(FakeLLM(answered(), tokens=(7, 3)), tmp_path, retriever=retriever, usage_header=True)
    assert json.loads(api.chat().headers["X-Medrag-Usage"]) == {"prompt_tokens": 14, "completion_tokens": 6, "llm_calls": 2}


# --- logging failures never reach the client ---

def test_a_logging_failure_does_not_fail_chat(tmp_path, monkeypatch, caplog):
    (tmp_path / "ref").mkdir()
    expected = Api(FakeLLM(answered()), tmp_path / "ref").chat().json()

    def broken(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(AnalyticsLog, "record", broken)
    r = Api(FakeLLM(answered()), tmp_path).chat()
    assert r.status_code == 200 and r.json() == expected and r.headers["X-Session-Id"]
    assert "could not write the conversation log" in caplog.text


def test_unwritable_data_is_swallowed_too(tmp_path):
    api = Api(FakeLLM(answered()), tmp_path)
    con = sqlite3.connect(api.db)
    con.execute("DROP TABLE call_log")  # the second insert of the write fails
    con.commit()
    con.close()
    assert api.chat().status_code == 200
    assert api.rows("SELECT COUNT(*) FROM turn_log") == [(0,)]  # the write rolled back as a whole


# --- the log is not pruned with the Sessions ---

def test_session_pruning_does_not_delete_log_rows(tmp_path):
    store = SessionStore(tmp_path / "s.db", max_sessions=1, max_turns=1)
    log = AnalyticsLog(store)
    for i in range(3):
        store.add_turn(f"s{i}", f"q{i}", "a", "answered")
        log.record(f"s{i}", f"q{i}", "answered", "en", ["KB-01"], 0.1, [CallRecord("answer", "m", 1, 1, 0.1)])
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM turn_log").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM call_log").fetchone()[0] == 3


def test_an_existing_sessions_database_gets_the_log_tables_and_keeps_its_sessions(tmp_path):
    path = tmp_path / "old.db"
    SessionStore(path).add_turn("s", "q", "a", "answered")  # a volume made before the log existed
    store = SessionStore(path)
    AnalyticsLog(store)
    AnalyticsLog(store)  # opening twice is harmless
    assert [t.message for t in store.turns("s")] == ["q"]
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM turn_log").fetchone()[0] == 0


# --- aggregation on seeded rows ---

DAY = 86400.0
T0 = 1_700_000_000.0 - 1_700_000_000.0 % DAY  # a UTC midnight


@pytest.fixture
def seeded(tmp_path):
    log = AnalyticsLog(SessionStore(tmp_path / "a.db"))

    def turn(offset_s, message, status, latency_ms, kb=("KB-01",), calls=()):
        log.record("s", message, status, "en", list(kb), latency_ms / 1000, list(calls), ts=T0 + offset_s)

    route, ans = lambda p, c: CallRecord("routing", "m", p, c, 0.1), lambda p, c: CallRecord("answer", "m", p, c, 0.5)
    emb = CallRecord("embedding", "bge", 12, 0, 0.05)
    turn(100, "What is Cataflam?", "answered", 100, calls=[route(10, 2), emb, ans(100, 20)])
    turn(200, "what  is   CATAFLAM?", "answered", 200, calls=[route(10, 2), emb, ans(110, 30)])
    turn(3700, "Capital of France", "out_of_scope", 300, kb=(), calls=[route(10, 2), ans(50, 5)])
    turn(3800, "capital of france", "out_of_scope", 400, kb=(), calls=[route(10, 2)])
    turn(DAY + 5, "He is unconscious", "escalated", 500, kb=(), calls=[])
    turn(DAY + 6, "boom", "error", 1000, kb=(), calls=[route(10, 2)])
    return log


def test_percentiles_are_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile(list(range(1, 11)), 50) == 5 and percentile(list(range(1, 11)), 95) == 10
    assert percentile([4, 1, 3, 2], 50) == 2 and percentile([4, 1, 3, 2], 95) == 4


def test_question_key_uses_the_project_normaliser():
    assert question_key("  What is   CATAFLAM? ") == "what is cataflam?"
    assert question_key("أَسْبِرِين ٢") == question_key("اسبرين 2")


def test_summary_status_counts_and_latency(seeded):
    rep = seeded.report()
    assert rep["summary"] == {"turns": 6, "unanswered": 3, "by_status": {"answered": 2, "error": 1, "escalated": 1, "out_of_scope": 2}}
    assert rep["latency_ms"]["p50"] == 300 and rep["latency_ms"]["p95"] == 1000 and rep["latency_ms"]["n"] == 6


def test_volume_buckets_by_day_and_hour(seeded):
    day = seeded.report(bucket="day")["volume"]
    assert day == [{"bucket": "2023-11-14T00:00:00Z", "turns": 4, "unanswered": 2, "errors": 0},
                   {"bucket": "2023-11-15T00:00:00Z", "turns": 2, "unanswered": 1, "errors": 1}]
    hour = seeded.report(bucket="hour")["volume"]
    assert [(h["bucket"], h["turns"]) for h in hour] == [
        ("2023-11-14T00:00:00Z", 2), ("2023-11-14T01:00:00Z", 2), ("2023-11-15T00:00:00Z", 2)]


def test_window_filters_turns_and_calls(seeded):
    rep = seeded.report(Window(since=T0 + DAY, until=None))
    assert rep["summary"]["turns"] == 2
    assert rep["tokens_by_call_type"]["total"]["calls"] == 1
    rep = seeded.report(Window(since=T0 + 100, until=T0 + 200))  # since inclusive, until exclusive
    assert rep["summary"]["turns"] == 1


def test_top_questions_group_by_normalised_text_and_show_the_latest_original(seeded):
    top = seeded.report()["top_questions"]
    assert [(t["question"], t["count"]) for t in top[:2]] == [("capital of france", 2), ("what  is   CATAFLAM?", 2)]
    assert {t["count"] for t in top} == {2, 1}
    assert len(seeded.report(limit=1)["top_questions"]) == 1


def test_unanswered_questions_with_their_status_exclude_errors(seeded):
    un = seeded.report()["unanswered_questions"]
    assert [(u["question"], u["count"], u["status"]) for u in un] == [
        ("capital of france", 2, "out_of_scope"), ("He is unconscious", 1, "escalated")]


def test_tokens_by_call_type_and_per_turn(seeded):
    t = seeded.report()["tokens_by_call_type"]
    kinds = {k["kind"]: k for k in t["by_kind"]}
    assert (kinds["routing"]["calls"], kinds["routing"]["prompt_tokens"], kinds["routing"]["completion_tokens"]) == (5, 50, 10)
    assert (kinds["answer"]["calls"], kinds["answer"]["prompt_tokens"], kinds["answer"]["total_tokens"]) == (3, 260, 315)
    assert (kinds["embedding"]["calls"], kinds["embedding"]["prompt_tokens"]) == (2, 24)
    assert t["total"] == {"kind": "total", "calls": 10, "prompt_tokens": 334, "completion_tokens": 65, "total_tokens": 399}
    assert t["llm_total"]["calls"] == 8 and t["llm_total"]["prompt_tokens"] == 310
    per_turn = seeded.report()["tokens_per_turn"]
    assert per_turn["turns"] == 6 and per_turn["total_tokens"] == 399 and per_turn["mean_total_tokens"] == 66.5


def test_an_empty_log_reports_zeros(tmp_path):
    rep = AnalyticsLog(SessionStore(tmp_path / "e.db")).report()
    assert rep["summary"]["turns"] == 0 and rep["latency_ms"]["p50"] is None and rep["volume"] == []
    assert rep["tokens_per_turn"]["mean_total_tokens"] is None


# --- the endpoint and its auth ---

def test_admin_is_disabled_without_a_token(tmp_path):
    api = Api(FakeLLM(answered()), tmp_path, admin_token="")
    r = api.analytics(token="anything")
    assert r.status_code == 503 and r.json() == {"error": "admin disabled: set MEDRAG_ADMIN_TOKEN"}
    assert api.analytics(token=None).status_code == 503


def test_wrong_or_missing_token_is_401(tmp_path):
    api = Api(FakeLLM(answered()), tmp_path)
    assert api.analytics(token=None).status_code == 401
    assert api.analytics(token="wrong").status_code == 401
    # non-ASCII bytes must not crash the comparison
    assert api.request("GET", "/admin/analytics", headers={"Authorization": b"Bearer "+bytes([0xc3, 0xa9])}).status_code == 401
    assert api.request("GET", "/admin/analytics", headers={"Authorization": TOKEN}).status_code == 401
    assert api.request("GET", "/admin/analytics", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401


def test_endpoint_returns_the_report_after_chat_turns(tmp_path):
    api = Api(FakeLLM(answered(), tokens=(100, 20)), tmp_path)
    for _ in range(2):
        assert api.chat().status_code == 200
    r = api.analytics(since="2000-01-01", bucket="hour", limit=5)
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["turns"] == 2 and body["summary"]["by_status"] == {"answered": 2}
    assert body["top_questions"][0]["count"] == 2
    assert body["tokens_by_call_type"]["total"]["prompt_tokens"] == 400
    assert {k["kind"] for k in body["tokens_by_call_type"]["by_kind"]} == {"routing", "answer"}
    assert body["latency_ms"]["p50"] > 0


def test_since_until_are_filters_and_bad_parameters_are_400(tmp_path):
    api = Api(FakeLLM(answered()), tmp_path)
    api.chat()
    future = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))
    assert api.analytics(since=future).json()["summary"]["turns"] == 0
    assert api.analytics(until=future).json()["summary"]["turns"] == 1
    for bad in ({"since": "yesterday"}, {"until": "x"}, {"bucket": "week"}, {"limit": "0"}, {"limit": "5000"}):
        assert api.analytics(**bad).status_code == 400
