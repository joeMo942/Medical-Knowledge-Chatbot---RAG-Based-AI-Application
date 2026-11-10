"""Hand-off flow (#12): POST /handoff, the handoffs table, its listing in analytics and /admin/handoffs, and the chat
page's form. Fake LLM, a SQLite file per test; all text is synthetic."""
import asyncio
import re
import sqlite3
from pathlib import Path

import httpx
from synthetic import SECTIONS, FakeLLM, answered

from medrag.analytics import AnalyticsLog
from medrag.app import create_app
from medrag.config import Settings
from medrag.escalation import ESCALATION_TEXT
from medrag.handoff import HandoffLog
from medrag.sessions import SessionStore

TOKEN = "s3cret-token"
EMERGENCY_EN = "my child swallowed a whole bottle of panadol"
EMERGENCY_AR = "طفلي ابتلع علبة بانادول كاملة"
DOSE = "what dose of panadol should I take"
ANSWERED = "What is the active ingredient in Cataflam?"
GOOD = {"summary": "My child swallowed tablets at 3pm (synthetic).", "contact": "test@example.invalid"}


class Api:
    def __init__(self, tmp_path, llm=None, **settings):
        self.db = tmp_path / "sessions.db"
        settings.setdefault("admin_token", TOKEN)
        self.app = create_app(SECTIONS, llm or FakeLLM(answered()), Settings(sessions_db=str(self.db), **settings))

    def request(self, method, url, **kw):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t") as c:
                return await c.request(method, url, **kw)
        return asyncio.run(go())

    def chat(self, message, sid=None):
        r = self.request("POST", "/chat", json={"message": message}, headers={"X-Session-Id": sid} if sid else {})
        return r.headers["X-Session-Id"], r

    def handoff(self, sid, body=GOOD, **kw):
        headers = {"X-Session-Id": sid} if sid else {}
        return self.request("POST", "/handoff", json=body, headers=headers, **kw)

    def admin(self, url, token=TOKEN, **params):
        headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
        return self.request("GET", url, params=params, headers=headers)

    def rows(self, sql):
        con = sqlite3.connect(self.db)
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()


def escalated_session(api, message=EMERGENCY_EN):
    sid, r = api.chat(message)
    assert r.json()["status"] == "escalated"
    return sid


def test_escalated_session_stores_a_handoff_with_kind_and_language(tmp_path):
    api = Api(tmp_path)
    sid = escalated_session(api)
    r = api.handoff(sid)
    assert r.status_code == 201
    body = r.json()
    assert set(body) == {"id", "session_id", "created_at"} and body["session_id"] == sid
    assert api.rows("SELECT session_id, summary, contact, kind, language FROM handoffs") == [
        (sid, GOOD["summary"], GOOD["contact"], "emergency", "en")]


def test_kind_is_that_of_the_latest_escalated_turn(tmp_path):
    api = Api(tmp_path)
    sid = escalated_session(api)
    api.chat(DOSE, sid)
    api.chat(ANSWERED, sid)  # a later answered turn does not hide the escalation
    assert api.handoff(sid).status_code == 201
    assert api.rows("SELECT kind FROM handoffs") == [("dose",)]


def test_session_without_an_escalated_turn_is_refused(tmp_path):
    api = Api(tmp_path)
    sid, _ = api.chat(ANSWERED)
    r = api.handoff(sid)
    assert r.status_code == 409 and r.json() == {"error": "no escalated question in this session"}
    assert api.rows("SELECT COUNT(*) FROM handoffs") == [(0,)]


def test_out_of_scope_session_is_refused(tmp_path):
    api = Api(tmp_path, FakeLLM({"answer": "", "status": "out_of_scope", "citations": []}))
    sid, r = api.chat("capital of france")
    assert r.json()["status"] == "out_of_scope"
    assert api.handoff(sid).status_code == 409


def test_unknown_session_is_409_and_missing_or_bad_header_is_400(tmp_path):
    api = Api(tmp_path)
    assert api.handoff("never-seen").status_code == 409
    assert api.handoff(None).status_code == 400
    assert api.handoff("bad id!").status_code == 400
    assert api.rows("SELECT COUNT(*) FROM handoffs") == [(0,)]


def test_sixth_handoff_in_a_session_is_refused(tmp_path):
    api = Api(tmp_path)
    sid = escalated_session(api)
    assert [api.handoff(sid).status_code for _ in range(5)] == [201] * 5
    r = api.handoff(sid)
    assert r.status_code == 409 and "5 hand-offs" in r.json()["error"]
    assert api.rows("SELECT COUNT(*) FROM handoffs") == [(5,)]


def test_validation_limits(tmp_path):
    api = Api(tmp_path)
    sid = escalated_session(api)
    bad = [
        {"summary": "", "contact": "0100000000"}, {"summary": "   ", "contact": "0100000000"},
        {"summary": "x" * 2001, "contact": "0100000000"}, {"summary": "ok", "contact": "ab"},
        {"summary": "ok", "contact": "  a  "}, {"summary": "ok", "contact": "x" * 201}, {"summary": "ok"},
        {"contact": "0100000000"}, {"summary": 5, "contact": "0100000000"},
        {"summary": "ok", "contact": "01\x00000000"}, {"summary": "o\x07k", "contact": "0100000000"},
        {"summary": "ok", "contact": "010\n000000"},
    ]
    for body in bad:
        r = api.handoff(sid, body)
        assert r.status_code == 400 and "error" in r.json(), body
    assert api.handoff(sid, {"summary": "x" * 2000, "contact": "x" * 200}).status_code == 201
    assert api.handoff(sid, {"summary": "a\nb\tc", "contact": "abc"}).status_code == 201
    assert api.handoff(sid, [1]).status_code == 400
    lone = api.request("POST", "/handoff", content=br'{"summary": "ok\ud800", "contact": "abc"}', headers={"X-Session-Id": sid})
    assert lone.status_code == 400
    assert api.request("POST", "/handoff", content=b"{nope", headers={"X-Session-Id": sid}).status_code == 400
    assert api.rows("SELECT COUNT(*) FROM handoffs") == [(2,)]


def test_oversized_body_is_413(tmp_path):
    api = Api(tmp_path)
    sid = escalated_session(api)
    r = api.handoff(sid, {"summary": "x" * 20000, "contact": "abc"})
    assert r.status_code == 413 and r.json() == {"error": "request body too large"}
    assert api.rows("SELECT COUNT(*) FROM handoffs") == [(0,)]


def test_arabic_round_trips_and_is_labelled(tmp_path):
    api = Api(tmp_path)
    sid = escalated_session(api, EMERGENCY_AR)
    body = {"summary": "ابني ابتلع أقراصاً الساعة الثالثة", "contact": "٠١٠٠٠٠٠٠٠٠٠ أو test@example.invalid"}
    assert api.handoff(sid, body).status_code == 201
    (item,) = api.admin("/admin/handoffs").json()["items"]
    assert item["summary"] == body["summary"] and item["contact"] == body["contact"]
    assert (item["kind"], item["language"]) == ("emergency", "ar")


def test_handoffs_survive_session_pruning(tmp_path):
    store = SessionStore(tmp_path / "p.db", max_sessions=1)
    AnalyticsLog(store)  # the app creates both tables
    log = HandoffLog(store)
    store.add_turn("s1", "q", ESCALATION_TEXT["dose"]["en"], "escalated")
    log.add("s1", "summary", "abc")
    store.add_turn("s2", "q", "a", "answered")  # prunes s1
    assert store.turns("s1") == []
    assert log.listing()["count"] == 1
    store.close()


def test_a_failed_write_is_500_and_chat_still_works(tmp_path):
    api = Api(tmp_path)
    sid = escalated_session(api)
    con = sqlite3.connect(api.db)
    con.execute("DROP TABLE handoffs")
    con.commit()
    con.close()
    r = api.handoff(sid)
    assert r.status_code == 500 and "could not be saved" in r.json()["error"]
    _, again = api.chat(ANSWERED, sid)
    assert again.status_code == 200


def test_analytics_and_handoffs_endpoint_list_it_with_its_session(tmp_path):
    api = Api(tmp_path)
    sid = escalated_session(api)
    other = escalated_session(api, DOSE)
    api.handoff(sid)
    api.handoff(other, {"summary": "second", "contact": "0100000000"})
    for rep in (api.admin("/admin/analytics").json()["handoffs"], api.admin("/admin/handoffs").json()):
        assert rep["count"] == 2
        assert [i["session_id"] for i in rep["items"]] == [other, sid]  # newest first
        assert set(rep["items"][0]) == {"id", "session_id", "created_at", "kind", "language", "summary", "contact"}
        assert rep["items"][0]["kind"] == "dose" and rep["items"][1]["kind"] == "emergency"
    assert len(api.admin("/admin/handoffs", limit=1).json()["items"]) == 1
    assert api.admin("/admin/handoffs", limit=1).json()["count"] == 2
    assert api.admin("/admin/handoffs", until="2000-01-01").json() == {"count": 0, "items": []}
    assert api.admin("/admin/analytics", since="2999-01-01").json()["handoffs"]["count"] == 0
    assert api.admin("/admin/handoffs", limit=0).status_code == 400
    assert "summary" in api.admin("/admin/analytics").json()  # existing fields stay


def test_admin_handoffs_needs_the_token(tmp_path):
    api = Api(tmp_path)
    assert api.admin("/admin/handoffs", token=None).status_code == 401
    assert api.admin("/admin/handoffs", token="wrong").status_code == 401
    assert Api(tmp_path / "off", admin_token="").admin("/admin/handoffs").status_code == 503


def test_chat_contract_is_unchanged(tmp_path):
    _, r = Api(tmp_path).chat(EMERGENCY_EN)
    assert set(r.json()) == {"answer", "citations", "status", "language"}
    assert r.json()["answer"] == ESCALATION_TEXT["emergency"]["en"]


# --- the chat page ---

PAGE = (Path(__file__).parent.parent / "medrag" / "static" / "index.html").read_text(encoding="utf-8")


def test_page_offers_the_form_only_for_escalated_after_the_answer():
    gate = re.search(r'status === "escalated"', PAGE)
    assert gate and PAGE.count("handoffForm(") == 2  # one definition, one call
    call = PAGE.index("handoffForm(", PAGE.index("function handoffForm") + 20)
    assert PAGE.rindex('status === "escalated"', 0, call) > PAGE.rindex("bot.textContent = data.answer", 0, call)
    assert PAGE.index("bot.textContent = data.answer") < call  # the reply is rendered before the form
    assert "/handoff" in PAGE and "autofocus" not in PAGE and ".focus(" not in PAGE
    assert "innerHTML" not in PAGE


def test_page_form_is_arabic_for_ar_and_mixed_never_autofocused_and_capped():
    assert 'language === "ar" || language === "mixed"' in PAGE
    assert "رقم الهاتف" in PAGE and "ملخص" in PAGE and "textContent" in PAGE
    assert "SUMMARY_MAX = 2000" in PAGE and "X-Session-Id" in PAGE
