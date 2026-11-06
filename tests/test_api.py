"""API tests: fake LLM, small synthetic Sections; no Ollama and no KB files needed."""
import asyncio
import json
import time

import httpx
import pytest
from synthetic import HEADER_ROW, NSAIDS, OVERVIEW, ROW, SECTIONS, FakeLLM, answered

from medrag.answering import NOT_FOUND
from medrag.app import create_app
from medrag.config import EMBEDDING_PLACEHOLDER, Settings
from medrag.llm import LLMError, LLMTruncated


def make_app(llm, **settings):
    return create_app(SECTIONS, llm, Settings(**settings))


class SyncClient:
    """Blocking calls into the app through httpx.ASGITransport.

    Not Starlette's TestClient: with the pinned Starlette 1.10 + httpx 0.28 it warns
    that its httpx support is deprecated (it wants the separate httpx2 package).
    Lifespan (the Ollama warm-up) is not run, which these tests do not need."""

    def __init__(self, app):
        self.app = app

    def request(self, method, url, **kw):
        async def go():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
                return await c.request(method, url, **kw)
        return asyncio.run(go())

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)


def client(llm, **settings):
    return SyncClient(make_app(llm, **settings))


Q = {"message": "What is the active ingredient in Cataflam?"}


def test_response_has_exactly_the_contract_fields():
    r = client(FakeLLM(answered())).post("/chat", json=Q)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "citations", "status", "language"}
    assert body["status"] == "answered" and body["language"] == "en"
    assert body["citations"] == [
        {"doc_id": "KB-04", "title": "Pain & Arthritis", "section": "NSAIDs", "snippet": ROW}]


def test_session_id_generated_then_reused():
    c = client(FakeLLM(answered()))
    sid = c.post("/chat", json=Q).headers["X-Session-Id"]
    assert sid
    r = c.post("/chat", json={**Q, "session_id": sid})
    assert r.headers["X-Session-Id"] == sid
    assert "session_id" not in r.json()
    assert c.post("/chat", json=Q).headers["X-Session-Id"] != sid


def test_session_id_from_request_header_is_reused():
    r = client(FakeLLM(answered())).post("/chat", json=Q, headers={"X-Session-Id": "from-header-1"})
    assert r.headers["X-Session-Id"] == "from-header-1"


@pytest.mark.parametrize("bad", [
    "Cataflam contains diclofenac (made up)",   # not in the Section
    "KB-04 › Pain & Arthritis",                  # an invented path, not KB text
    "Asthma",                                    # not in this Section
    "NSAIDs",                                    # the heading, never a candidate
    HEADER_ROW,                                  # a table header row
    f"{HEADER_ROW}\n{ROW}",                      # multi-line with the header row
    f"{ROW}\nVoltaren – Diclofenac sodium",      # several rows
    "C",                                         # one character
    "",                                          # empty
])
def test_invalid_snippet_is_repaired_to_one_row(bad):
    r = client(FakeLLM(answered(snippet=bad))).post("/chat", json=Q)
    (cit,) = r.json()["citations"]
    assert cit["snippet"] == ROW


@pytest.mark.parametrize("good", ["Used for pain and inflammation.", "Diclofenac potassium", ROW])
def test_valid_snippet_within_one_line_or_row_is_kept(good):
    r = client(FakeLLM(answered(snippet=f" {good} "))).post("/chat", json=Q)
    assert r.json()["citations"][0]["snippet"] == good


def test_citation_text_comes_from_the_section_not_the_model():
    reply = answered()
    reply["citations"][0].update(title="Invented", doc_id="KB-99", section="Invented")
    (cit,) = client(FakeLLM(reply)).post("/chat", json=Q).json()["citations"]
    assert (cit["doc_id"], cit["title"], cit["section"]) == ("KB-04", "Pain & Arthritis", "NSAIDs")


def test_answered_without_valid_citation_is_downgraded():
    for bad in ([], [{"section_id": "KB-77 › nowhere", "snippet": "x"}]):
        reply = {"answer": "Diclofenac.", "status": "answered", "citations": bad}
        body = client(FakeLLM(reply)).post("/chat", json=Q).json()
        assert body["status"] == "out_of_scope" and body["citations"] == []
        assert body["answer"] == NOT_FOUND["en"]


def test_out_of_scope_uses_fixed_message_never_model_text():
    reply = {"answer": "Take 500 mg every 6 hours.", "status": "out_of_scope",
             "citations": [{"section_id": NSAIDS.section_id, "snippet": ROW}]}
    body = client(FakeLLM(reply)).post("/chat", json=Q).json()
    assert body == {"answer": NOT_FOUND["en"], "citations": [], "status": "out_of_scope", "language": "en"}
    ar = client(FakeLLM(reply)).post("/chat", json={"message": "كم جرعة Cataflam"}).json()
    assert ar["answer"] == NOT_FOUND["ar"] and ar["language"] == "ar"


def test_no_lexical_match_skips_the_llm():
    llm = FakeLLM(answered())
    body = client(llm).post("/chat", json={"message": "zzzz qqqq"}).json()
    assert body["status"] == "out_of_scope" and body["citations"] == [] and llm.calls == []


def test_prompt_has_only_render_text_and_section_ids_not_path():
    llm = FakeLLM(answered())
    client(llm).post("/chat", json={"message": "Asthma overview"})
    _, user = llm.calls[0]
    assert OVERVIEW.render() in user and OVERVIEW.section_id in user
    assert "[overview, no heading in source]" not in user and OVERVIEW.path not in user


def test_arabic_question_gets_language_ar():
    body = client(FakeLLM(answered())).post("/chat", json={"message": "ما هو Cataflam؟"}).json()
    assert body["language"] == "ar"


def test_reply_cut_off_at_the_output_cap_is_out_of_scope_not_500():
    for message, lang in (("Tell me everything about Cataflam", "en"), ("كل حاجة عن Cataflam", "ar")):
        r = client(FakeLLM(exc=LLMTruncated("cut off"))).post("/chat", json={"message": message})
        assert r.status_code == 200
        assert r.json() == {"answer": NOT_FOUND[lang], "citations": [], "status": "out_of_scope", "language": lang}


def test_prompt_asks_for_a_concise_answer():
    llm = FakeLLM(answered())
    client(llm).post("/chat", json=Q)
    system, _ = llm.calls[0]
    assert "120 words" in system and "only what was asked" in system


def test_llm_exception_is_http_500_with_error():
    r = client(FakeLLM(exc=LLMError("boom"))).post("/chat", json=Q)
    assert r.status_code == 500 and set(r.json()) == {"error"}


def test_slow_llm_is_500_within_budget():
    t0 = time.monotonic()
    r = client(FakeLLM(answered(), delay=5), budget_s=0.2).post("/chat", json=Q)
    assert r.status_code == 500 and "error" in r.json()
    assert time.monotonic() - t0 < 2


async def _burst(app, n):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        async def one():
            t0 = time.monotonic()
            r = await c.post("/chat", json=Q)
            return r, time.monotonic() - t0
        return await asyncio.gather(*(one() for _ in range(n)))


def test_queue_that_cannot_finish_in_budget_fails_fast():
    # One call takes ~1 s, budget 2.5 s: two requests fit, the rest must be refused at once.
    app = make_app(FakeLLM(answered(), delay=1.0), budget_s=2.5, llm_est_s=1.0, llm_concurrency=1)
    results = asyncio.run(_burst(app, 5))
    ok = [r for r, _ in results if r.status_code == 200]
    refused = [(r, t) for r, t in results if r.status_code == 500]
    assert len(ok) == 2 and len(refused) == 3
    for r, t in refused:
        assert set(r.json()) == {"error"} and "busy" in r.json()["error"]
        assert t < 0.5


def test_client_disconnect_cancels_the_llm_call():
    llm = FakeLLM(answered(), delay=10)
    app = make_app(llm, budget_s=20)

    async def run():
        inbox: asyncio.Queue = asyncio.Queue()
        await inbox.put({"type": "http.request", "body": json.dumps(Q).encode(), "more_body": False})
        scope = {"type": "http", "method": "POST", "path": "/chat", "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json")], "http_version": "1.1",
                 "scheme": "http", "server": ("t", 80), "client": ("c", 1)}

        async def send(_):
            pass

        task = asyncio.create_task(app(scope, inbox.get, send))
        await asyncio.sleep(0.5)
        await inbox.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, 3)

    asyncio.run(run())
    assert llm.cancelled


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "   "}, {"message": 5}, {"session_id": "a"}])
def test_missing_or_empty_message_is_400(payload):
    r = client(FakeLLM(answered())).post("/chat", json=payload)
    assert r.status_code == 400 and set(r.json()) == {"error"}


def test_non_json_body_is_400():
    r = client(FakeLLM(answered())).post("/chat", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and set(r.json()) == {"error"}


def test_over_long_message_is_400_and_llm_not_called():
    llm = FakeLLM(answered())
    c = client(llm, max_message_chars=50)
    assert c.post("/chat", json={"message": "a" * 51}).status_code == 400
    assert c.post("/chat", json={"message": "Cataflam " * 5000}).status_code == 400  # body-size cap
    assert c.post("/chat", json={"message": "Cataflam"}).status_code == 200
    assert len(llm.calls) == 1


def test_longest_legal_message_fits_the_body_cap_even_ascii_escaped():
    # 2000 emoji sent with ensure_ascii are 12 bytes each ("\ud83d\ude00").
    llm = FakeLLM(answered())
    body = json.dumps({"message": "😀" * 1990 + " Cataflam", "session_id": "s" * 128})
    assert len(body) > 2000 * 12
    r = client(llm).post("/chat", content=body.encode(), headers={"content-type": "application/json"})
    assert r.status_code == 200 and len(llm.calls) == 1


def test_huge_body_is_rejected_without_reading_it_all():
    r = client(FakeLLM(answered())).post(
        "/chat", content=b'{"message": "' + b"a" * 5_000_000 + b'"}', headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json() == {"error": "request body too large"}


def test_lone_surrogate_is_400_not_500():
    raw = b'{"message": "Cataflam \\ud800"}'
    r = client(FakeLLM(answered())).post("/chat", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400 and set(r.json()) == {"error"}


def test_unknown_route_and_wrong_method_return_json_error():
    c = client(FakeLLM())
    r = c.get("/nope")
    assert r.status_code == 404 and set(r.json()) == {"error"}
    r = c.get("/chat")
    assert r.status_code == 405 and set(r.json()) == {"error"}


def test_health_reports_real_names():
    r = client(FakeLLM()).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "llm": "fake-model", "embedding": EMBEDDING_PLACEHOLDER}


def test_health_is_503_when_the_backend_is_not_ready():
    class Down(FakeLLM):
        async def ready(self):
            return False

    r = client(Down()).get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "error", "llm": "fake-model", "embedding": EMBEDDING_PLACEHOLDER}


def test_chat_page_is_a_working_chat_ui():
    r = client(FakeLLM()).get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    assert 'dir="auto"' in html and "<form" in html
    assert 'fetch("/chat"' in html and "X-Session-Id" in html and "citations" in html
