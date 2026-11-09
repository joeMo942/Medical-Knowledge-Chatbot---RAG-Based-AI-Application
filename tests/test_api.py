"""API tests: fake LLM, small synthetic Sections; no Ollama and no KB files needed."""
import asyncio
import json
import re
import time
from pathlib import Path

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
    ar = client(FakeLLM(reply)).post("/chat", json={"message": "ما هو Cataflam؟"}).json()
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
        r = client(FakeLLM(exc=LLMTruncated(990, 1000))).post("/chat", json={"message": message})
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
    # Tiny budget, gate thresholds lowered so the call starts and the timeout fires.
    r = client(FakeLLM(answered(), delay=5), budget_s=0.2, llm_est_s=0.05, llm_margin_s=0.0,
               min_output_tokens=1).post("/chat", json=Q)
    assert r.status_code == 500 and r.json() == {"error": "the request timed out"}
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
    # One call takes ~1 s, budget 3.5 s. A request on an idle server makes a routing call first;
    # the others arrive while it runs, skip theirs (see routing.route_message) and queue one answer
    # call each. Some requests finish, the rest are refused as "busy", and none outlives the budget.
    llm = FakeLLM(answered(), delay=1.0)
    app = make_app(llm, budget_s=3.5, llm_est_s=1.0, llm_concurrency=1, llm_margin_s=0.1, min_output_tokens=1)
    results = asyncio.run(_burst(app, 5))
    ok = [r for r, _ in results if r.status_code == 200]
    refused = [(r, t) for r, t in results if r.status_code == 500]
    assert len(ok) >= 2 and len(ok) + len(refused) == 5
    assert len(llm.route_calls) == 1  # only the request that found the server idle was routed
    for r, t in refused:
        assert set(r.json()) == {"error"} and "busy" in r.json()["error"]
        assert t < 3.5


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


def test_huge_chunked_body_without_content_length_is_cut_off_while_streaming():
    sent = {"chunks": 0}
    total = 100  # 100 x 64 KB = 6.4 MB if it were read to the end

    async def body():
        yield b'{"message": "'
        for _ in range(total):
            sent["chunks"] += 1
            yield b"a" * 65536
        yield b'"}'

    async def go():
        transport = httpx.ASGITransport(app=make_app(FakeLLM(answered())))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post("/chat", content=body(), headers={"content-type": "application/json"})

    r = asyncio.run(go())
    assert "content-length" not in {k.lower() for k in r.request.headers}  # really chunked
    assert r.status_code == 400 and r.json() == {"error": "request body too large"}
    assert sent["chunks"] < 5  # stopped at the cap (~28 KB), not after 6.4 MB


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


# --- X-Medrag-Usage header (tokens and LLM calls); off by default, the body never changes ---

def uclient(llm, **settings):
    return client(llm, usage_header=True, **settings)


def usage_of(r):
    return json.loads(r.headers["X-Medrag-Usage"])


def test_usage_header_is_off_by_default():
    r = client(FakeLLM(answered(), tokens=(321, 45))).post("/chat", json=Q)
    assert r.status_code == 200 and "X-Medrag-Usage" not in r.headers


def test_usage_header_reports_tokens_and_calls_without_touching_the_body():
    r = uclient(FakeLLM(answered(), tokens=(321, 45))).post("/chat", json=Q)
    # The routing call and the answer call, each reporting 321 + 45 tokens.
    assert usage_of(r) == {"prompt_tokens": 642, "completion_tokens": 90, "llm_calls": 2}
    assert set(r.json()) == {"answer", "citations", "status", "language"}


def test_usage_header_counts_only_the_routing_call_when_the_answer_call_is_skipped():
    r = uclient(FakeLLM(answered(), tokens=(321, 45))).post("/chat", json={"message": "zzzz qqqq"})
    assert r.status_code == 200 and r.json()["status"] == "out_of_scope"
    assert usage_of(r) == {"prompt_tokens": 321, "completion_tokens": 45, "llm_calls": 1}


def test_usage_header_counts_zero_when_a_rule_escalates_without_any_model_call():
    r = uclient(FakeLLM(answered(), tokens=(321, 45))).post("/chat", json={"message": "What is the dose of Brufen?"})
    assert r.status_code == 200 and r.json()["status"] == "escalated"
    assert usage_of(r) == {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}


def test_usage_header_counts_a_truncated_call_with_its_prompt_tokens():
    r = uclient(FakeLLM(exc=LLMTruncated(150, 150, prompt_tokens=900))).post("/chat", json=Q)
    assert r.status_code == 200 and r.json()["status"] == "out_of_scope"
    # routing call (no tokens reported by the fake) + the truncated answer call
    assert usage_of(r) == {"prompt_tokens": 900, "completion_tokens": 150, "llm_calls": 2}


def test_usage_header_unreported_tokens_count_as_zero():
    r = uclient(FakeLLM(answered())).post("/chat", json=Q)
    assert usage_of(r) == {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 2}


def test_usage_header_is_not_shared_between_requests():
    c = uclient(FakeLLM(answered(), tokens=(10, 5)))
    first = c.post("/chat", json=Q).headers["X-Medrag-Usage"]
    assert c.post("/chat", json=Q).headers["X-Medrag-Usage"] == first


def test_usage_header_is_also_sent_on_500_responses():
    r = uclient(FakeLLM(exc=LLMError("boom"))).post("/chat", json=Q)   # the failed call still held the GPU
    assert r.status_code == 500
    assert "X-Medrag-Usage" in r.headers and usage_of(r)["llm_calls"] == 2
    r = uclient(FakeLLM(answered(), delay=1.0), budget_s=0.3, llm_est_s=0.1).post("/chat", json=Q)
    assert r.status_code == 500 and "X-Medrag-Usage" in r.headers


def test_usage_header_on_busy_500_counts_only_the_routing_call():
    # The short routing call still fits; the answer call (minimum 1200 tokens) is refused as busy.
    r = uclient(FakeLLM(answered()), min_output_tokens=1200).post("/chat", json=Q)
    assert r.status_code == 500 and usage_of(r)["llm_calls"] == 1


def test_usage_header_on_a_timeout_500():
    # Same set-up as test_slow_llm_is_500_within_budget: the call starts, the request budget runs out.
    r = uclient(FakeLLM(answered(), delay=5), budget_s=0.2, llm_est_s=0.05, llm_margin_s=0.0,
                min_output_tokens=1).post("/chat", json=Q)
    assert r.status_code == 500 and r.json() == {"error": "the request timed out"}
    # The cancelled call reported no usage, and is not counted as a finished call.
    assert usage_of(r) == {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}


def test_usage_meter_adds_up_several_llm_calls_of_one_request():
    from medrag.gate import LLMGate
    from medrag.usage import start_meter

    async def go():
        meter = start_meter()
        gate = LLMGate(FakeLLM(answered(), tokens=(100, 10)))
        await gate.complete_json("s", "u", {})
        await gate.complete_json("s", "u", {})   # e.g. a classifier call plus the answer call
        return meter

    meter = asyncio.run(go())
    assert (meter.prompt_tokens, meter.completion_tokens, meter.llm_calls) == (200, 20, 2)
    assert json.loads(meter.header_value()) == {"prompt_tokens": 200, "completion_tokens": 20, "llm_calls": 2}


def test_usage_header_can_be_turned_off_and_setting_from_env(monkeypatch):
    r = client(FakeLLM(answered()), usage_header=False).post("/chat", json=Q)
    assert r.status_code == 200 and "X-Medrag-Usage" not in r.headers
    monkeypatch.delenv("MEDRAG_USAGE_HEADER", raising=False)
    assert Settings.from_env().usage_header is False
    monkeypatch.setenv("MEDRAG_USAGE_HEADER", "1")
    assert Settings.from_env().usage_header is True
    monkeypatch.setenv("MEDRAG_USAGE_HEADER", "0")
    assert Settings.from_env().usage_header is False


def test_compose_keeps_the_usage_header_off_by_default():
    # The graded stack is docker compose: the header must be opt-in there too, not only in Settings.
    root = Path(__file__).resolve().parents[1]
    for f in sorted(root.glob("docker-compose*.yml")):
        for value in re.findall(r"MEDRAG_USAGE_HEADER:\s*(\S+)", f.read_text(encoding="utf-8")):
            assert value == "${MEDRAG_USAGE_HEADER:-0}", f"{f.name}: {value}"
    assert re.search(r"MEDRAG_USAGE_HEADER:\s*\$\{MEDRAG_USAGE_HEADER:-0\}",
                     (root / "docker-compose.yml").read_text(encoding="utf-8"))
