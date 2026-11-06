"""API tests: fake LLM, small synthetic Sections; no Ollama and no KB files needed."""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from medrag.app import create_app
from medrag.config import EMBEDDING_PLACEHOLDER, Settings
from medrag.llm import LLMError
from medrag.models import Line, Row, Section, Table

PRODUCTS = Table(
    header=Row.from_cells(["Trade Name", "Active Ingredient"]),
    rows=(Row.from_cells(["Cataflam", "Diclofenac potassium"]), Row.from_cells(["Voltaren", "Diclofenac sodium"])),
    kind="products", trade_name_col=0, active_ingredient_col=1,
)
NSAIDS = Section("KB-04", "Pain & Arthritis", "NSAIDs", "class",
                 (Line("Used for pain and inflammation."), PRODUCTS))
PPI = Section("KB-02", "Peptic Ulcer & GERD", "Proton Pump Inhibitors (PPIs)", "class",
              (Table(Row.from_cells(["Trade Name", "Active Ingredient"]),
                     (Row.from_cells(["Controloc", "Pantoprazole"]),), "products", 0, 1),))
OVERVIEW = Section("KB-01", "Asthma", "Asthma", "overview", (Line("Asthma is a chronic airway disease."),),
                   heading_present=False)
SECTIONS = [NSAIDS, PPI, OVERVIEW]
ROW = "Cataflam – Diclofenac potassium"


class FakeLLM:
    model = "fake-model"

    def __init__(self, reply=None, exc=None, delay=0.0):
        self.reply, self.exc, self.delay = reply, exc, delay
        self.calls = []

    async def complete_json(self, system, user, schema):
        self.calls.append((system, user))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.reply


def answered(snippet=ROW, section_id=NSAIDS.section_id):
    return {"answer": "Diclofenac.", "status": "answered",
            "citations": [{"section_id": section_id, "snippet": snippet}]}


def client(llm, **settings):
    return TestClient(create_app(SECTIONS, llm, Settings(**settings)))


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


def test_non_verbatim_snippet_is_replaced_with_a_section_row():
    r = client(FakeLLM(answered(snippet="Cataflam contains diclofenac (made up)"))).post("/chat", json=Q)
    (cit,) = r.json()["citations"]
    assert cit["snippet"] == ROW
    assert cit["snippet"] in NSAIDS.render()


def test_verbatim_snippet_is_kept():
    r = client(FakeLLM(answered(snippet="Used for pain and inflammation."))).post("/chat", json=Q)
    assert r.json()["citations"][0]["snippet"] == "Used for pain and inflammation."


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


def test_out_of_scope_has_no_citations_even_if_model_cites():
    reply = {"answer": "Not covered.", "status": "out_of_scope",
             "citations": [{"section_id": NSAIDS.section_id, "snippet": ROW}]}
    body = client(FakeLLM(reply)).post("/chat", json=Q).json()
    assert body == {"answer": "Not covered.", "citations": [], "status": "out_of_scope", "language": "en"}


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


def test_llm_exception_is_http_500_with_error():
    r = client(FakeLLM(exc=LLMError("boom"))).post("/chat", json=Q)
    assert r.status_code == 500 and set(r.json()) == {"error"}


def test_slow_llm_is_500_within_budget():
    t0 = time.monotonic()
    r = client(FakeLLM(answered(), delay=5), budget_s=0.2).post("/chat", json=Q)
    assert r.status_code == 500 and "error" in r.json()
    assert time.monotonic() - t0 < 2


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "   "}, {"message": 5}, {"session_id": "a"}])
def test_missing_or_empty_message_is_400(payload):
    r = client(FakeLLM(answered())).post("/chat", json=payload)
    assert r.status_code == 400 and set(r.json()) == {"error"}


def test_non_json_body_is_400():
    r = client(FakeLLM(answered())).post("/chat", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and set(r.json()) == {"error"}


def test_health_reports_real_names():
    r = client(FakeLLM()).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "llm": "fake-model", "embedding": EMBEDDING_PLACEHOLDER}


def test_chat_page_is_served():
    r = client(FakeLLM()).get("/")
    assert r.status_code == 200 and 'dir="auto"' in r.text
