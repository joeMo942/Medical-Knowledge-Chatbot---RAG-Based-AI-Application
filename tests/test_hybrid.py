"""Hybrid search on a fake embedder (no torch, no model): fusion, the index cache, thinning, Trade Name expansion,
Alternatives, the no-match gate, and the app wiring (/health 503 until loaded, retrieval off the event loop)."""
import asyncio
import json
import threading
import zlib

import httpx
import pytest
from synthetic import FakeLLM, answered
from test_drugs import ALL, BIO, HEAD, PAIN, PPI, THROAT, products, section

from medrag.app import create_app
from medrag.config import Settings
from medrag.embedding import Encoded, dense_score, sparse_score
from medrag.hybrid import HybridRetriever, kb_fingerprint
from medrag.models import Line, Section
from medrag.retrieval import tokenize

DIM = 64


def _h(token: str, mod: int) -> int:
    return zlib.crc32(token.encode()) % mod


class FakeEmbedder:
    """Bag-of-words hashing: dense = normalised counts in DIM buckets, sparse = token weights. Counts its calls."""

    model_id, revision = "fake/bge-m3", "r1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, max_length=512):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            vec = [0.0] * DIM
            sparse: dict[int, float] = {}
            for tok in tokenize(t):
                vec[_h(tok, DIM)] += 1.0
                sparse[_h(tok, 10_000)] = sparse.get(_h(tok, 10_000), 0.0) + 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append(Encoded(tuple(x / norm for x in vec), sparse))
        return out


def make(sections=ALL, cache_dir=None, files_hash="", min_dense=0.0, embedder=None):
    emb = embedder or FakeEmbedder()
    r = HybridRetriever(sections, lambda: emb, emb.model_id, emb.revision, cache_dir=cache_dir,
                        files_hash=files_hash, min_dense=min_dense)
    r.load()
    return r, emb


def test_scores_of_the_fake_embedder_behave():
    e = FakeEmbedder().encode(["Brufen Ibuprofen", "Brufen", "Controloc"])
    assert dense_score(e[0].dense, e[1].dense) > dense_score(e[0].dense, e[2].dense)
    assert sparse_score(e[0].sparse, e[1].sparse) > 0 == sparse_score(e[1].sparse, e[2].sparse)


def test_search_before_load_is_refused_and_ready_flips_after():
    emb = FakeEmbedder()
    r = HybridRetriever(ALL, lambda: emb, emb.model_id)
    assert not r.ready
    with pytest.raises(RuntimeError):
        r.search("Controloc", 3)
    r.load()
    assert r.ready and r.search("Controloc", 3)[0] is PPI


def test_fused_search_finds_the_section_and_returns_whole_sections():
    r, _ = make()
    assert r.search("active ingredient of Controloc", 2)[0] is PPI
    assert r.search("Humira", 1)[0] is BIO


def test_a_question_with_the_trade_name_finds_a_section_that_only_says_the_ingredient():
    only_generic = Section("KB-07", "Misc", "Notes", "class", (Line("Pantoprazole is given before breakfast."),))
    filler = [Section("KB-0%d" % n, "T", f"S{n}", "class", (Line(f"unrelated text number {n} about asthma"),)) for n in range(1, 6)]
    r, _ = make([PPI, only_generic, *filler])
    got = r.search("Controloc", 2)
    assert only_generic in got  # found through the ingredient the products table gives for Controloc


def test_expansion_adds_trade_names_for_an_ingredient_question():
    r, emb = make()
    r.search("which products contain Pantoprazole", 3)
    queries = emb.calls[-1]
    assert len(queries) == 2 and "Controloc" in queries[1] and "Controloc" not in queries[0]


def test_alternatives_get_every_section_with_the_ingredient_not_just_top_k():
    r, _ = make()
    got = r.search("ايه البدائل التجارية لـ Brufen؟", 4)
    assert [s.section_id for s in got] == [HEAD.section_id, PAIN.section_id]
    text = "\n".join(s.render() for s in got)
    for name in ("Brufen", "Nurofen", "Flabu", "Spididol"):
        assert name in text
    assert "Cataflam" not in text and "Cetafen" not in text and "Controloc" not in text  # nothing of another drug
    assert HEAD.rows != got[0].rows and len(HEAD.rows) == 4  # the original Section is untouched


def test_alternatives_with_a_further_question_keep_whole_sections():
    r, _ = make()
    got = r.search("alternatives to Brufen and their side effects", 4)
    assert got[0] is HEAD  # whole Section, so its lines are there
    assert all(s.section_id in {HEAD.section_id, PAIN.section_id, THROAT.section_id} for s in got)


def test_franco_arabic_and_ingredient_forms_of_the_question_are_alternative_questions():
    r, _ = make()
    assert r.search("fe bedail tegareya le Brufen?", 4)[0].section_id == HEAD.section_id
    got = r.search("Which trade names contain Ibuprofen", 4)
    assert got and got[0].section_id == HEAD.section_id


def test_thinning_moves_a_duplicate_section_behind_other_candidates():
    dup1 = section("KB-03", "Pain", "NSAIDs", [("Brufen", "Ibuprofen"), ("Nurofen", "Ibuprofen"), ("Ibuflam", "Ibuprofen")],
                   lines=["NSAIDs relieve pain and inflammation."])
    dup2 = section("KB-04", "Throat", "NSAIDs", [("Brufen", "Ibuprofen"), ("Nurofen", "Ibuprofen"), ("Ibuflam", "Ibuprofen")],
                   lines=["NSAIDs relieve pain and inflammation."])
    other = section("KB-09", "Arthritis", "Analgesics", [("Panadol", "Paracetamol")], lines=["Pain relief for arthritis."])
    r, _ = make([dup1, dup2, other])
    got = r.search("pain NSAIDs Brufen relief", 2)
    assert got == [dup1, other]  # dup2 repeats dup1 entirely, so it does not take the second place
    assert r.search("pain NSAIDs Brufen relief", 3)[2] is dup2  # but it still fills a free slot


def test_no_lexical_match_and_low_dense_returns_nothing():
    r, _ = make(min_dense=0.99)
    assert r.search("zzzz qqqq", 4) == []
    assert r.search("Controloc", 4)  # a lexical hit is never gated


def test_index_cache_is_reused_and_keyed_on_kb_and_model(tmp_path):
    r1, e1 = make(cache_dir=tmp_path, files_hash="a")
    assert r1.index_source == "built" and len(e1.calls[0]) == len(ALL)
    r2, e2 = make(cache_dir=tmp_path, files_hash="a")
    assert r2.index_source == "cache" and e2.calls == []  # nothing was embedded at start
    assert r2.search("Controloc", 1)[0] is PPI
    r3, _ = make(cache_dir=tmp_path, files_hash="b")  # a KB file changed
    assert r3.index_source == "built"
    changed = [*ALL[:-1], Section(*(*BIO.__dict__.values(),)[:3], "class", (Line("Adalimumab changed."),))]
    r4, _ = make(changed, cache_dir=tmp_path, files_hash="b")  # same files hash, different Section text
    assert r4.index_source == "built"
    other_model = FakeEmbedder()
    other_model.model_id = "fake/other"
    r5, _ = make(changed, cache_dir=tmp_path, files_hash="b", embedder=other_model)
    assert r5.index_source == "built"
    assert len(list(tmp_path.glob("index-*.json"))) == 1  # stale indexes are removed
    (next(tmp_path.glob("index-*.json"))).write_text("{not json")
    r6, _ = make(changed, cache_dir=tmp_path, files_hash="b", embedder=other_model)
    assert r6.index_source == "built"  # an unreadable cache is rebuilt, not fatal


def test_kb_fingerprint_changes_with_any_text():
    assert kb_fingerprint(ALL) == kb_fingerprint(list(ALL))
    assert kb_fingerprint(ALL) != kb_fingerprint(ALL[:-1])
    assert kb_fingerprint(ALL, "x") != kb_fingerprint(ALL, "y")


# ------------------------------------------------------------------ the app

class SlowRetriever:
    """The Retriever protocol plus a slow load(): what the app waits for."""

    model_id, revision = "fake/bge-m3", "r1"

    def __init__(self, inner):
        self.inner, self.ready, self.gate = inner, False, threading.Event()
        self.search_threads: list[int] = []

    def load(self):
        self.gate.wait(5)
        self.ready = True

    def search(self, query, k):
        self.search_threads.append(threading.get_ident())
        return self.inner.search(query, k)


def test_health_and_chat_are_503_until_the_retriever_is_loaded_then_report_the_model():
    inner, _ = make()
    slow = SlowRetriever(inner)
    app = create_app(ALL, FakeLLM(answered()), Settings(), retriever=slow)

    async def go():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                h = await c.get("/health")
                assert h.status_code == 503 and h.json()["status"] == "error"
                assert h.json()["embedding"] == "fake/bge-m3"
                chat = await c.post("/chat", json={"message": "What is the active ingredient in Cataflam?"})
                assert chat.status_code == 500 and set(chat.json()) == {"error"}  # the contract's error status
                slow.gate.set()
                for _ in range(50):
                    await asyncio.sleep(0.05)
                    h = await c.get("/health")
                    if h.status_code == 200:
                        break
                assert h.status_code == 200
                assert h.json() == {"status": "ok", "llm": "fake-model", "embedding": "fake/bge-m3", "embedding_revision": "r1"}
                chat = await c.post("/chat", json={"message": "What is the active ingredient in Cataflam?"})
                assert chat.status_code == 200
    asyncio.run(go())
    assert slow.search_threads and threading.get_ident() not in slow.search_threads  # off the event loop


def test_a_failed_load_keeps_health_red():
    class Broken(SlowRetriever):
        def load(self):
            raise RuntimeError("no model")

    app = create_app(ALL, FakeLLM(), Settings(), retriever=Broken(None))

    async def go():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.2)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                return (await c.get("/health")).status_code
    assert asyncio.run(go()) == 503


def test_the_cache_file_holds_no_pickle(tmp_path):
    make(cache_dir=tmp_path)
    data = json.loads(next(tmp_path.glob("index-*.json")).read_text())
    assert data["model"] == "fake/bge-m3" and len(data["docs"]) == len(ALL)


# ------------------------------------------------------------------ citations naming only the start of a section_id

def test_a_citation_naming_only_the_kb_id_is_resolved_when_one_retrieved_section_fits():
    from medrag.answering import build_citations

    one, other = PAIN, PPI  # KB-03 and KB-02: one retrieved Section each
    got = build_citations([{"section_id": "KB-03", "snippet": "Brufen – Ibuprofen"}], [one, other], "q")
    assert [(c.doc_id, c.snippet) for c in got] == [("KB-03", "Brufen – Ibuprofen")]


def test_an_ambiguous_prefix_needs_the_snippet_to_settle_it_and_an_invented_id_is_still_dropped():
    from medrag.answering import build_citations

    only_head = section("KB-03", "Headache", "NSAIDs", [("Flabu", "Ibuprofen")])
    secs = [PAIN, only_head]  # two KB-03 Sections
    assert [c.section for c in build_citations([{"section_id": "KB-03", "snippet": "Flabu – Ibuprofen"}], secs, "q")] == ["NSAIDs"]
    assert build_citations([{"section_id": "KB-03", "snippet": "Brufen – Ibuprofen"}], [PAIN, THROAT_LIKE_PAIN], "q") == []
    assert build_citations([{"section_id": "KB-77", "snippet": "Brufen – Ibuprofen"}], secs, "q") == []
    assert build_citations([{"section_id": "KB-0", "snippet": "Brufen – Ibuprofen"}], secs, "q") == []  # not a KB ID
    # a garbled path (the heading translated) still names the Section that holds the snippet
    got = build_citations([{"section_id": "KB-03 › Headache › NSAIDs", "snippet": "Flabu – Ibuprofen"}], secs, "q")
    assert [c.section for c in got] == ["NSAIDs"] and got[0].title == "Headache"


THROAT_LIKE_PAIN = section("KB-03", "Fever", "NSAIDs", [("Brufen", "Ibuprofen")])


def test_a_named_drug_keeps_its_sections_ahead_of_noise_words():
    otc = [Section("KB-0%d" % n, "T%d" % n, "OTC vs Prescription", "otc", (Line("OTC or prescription: some OTC products."),)) for n in range(1, 5)]
    r, _ = make([*otc, PPI])
    assert r.search("OTC Controloc", 2)[0] is PPI  # the Section that lists Controloc, though four others repeat "OTC"


def test_prefix_resolution_never_redirects_an_invented_or_unretrieved_id():
    from medrag.answering import build_citations

    head = section("KB-03", "Headache & Migraine", "NSAIDs", [("Brufen", "Ibuprofen"), ("Spididol", "Ibuprofen (as ibuprofen arginine)")])
    para = section("KB-03", "Pain & Fever", "Paracetamol", [("Panadol", "Paracetamol")])
    known = frozenset({head.section_id, para.section_id})  # the whole KB; only `head` was retrieved
    # an invented path and a snippet no retrieved Section holds: dropped, not "repaired" to another row
    assert build_citations([{"section_id": "KB-03 > Malaria", "snippet": "Chloroquine treats malaria"}], [head], "q", known) == []
    # the exact ID of a real Section that was NOT retrieved: dropped, even though its KB ID matches a retrieved one
    assert build_citations([{"section_id": para.section_id, "snippet": "Panadol – Paracetamol"}], [head], "q", known) == []
    assert build_citations([{"section_id": para.section_id, "snippet": "Brufen – Ibuprofen"}], [head], "q", known) == []
    # the regression this resolver exists for still works: only the KB ID, and a verbatim snippet
    got = build_citations([{"section_id": "KB-03", "snippet": "Brufen – Ibuprofen"}], [head], "q", known)
    assert [(c.section, c.snippet) for c in got] == [("NSAIDs", "Brufen – Ibuprofen")]


# ------------------------------------------------------------------ Alternatives wordings, end to end

ALT_QUESTIONS = ["bdayel brufen", "badayel brufen eh", "eh el bdayel bta3et brufen", "بدايله ايه Brufen",
                 "what can replace Brufen", "fe bedail tegareya le Brufen?", "ايه البدائل التجارية لـ Brufen؟",
                 "What are the alternatives to Brufen?", "any substitute for Brufen"]


@pytest.mark.parametrize("q", ALT_QUESTIONS)
def test_every_alternatives_wording_passes_the_support_check_and_gets_the_name_sections(q):
    from medrag.support import SupportCheck

    assert SupportCheck(ALL).supported(q, [HEAD])  # asking for alternatives of a listed drug is covered
    r, _ = make()
    assert r.search(q, 4)[0].section_id == HEAD.section_id


@pytest.mark.parametrize("q", ALT_QUESTIONS)
def test_alternatives_wordings_are_answered_through_the_pipeline(q):
    from synthetic import FakeLLM

    reply = {"answer": "Nurofen, Flabu.", "status": "answered",
             "citations": [{"section_id": HEAD.section_id, "snippet": "Flabu – Ibuprofen"}]}
    inner, _ = make()
    app = create_app(ALL, FakeLLM(reply), Settings(), retriever=inner)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            return (await c.post("/chat", json={"message": q})).json()
    body = asyncio.run(go())
    assert body["status"] == "answered" and body["citations"][0]["snippet"] == "Flabu – Ibuprofen"
