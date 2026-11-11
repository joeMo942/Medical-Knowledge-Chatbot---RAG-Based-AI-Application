"""Admin page API (#11): upload, replace and delete KB Documents in a writable store. Small .docx files made with
python-docx, a temp store, the fake LLM; BM25 for the app tests and a fake embedder for the hybrid path. No KB files needed."""
import base64
import hashlib
import io
import json
import re
import zlib

import pytest
from docx import Document
from synthetic import FakeLLM
from test_api import SyncClient

from medrag.app import create_app
from medrag.config import Settings
from medrag.embedding import Encoded
from medrag.hybrid import HybridRetriever
from medrag.kbmanager import KBManager, KBSnapshot, open_store
from medrag.kbstore import MAX_UPLOAD_BYTES, KBStore
from medrag.manifest import load_manifest
from medrag.retrieval import tokenize
from medrag.splitter import strip_numbering

DISEASE = "1. Asthma"


def make_docx(disease=DISEASE, classes=None, intro=None) -> bytes:
    """A KB Document: a disease heading, then numbered drug-class headings each with a products table."""
    classes = classes if classes is not None else {"SABA": [("Ventolin", "Salbutamol")]}
    doc = Document()
    seen = set()
    for heading in [disease] if isinstance(disease, str) else disease:  # each disease gets the same classes
        if strip_numbering(heading) in seen:  # KB-05 repeats its heading ("1." and "10."): one disease
            continue
        seen.add(strip_numbering(heading))
        doc.add_paragraph(heading)
        if intro:
            doc.add_paragraph(intro)
        for n, (name, rows) in enumerate(classes.items(), start=1):
            doc.add_paragraph(f"{n}. {name}")
            doc.add_paragraph(f"{name} are used for symptom relief.")
            table = doc.add_table(rows=1 + len(rows), cols=2)
            table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Trade Name", "Active Ingredient"
            for i, (trade, ingredient) in enumerate(rows, start=1):
                table.rows[i].cells[0].text, table.rows[i].cells[1].text = trade, ingredient
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def cite_first(user, seen=None):
    """A fake model that answers from the first Section it is shown, citing its first product row."""
    if seen is not None:
        seen.append(user)
    m = re.search(r"\[section_id: (.+?)\]\n(.*?)(?:\n\n|\Z)", user, re.S)
    if not m:
        return {"answer": "nothing", "status": "answered", "citations": []}
    rows = [ln for ln in m.group(2).splitlines() if " – " in ln and not ln.startswith("Trade Name")]
    return {"answer": "From the KB.", "status": "answered", "citations": [{"section_id": m.group(1), "snippet": rows[0]}]}


def seed_dir(tmp_path):
    """A read-only KB folder in the shape of the real one: the bundled manifest's nine files, tiny."""
    src = tmp_path / "kb"
    src.mkdir()
    for e in load_manifest():
        (src / e.file).write_bytes(make_docx(e.diseases, {f"Class of {e.kb_id}": [(f"Brand{e.kb_id[-2:]}", f"Ingredient{e.kb_id[-2:]}")]}))
    return src


class Admin:
    def __init__(self, tmp_path, retriever=None, llm=None, seed=True):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.store_dir = tmp_path / "store"
        self.src = seed_dir(tmp_path) if seed else tmp_path / "none"
        self.llm = llm or FakeLLM(answer_fn=cite_first)
        self.start(retriever)

    def start(self, retriever_for=None):
        """(Re)start the app on the same store, as a restart of the process would."""
        store, docs = open_store(str(self.store_dir), self.src)
        sections = [s for d in docs for s in d.sections]
        retriever = retriever_for(sections) if retriever_for else None
        self.app = create_app(sections, self.llm, Settings(), retriever=retriever, kb_store=store,
                              documents=docs)
        self.api = SyncClient(self.app)
        return self

    def docs(self):
        r = self.api.get("/admin/documents")
        assert r.status_code == 200, r.text
        return {d["kb_id"]: d for d in r.json()["documents"]}

    def upload(self, kb_id, content, diseases=None, filename="f.docx", headers=None, **extra):
        body = {"kb_id": kb_id, "content_base64": b64(content), "filename": filename, **extra}
        if diseases is not None:
            body["diseases"] = diseases
        return self.api.post("/admin/documents", json=body, headers=headers)

    def chat(self, message):
        return self.api.post("/chat", json={"message": message}).json()

    def files(self):
        """Every file in the store with its bytes' hash: for the 'nothing changed' checks."""
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.store_dir.iterdir())}


@pytest.fixture
def admin(tmp_path):
    return Admin(tmp_path)


# ------------------------------------------------------------------ seeding and listing
def test_first_start_seeds_the_store_from_the_read_only_folder(admin):
    names = sorted(p.name for p in admin.store_dir.iterdir())
    assert names == sorted(["manifest.json"] + [f"{e.kb_id}.docx" for e in load_manifest()])
    docs = admin.docs()
    assert sorted(docs) == [e.kb_id for e in load_manifest()]
    d = docs["KB-01"]
    assert d["sections"] == 2 and d["file_name"] == "asthma w copd.docx" and d["title"] and d["updated"]


def test_a_second_start_does_not_seed_again_even_if_the_folder_changed(tmp_path):
    a = Admin(tmp_path)
    a.api.request("DELETE", "/admin/documents/KB-01")
    a.start()  # the folder still has KB-01, the store (the source of truth) does not
    assert "KB-01" not in a.docs()


def test_without_a_store_the_endpoints_answer_503_and_health_is_unchanged():
    api = SyncClient(create_app([], FakeLLM({}), Settings()))
    for r in (api.get("/admin/documents"), api.request("DELETE", "/admin/documents/KB-01"),
              api.post("/admin/documents", json={})):
        assert r.status_code == 503 and "KB store not configured" in r.json()["error"]
    assert set(api.get("/health").json()) == {"status", "llm", "embedding"}


def test_health_gains_the_document_count_with_a_store(admin):
    assert admin.api.get("/health").json()["kb_documents"] == 9


def test_the_admin_page_is_served_without_any_sign_in(admin):
    r = admin.api.get("/admin")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "Bearer" not in r.text


# ------------------------------------------------------------------ no authentication
@pytest.mark.parametrize("method,url,ok", [("GET", "/admin/documents", 200), ("POST", "/admin/documents", 400),
                                           ("DELETE", "/admin/documents/KB-01", 200)])
def test_the_admin_endpoints_answer_without_any_authorization_header(tmp_path, method, url, ok):
    a = Admin(tmp_path)
    r = a.api.request(method, url, json={} if method == "POST" else None)
    assert r.status_code == ok, r.text  # never 401 or 503 "admin disabled"
    assert "Authorization" not in r.request.headers


# ------------------------------------------------------------------ upload, replace, delete
def test_upload_a_new_kb_document_and_ask_about_it(admin):
    before = admin.chat("What is the active ingredient of Zorbex?")
    assert before["status"] == "out_of_scope"
    r = admin.upload("KB-10", make_docx("1. Gout", {"Xanthine oxidase inhibitors": [("Zorbex", "Allopurinol")]}),
                     diseases=["1. Gout"], filename="gout.docx")
    assert r.status_code == 200, r.text
    assert r.json() == {"kb_id": "KB-10", "title": "Gout", "sections": 1}
    assert admin.docs()["KB-10"]["file_name"] == "gout.docx"
    after = admin.chat("What is the active ingredient of Zorbex?")
    assert after["status"] == "answered" and after["citations"][0]["doc_id"] == "KB-10"


def test_replacing_with_an_edited_copy_changes_the_next_answer(admin):
    seen = []
    admin.llm.answer_fn = lambda user: cite_first(user, seen)
    admin.chat("What is the active ingredient of Brand01?")
    assert "Ingredient01" in seen[-1]
    r = admin.upload("KB-01", make_docx(load_manifest()[0].diseases, {"Class of KB-01": [("Brand01", "Editedine")]}))
    assert r.status_code == 200
    admin.chat("What is the active ingredient of Brand01?")
    assert "Editedine" in seen[-1] and "Ingredient01" not in seen[-1]
    assert len(admin.docs()) == 9  # replaced, not added


def test_replace_keeps_the_existing_headings_when_none_are_sent_and_a_new_id_needs_them(admin):
    heading = load_manifest()[0].diseases
    r = admin.upload("KB-01", make_docx(heading, {"Other class": [("B", "I")]}))
    assert r.status_code == 200
    assert admin.docs()["KB-01"]["diseases"] == list(heading)
    r = admin.upload("KB-77", make_docx())
    assert r.status_code == 400 and "'diseases' is required" in r.json()["error"]


def test_delete_makes_its_questions_out_of_scope_and_a_missing_id_is_404(admin):
    assert admin.chat("What is the active ingredient of Brand02?")["status"] != "out_of_scope"
    r = admin.api.request("DELETE", "/admin/documents/KB-02")
    assert r.status_code == 200 and r.json() == {"deleted": "KB-02"}
    assert admin.chat("What is the active ingredient of Brand02?")["status"] == "out_of_scope"
    assert "KB-02" not in admin.docs()
    assert admin.api.request("DELETE", "/admin/documents/KB-02").status_code == 404


def test_deleting_every_document_leaves_an_empty_kb_that_answers_out_of_scope(admin):
    for kb_id in list(admin.docs()):
        assert admin.api.request("DELETE", f"/admin/documents/{kb_id}").status_code == 200
    assert admin.docs() == {}
    assert admin.chat("What is the active ingredient of Brand01?")["status"] == "out_of_scope"
    assert admin.api.get("/health").json()["kb_documents"] == 0
    assert admin.upload("KB-01", make_docx(), diseases=[DISEASE]).status_code == 200  # and it can be filled again
    assert list(admin.docs()) == ["KB-01"]


def test_changes_survive_a_restart(admin):
    admin.upload("KB-10", make_docx("1. Gout", {"Xanthine": [("Zorbex", "Allopurinol")]}), diseases=["1. Gout"])
    admin.api.request("DELETE", "/admin/documents/KB-03")
    before = admin.docs()
    admin.start()
    after = admin.docs()
    assert after == before and "KB-10" in after and "KB-03" not in after
    assert admin.chat("What is the active ingredient of Zorbex?")["status"] == "answered"


def test_the_replaced_file_is_the_only_one_kept_for_that_kb_id(admin):
    admin.upload("KB-01", make_docx(load_manifest()[0].diseases, {"Other": [("B", "I")]}))
    names = [n for n in admin.files() if n.startswith("KB-01")]
    assert len(names) == 1 and names[0] != "KB-01.docx"  # content-named, never the client's file name


# ------------------------------------------------------------------ rejected uploads leave everything as it was
def assert_unchanged(admin, before_files, before_docs, r, status):
    assert r.status_code == status, r.text
    assert "error" in r.json() and r.json()["error"]
    assert admin.files() == before_files
    assert admin.docs() == before_docs


@pytest.mark.parametrize("label,content,diseases,kb_id,status,fragment", [
    ("not a zip", b"this is not a docx", [DISEASE], "KB-10", 400, "not a readable .docx"),
    ("empty", b"", [DISEASE], "KB-10", 400, "'content_base64' is required"),
    ("garbage zip", None, [DISEASE], "KB-10", 400, "not a readable .docx"),
    ("missing heading", make_docx(), ["1. Not in the file"], "KB-10", 400, "disease heading not found in the file: '1. Not in the file'"),
    ("no sections", make_docx(classes={}), [DISEASE], "KB-10", 400, "no Sections"),
    ("bad id", make_docx(), [DISEASE], "KB-1", 400, "'kb_id' must look like KB-01"),
    ("bad id 2", make_docx(), [DISEASE], "KB-10\n", 400, "'kb_id' must look like KB-01"),
    ("traversal", make_docx(), [DISEASE], "../../etc/passwd", 400, "'kb_id'"),
    ("traversal 2", make_docx(), [DISEASE], "KB-01/../../x", 400, "'kb_id'"),
    ("bad headings", make_docx(), "1. Asthma", "KB-10", 400, "'diseases' must be a list"),
    ("blank heading", make_docx(), ["  "], "KB-10", 400, "'diseases' must be a list"),
    ("too large", b"x" * (MAX_UPLOAD_BYTES + 1), [DISEASE], "KB-10", 413, "10 MB"),
], ids=lambda v: v if isinstance(v, str) and len(v) < 30 else "")
def test_invalid_uploads_are_rejected_and_change_nothing(admin, label, content, diseases, kb_id, status, fragment):
    if label == "garbage zip":
        content = b"PK\x03\x04" + bytes(range(256)) * 4
    before_files, before_docs = admin.files(), admin.docs()
    live = admin.chat  # the live KB keeps answering
    r = admin.upload(kb_id, content, diseases=diseases)
    assert fragment in r.json()["error"]
    assert_unchanged(admin, before_files, before_docs, r, status)
    assert live("What is the active ingredient of Brand01?")["status"] in ("answered", "out_of_scope")


def test_invalid_replace_keeps_the_live_document(admin):
    seen = []
    admin.llm.answer_fn = lambda user: cite_first(user, seen)
    before_files = admin.files()
    r = admin.upload("KB-01", make_docx(), diseases=["1. Wrong heading"])
    assert r.status_code == 400
    assert admin.files() == before_files
    admin.chat("What is the active ingredient of Brand01?")
    assert "Ingredient01" in seen[-1]


def test_body_problems_are_400(admin):
    for body in (b"not json", b"[1]", json.dumps({"kb_id": "KB-10", "content_base64": "@@@"}).encode(),
                 json.dumps({"kb_id": "KB-10", "content_base64": b64(b"x"), "filename": 5}).encode()):
        r = admin.api.post("/admin/documents", content=body)
        assert r.status_code == 400, body


def test_a_huge_request_body_is_413_before_it_is_parsed(admin):
    r = admin.api.post("/admin/documents", content=b"{" + b" " * (MAX_UPLOAD_BYTES * 2))
    assert r.status_code == 413


def test_delete_with_a_bad_id_is_400(admin):
    assert admin.api.request("DELETE", "/admin/documents/KB-1").status_code == 400
    assert admin.api.request("DELETE", "/admin/documents/..%2F..%2Fx").status_code in (400, 404)


def test_a_failure_while_building_the_next_snapshot_leaves_store_and_live_kb_alone(admin):
    manager_calls = []

    def broken(old, sections):
        manager_calls.append(1)
        raise RuntimeError("embedding failed")

    # reach the manager through a second one on the same store, sharing nothing with the app
    store = KBStore(admin.store_dir)
    docs = store.load_documents()
    sections = tuple(s for d in docs for s in d.sections)
    from medrag.retrieval import LexicalRetriever
    m = KBManager(KBSnapshot.build(docs, LexicalRetriever(sections), sections), store, derive=broken)
    before = admin.files()
    with pytest.raises(RuntimeError):
        m.upload("KB-10", ["1. Gout"], make_docx("1. Gout", {"X": [("Z", "A")]}))
    assert manager_calls and admin.files() == before and m.snapshot.sections == sections
    assert m._lock.acquire(blocking=False)  # the lock was released


def test_a_change_while_another_runs_is_409(admin):
    manager = KBManager(KBSnapshot.build((), __import__("medrag.retrieval", fromlist=["x"]).LexicalRetriever([]), ()),
                        KBStore(admin.store_dir))
    from medrag.kbmanager import Busy
    manager._lock.acquire()
    with pytest.raises(Busy):
        manager.delete("KB-01")


def test_a_request_in_flight_keeps_its_snapshot(admin):
    """The pipeline reads the snapshot once; a swap during the request does not change what that request searches."""
    import asyncio

    import httpx

    seen = []

    admin.llm.answer_fn = lambda user: cite_first(user, seen)
    admin.llm.delay = 0.3  # the route call takes a moment, during which the document is deleted

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=admin.app), base_url="http://t") as c:
            chat = asyncio.ensure_future(c.post("/chat", json={"message": "What is the active ingredient of Brand01?"}))
            await asyncio.sleep(0.1)
            d = await c.delete("/admin/documents/KB-01")
            return d.status_code, await chat

    status, resp = asyncio.run(go())
    assert status == 200 and resp.status_code == 200
    assert resp.json()["status"] == "answered" and "Ingredient01" in seen[-1]  # answered from the KB it started with
    assert admin.chat("What is the active ingredient of Brand01?")["status"] == "out_of_scope"


# ------------------------------------------------------------------ hybrid path (fake embedder)
DIM = 64


def _h(token, mod):
    return zlib.crc32(token.encode()) % mod


class FakeEmbedder:
    model_id, revision = "fake/bge-m3", "r1"

    def __init__(self):
        self.calls = []

    def encode(self, texts, max_length=512):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            vec, sparse = [0.0] * DIM, {}
            for tok in tokenize(t):
                vec[_h(tok, DIM)] += 1.0
                sparse[_h(tok, 10_000)] = sparse.get(_h(tok, 10_000), 0.0) + 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append(Encoded(tuple(x / norm for x in vec), sparse))
        return out


def hybrid(tmp_path):
    emb = FakeEmbedder()

    def factory(sections):
        r = HybridRetriever(sections, lambda: emb, emb.model_id, emb.revision, cache_dir=tmp_path / "index")
        r.load()
        return r
    return emb, factory


def test_hybrid_replace_reembeds_only_the_changed_sections_and_writes_the_index_cache(tmp_path):
    emb, factory = hybrid(tmp_path)
    admin = Admin(tmp_path, retriever=factory)
    total = sum(len({strip_numbering(d) for d in e.diseases}) for e in load_manifest())
    assert len(emb.calls[0]) == total  # the initial index: every Section
    emb.calls.clear()
    heading = load_manifest()[0].diseases
    r = admin.upload("KB-01", make_docx(heading, {"Class of KB-01": [("Brand01", "Editedine")]}))
    assert r.status_code == 200
    assert len(emb.calls) == 1 and len(emb.calls[0]) == len(heading) and all("Editedine" in t for t in emb.calls[0])  # KB-01's Sections only
    caches = list((tmp_path / "index").glob("index-*.json"))
    assert len(caches) == 1  # the new key, the old index removed
    data = json.loads(caches[0].read_text("utf-8"))
    assert len(data["docs"]) == total
    emb.calls.clear()
    assert admin.api.request("DELETE", "/admin/documents/KB-02").status_code == 200
    assert emb.calls == []  # a delete embeds nothing
    assert len(json.loads(next((tmp_path / "index").glob("index-*.json")).read_text("utf-8"))["docs"]) == total - len(load_manifest()[1].diseases)


def test_hybrid_search_sees_the_new_document_and_an_empty_kb_is_out_of_scope(tmp_path):
    emb, factory = hybrid(tmp_path)
    admin = Admin(tmp_path, retriever=factory)
    admin.upload("KB-10", make_docx("1. Gout", {"Xanthine": [("Zorbex", "Allopurinol")]}), diseases=["1. Gout"])
    seen = []
    admin.llm.answer_fn = lambda user: cite_first(user, seen)
    admin.chat("What is the active ingredient of Zorbex?")
    assert "Allopurinol" in seen[-1]
    for kb_id in list(admin.docs()):
        admin.api.request("DELETE", f"/admin/documents/{kb_id}")
    assert admin.chat("What is the active ingredient of Zorbex?")["status"] == "out_of_scope"


def test_a_change_before_the_index_is_loaded_is_503(tmp_path):
    emb = FakeEmbedder()
    unloaded = lambda sections: HybridRetriever(sections, lambda: emb, emb.model_id, emb.revision)  # noqa: E731
    admin = Admin(tmp_path, retriever=unloaded)
    r = admin.api.request("DELETE", "/admin/documents/KB-01")
    assert r.status_code == 503 and "still loading" in r.json()["error"]


# ------------------------------------------------------------------ the store on its own
def test_stray_files_of_an_interrupted_change_are_removed_at_start(admin):
    (admin.store_dir / "KB-01.0123456789ab.docx").write_bytes(b"orphan")
    (admin.store_dir / ".tmp-abc.part").write_bytes(b"half")
    (admin.store_dir / "notes.txt").write_bytes(b"not ours")
    admin.start()
    names = set(admin.files())
    assert "KB-01.0123456789ab.docx" not in names and ".tmp-abc.part" not in names and "notes.txt" in names


def test_seeding_fails_loudly_when_the_folder_lacks_files(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        open_store(str(tmp_path / "store"), tmp_path / "empty")
    assert not (tmp_path / "store" / "manifest.json").exists()  # left unseeded: the next start tries again


# ------------------------------------------------------------------ unpacked-size limits and ASCII-only KB IDs
def zip_with(document_xml_size=10, extra_entries=0):
    """A real docx whose word/document.xml is padded (with trailing whitespace) to a size, plus filler entries."""
    import zipfile
    src = zipfile.ZipFile(io.BytesIO(make_docx()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "word/document.xml":
                data += b" " * max(0, document_xml_size - len(data))
            z.writestr(info.filename, data)
        for n in range(extra_entries):
            z.writestr(f"extra/{n}.txt", b"x")
    return out.getvalue()


@pytest.mark.parametrize("label,patch,content,fragment", [
    ("xml bomb", {"MAX_DOCUMENT_XML_BYTES": 50_000}, zip_with(document_xml_size=2_000_000), "document text is larger"),
    ("unpacks too much", {"MAX_UNPACKED_BYTES": 50_000}, zip_with(document_xml_size=2_000_000), "unpacks to more than"),
    ("too many entries", {"MAX_ZIP_ENTRIES": 5}, zip_with(extra_entries=20), "too many parts"),
    ("too many blocks", {"MAX_BLOCKS": 3}, make_docx(classes={"A": [("a", "b")] * 4}), "too many paragraphs"),
    ("section too long", {"MAX_SECTION_CHARS": 30}, make_docx(), "too long"),
], ids=["xml-bomb", "unpacks-too-much", "too-many-entries", "too-many-blocks", "section-too-long"])
def test_oversized_uploads_are_rejected_and_change_nothing(admin, monkeypatch, label, patch, content, fragment):
    import medrag.kbstore as kbstore
    for name, value in patch.items():
        monkeypatch.setattr(kbstore, name, value)
    before_files, before_docs = admin.files(), admin.docs()
    r = admin.upload("KB-10", content, diseases=[DISEASE])
    assert r.status_code == 400 and fragment in r.json()["error"], r.text
    assert admin.files() == before_files and admin.docs() == before_docs


def test_a_decompression_bomb_is_refused_before_it_is_unpacked(admin, monkeypatch):
    """The check reads only the zip directory: parsing is never reached."""
    import medrag.kbstore as kbstore
    monkeypatch.setattr(kbstore, "read_blocks", lambda path: pytest.fail("the bomb was parsed"))
    r = admin.upload("KB-10", zip_with(document_xml_size=30 * 1024 * 1024), diseases=[DISEASE])  # compresses to ~30 KB
    assert r.status_code == 400 and "document text is larger" in r.json()["error"]


def test_the_default_limits_pass_a_normal_document(admin):
    assert admin.upload("KB-10", make_docx(), diseases=[DISEASE]).status_code == 200


def test_the_real_kb_files_pass_validation(real_kb, tmp_path):
    from medrag.ingest import kb_dir
    store = KBStore(tmp_path)
    for e in load_manifest():
        doc = store.validate(e.kb_id, e.diseases, (kb_dir() / e.file).read_bytes())
        assert doc.sections


@pytest.mark.parametrize("kb_id", ["KB-٠١", "KB-۰۱", "KB-０１"])
def test_kb_ids_with_non_ascii_digits_are_rejected(admin, kb_id):
    r = admin.upload(kb_id, make_docx(), diseases=[DISEASE])
    assert r.status_code == 400 and "'kb_id'" in r.json()["error"]
    assert admin.api.request("DELETE", f"/admin/documents/{kb_id}").status_code == 400
