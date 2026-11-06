"""Smoke test against the running docker compose stack (and the real KB files).

Skipped unless the API answers /health at MEDRAG_API_URL (default http://localhost:8000).
"""
import os

import httpx
import pytest

from medrag.ingest import kb_dir, load_knowledge_base, missing_files
from medrag.manifest import load_manifest

pytestmark = pytest.mark.e2e
URL = os.environ.get("MEDRAG_API_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def api():
    try:
        httpx.get(f"{URL}/health", timeout=3).raise_for_status()
    except Exception:
        pytest.skip(f"stack not running at {URL} (start it with: docker compose up -d)")
    return httpx.Client(base_url=URL, timeout=40)


def test_dev_question_1_cataflam(api):
    h = api.get("/health").json()
    assert h["status"] == "ok" and h["llm"] and h["embedding"]
    r = api.post("/chat", json={"message": "What is the active ingredient in Cataflam?"})
    assert r.status_code == 200 and r.headers["X-Session-Id"]
    body = r.json()
    assert set(body) == {"answer", "citations", "status", "language"}
    assert body["status"] == "answered" and body["language"] == "en"
    assert "diclofenac" in body["answer"].lower()
    assert body["citations"]
    directory = kb_dir()
    if missing_files(load_manifest(), directory):
        pytest.skip("KB files not available locally to verify citations against")
    sections = {(s.kb_id, s.title, s.section): s for d in load_knowledge_base(directory) for s in d.sections}
    for c in body["citations"]:
        sec = sections[(c["doc_id"], c["title"], c["section"])]
        assert c["snippet"] in sec.render()
