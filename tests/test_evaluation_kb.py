"""The test set checked against the real KB (needs the .docx files; skipped without them), plus an
end-to-end run of a few cases against the running stack (skipped unless it is up)."""
import os
from pathlib import Path

import httpx
import pytest

from evaluation import kb_check
from evaluation.run import DEFAULT_CASES, run_case
from evaluation.scoring import build_kb_index, load_cases


@pytest.mark.kb
def test_every_expected_kb_id_is_backed_by_the_real_kb_and_every_trap_is_a_real_gap(real_kb):
    cases = load_cases(kb_check.DEFAULT_CASES)
    texts = kb_check.doc_texts(list(real_kb.values()))
    errors, notes = kb_check.anchor_problems(cases, texts)
    errors += kb_check.absent_problems(cases, texts)
    errors += kb_check.relevance_problems(cases, texts, kb_check.section_names(list(real_kb.values())))
    assert errors == []
    assert notes == []   # no Document with an anchor is left out of the expected IDs unexplained


# Short body lines of the real KB with no sentence content that DO carry a fact (an item of a list).
KB_SHORT_CONTENT_LINES = {
    "Menthol", "Camphor", "Methyl salicylate", "Capsaicin", "Diethylamine salicylate", "OTC: None",
    "Depends on formulation", "Psyllium / Ispaghula husk", "Plantago ovata", "Methylcellulose",
}


@pytest.mark.kb
def test_every_short_heading_like_line_of_the_real_kb_is_classified(real_kb):
    """Every body line of up to 7 words that is not a sentence (no final '.') is either a label or a
    known content line, so a new heading shape in the KB cannot slip through as support unnoticed."""
    from evaluation.scoring import is_label_line
    short = {p.strip() for d in real_kb.values() for s in d.sections for ln in s.lines for p in ln.split("\n")
             if p.strip() and len(p.split()) <= 7 and not p.strip().endswith(".")}
    labels = {x for x in short if is_label_line(x)}
    assert short - labels == KB_SHORT_CONTENT_LINES
    for heading in ("Prescription Only", "Side Effects", "Contraindications", "C. Nitroimidazoles",
                    "الأدوية الموجودة في السوق المصري", "OTC ولا Prescription؟", "Common side effects:"):
        assert heading in labels


def test_normalisation_for_matching():
    assert kb_check.norm("يُتجنب") == kb_check.norm("يتجنب")
    assert kb_check.norm("أمي") == kb_check.norm("امي")
    assert kb_check.norm("Malaria") == "malaria"


@pytest.mark.e2e
def test_runner_scores_real_cases_against_the_stack(real_kb):
    url = os.environ.get("MEDRAG_API_URL", "http://localhost:8000")
    try:
        httpx.get(f"{url}/health", timeout=3).raise_for_status()
    except Exception:
        pytest.skip(f"stack not running at {url} (start it with: docker compose up -d)")
    kb_index = build_kb_index(real_kb.values())
    cases = {c.id: c for c in load_cases(DEFAULT_CASES)}
    with httpx.Client(base_url=url, timeout=45) as client:
        rows = run_case(client, cases["dev-09-voltaren-daily-mg"], kb_index)
        rows += run_case(client, cases["multi-controloc-class"], kb_index)
    assert [r["turn"] for r in rows] == [1, 1, 2]
    for r in rows:
        assert r["http_status"] == 200 and r["session_id"] and r["latency_s"] > 0
        assert r["contract_violations"] == []
        assert r["usage"] is not None and r["usage"]["llm_calls"] >= 0
    assert rows[2]["sent_session_id"] == rows[1]["session_id"]   # the Session was carried into turn 2
