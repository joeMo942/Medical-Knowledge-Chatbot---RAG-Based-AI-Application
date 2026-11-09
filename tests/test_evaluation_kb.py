"""The test set checked against the real KB (needs the .docx files; skipped without them), plus an
end-to-end run of a few cases against the running stack (skipped unless it is up)."""
import os

import httpx
import pytest

from evaluation import kb_check
from evaluation.run import DEFAULT_CASES, run_case
from evaluation.scoring import build_kb_index, load_cases


@pytest.mark.kb
@pytest.mark.parametrize("cases_file", [kb_check.DEFAULT_CASES, kb_check.DEFAULT_CASES.with_name("cases_issue5.json")])
def test_every_expected_kb_id_is_backed_by_the_real_kb_and_every_trap_is_a_real_gap(real_kb, cases_file):
    cases = load_cases(cases_file)
    texts = kb_check.doc_texts(list(real_kb.values()))
    errors, notes = kb_check.anchor_problems(cases, texts)
    errors += kb_check.absent_problems(cases, texts)
    errors += kb_check.relevance_problems(cases, texts, kb_check.section_names(list(real_kb.values())))
    errors += kb_check.echo_problems(cases)
    assert errors == []
    assert notes == []   # no Document with an anchor is left out of the expected IDs unexplained


# Body lines of the real KB that look like headings but DO carry a fact: short list items, the
# prescription status, and sentences that happen to end in ':'.
KB_HEADING_LIKE_CONTENT_LINES = {
    "Menthol", "Camphor", "Methyl salicylate", "Capsaicin", "Diethylamine salicylate", "OTC: None",
    "Depends on formulation", "Psyllium / Ispaghula husk", "Plantago ovata", "Methylcellulose",
    "Prescription Only",
    "ممكن تقلل امتصاص أدوية كتير، مثل:",
    "الاستخدام: خفض LDL Cholesterol بشكل كبير، خصوصاً في:",
    "في المجموعة دي فيه 3 مواد فعالة أساسية، وبعضها Thiazide-like:",
    "أهم دواء هنا هو Paracetamol، ونفس الأسماء التجارية المذكورة في Analgesics:",
}


@pytest.mark.kb
def test_every_heading_like_line_of_the_real_kb_is_classified(real_kb):
    """Every body line that could be a label (up to 7 words and not a sentence ending in '.', or any line
    ending in ':' or a question mark, whatever its length) is either a label or a known content line, so a
    new heading shape in the KB cannot slip through unnoticed."""
    from evaluation.scoring import is_label_line
    parts = {p.strip() for d in real_kb.values() for s in d.sections for ln in s.lines for p in ln.splitlines()}
    candidates = {p for p in parts if p and ((len(p.split()) <= 7 and not p.endswith("."))
                                             or p.endswith((":", "؟", "?")))}
    labels = {x for x in candidates if is_label_line(x)}
    assert candidates - labels == KB_HEADING_LIKE_CONTENT_LINES
    for heading in ("Side Effects", "Contraindications", "C. Nitroimidazoles", "الأدوية الموجودة في السوق المصري",
                    "OTC ولا Prescription؟", "Common side effects:", "Contraindications / أهم التحذيرات:",
                    "أمثلة إضافية موثقة للـCaffeine combinations في مصر:"):
        assert heading in labels
    assert len([x for x in candidates if x.endswith(":")]) > 25          # the long colon lines are covered too


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
    assert rows[2]["sent_session_id"] == rows[1]["session_id"]   # the Session was carried into turn 2
    # The usage header is opt-in (MEDRAG_USAGE_HEADER=1): a stack started with the default sends none.
    if all(r["usage"] is None for r in rows):
        pytest.skip("usage header not checked: the stack runs without MEDRAG_USAGE_HEADER=1 (the default); "
                    "start it with MEDRAG_USAGE_HEADER=1 docker compose up -d to check it")
    assert all(r["usage"] is not None and r["usage"]["llm_calls"] >= 0 for r in rows)
