"""Checks against the real KB .docx files. Skipped when data/ is absent."""
import json
from pathlib import Path

import pytest
from docx import Document

from medrag.dump import render_dump
from medrag.ingest import kb_dir, load_knowledge_base
from medrag.manifest import load_manifest

pytestmark = pytest.mark.kb

ROOT = Path(__file__).resolve().parents[1]


def test_all_nine_documents_load_with_manifest_ids(real_kb):
    assert sorted(real_kb) == [f"KB-0{i}" for i in range(1, 10)]


def test_every_section_is_complete_and_unique(real_kb):
    for doc in real_kb.values():
        assert doc.sections
        for s in doc.sections:
            assert s.kb_id == doc.kb_id and s.title
            assert s.section or s.kind == "overview"
            for row in s.rows:
                assert row.text == " – ".join(row.cells)
        assert len({s.path for s in doc.sections}) == len(doc.sections)


def test_kb03_has_no_ibs_or_arthritis_and_kb09_does(real_kb):
    kb03 = " ".join(s.title for s in real_kb["KB-03"].sections)
    assert "IBS" not in kb03 and "Arthritis" not in kb03
    assert len(real_kb["KB-03"].diseases) == 2
    kb09 = " ".join(s.title for s in real_kb["KB-09"].sections)
    assert "IBS" in kb09 and "Arthritis" in kb09


def test_kb01_has_separate_asthma_and_copd_saba_sections(real_kb):
    saba = [s for s in real_kb["KB-01"].sections if (s.section or "").startswith("SABA")]
    assert len(saba) == 2
    assert {s.title.rsplit("– ", 1)[-1] for s in saba} == {"Asthma", "COPD"}
    assert saba[0].path != saba[1].path


def test_controloc_pantoprazole_is_in_kb02_ppis(real_kb):
    ppi = next(
        s for s in real_kb["KB-02"].sections if s.section == "Proton Pump Inhibitors (PPIs)"
    )
    assert ppi.title == "أدوية قرحة المعدة والحموضة – Peptic Ulcer & GERD"
    assert "Controloc – Pantoprazole" in [r.text for r in ppi.rows]


def test_known_traps(real_kb):
    kb08 = [s.section for s in real_kb["KB-08"].sections]
    assert "Fibrates" in kb08 and "Omega-3 Fatty Acids" in kb08
    kb05 = real_kb["KB-05"]
    assert len(kb05.diseases) == 1
    assert [s.section for s in kb05.sections].count("Penicillins") == 1
    assert len(real_kb["KB-02"].diseases) == 2
    kb04_rows = [r.cells for s in real_kb["KB-04"].sections for r in s.rows]
    assert ("Strepsils Plus / Orofar", "") in kb04_rows


def test_no_table_row_is_lost(real_kb):
    for entry in load_manifest():
        doc = Document(str(kb_dir() / entry.file))
        expected = sum(len(t.rows) - 1 for t in doc.tables)
        assert sum(s.row_count for s in real_kb[entry.kb_id].sections) == expected


def test_ingestion_twice_is_identical(real_kb):
    a = load_knowledge_base()
    b = load_knowledge_base()
    assert a == b
    assert json.dumps([d.to_dict() for d in a], ensure_ascii=False) == json.dumps(
        [d.to_dict() for d in b], ensure_ascii=False
    )


def test_committed_structure_dump_is_current(real_kb):
    committed = (ROOT / "docs" / "kb-structure.md").read_text(encoding="utf-8")
    assert committed == render_dump(load_knowledge_base())
