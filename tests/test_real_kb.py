"""Checks against the real KB .docx files. Skipped when data/ is absent."""
import json
from collections import Counter
from pathlib import Path

import pytest
from docx import Document

from medrag.dump import render_dump
from medrag.ingest import kb_dir, load_knowledge_base
from medrag.manifest import load_manifest
from medrag.models import Line, Table
from medrag.splitter import strip_numbering

pytestmark = pytest.mark.kb

ROOT = Path(__file__).resolve().parents[1]


def test_all_nine_documents_load_with_manifest_ids(real_kb):
    assert sorted(real_kb) == [f"KB-0{i}" for i in range(1, 10)]


def test_every_section_is_complete_and_unique(real_kb):
    for doc in real_kb.values():
        assert doc.sections
        for s in doc.sections:
            assert s.kb_id == doc.kb_id and s.title
            assert s.section, "every Section has a citation heading"
            for t in s.tables:
                if t.kind == "products":
                    n = len(t.header.cells)
                    assert {t.trade_name_col, t.active_ingredient_col} <= set(range(n))
                    assert t.trade_name_col != t.active_ingredient_col
                    assert all(len(r.cells) == n for r in t.rows)
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


# (product rows, other rows) per KB Document, as classified from table headers.
EXPECTED_ROWS = {
    "KB-01": (144, 18), "KB-02": (76, 17), "KB-03": (98, 11), "KB-04": (48, 7),
    "KB-05": (76, 7), "KB-06": (154, 15), "KB-07": (103, 73), "KB-08": (157, 123),
    "KB-09": (116, 19),
}


def test_product_and_other_row_counts_are_pinned(real_kb):
    got = {
        k: (sum(len(s.product_rows) for s in d.sections), sum(len(s.other_rows) for s in d.sections))
        for k, d in real_kb.items()
    }
    assert got == EXPECTED_ROWS
    assert sum(p for p, _ in got.values()) == 972 and sum(o for _, o in got.values()) == 290


def test_table_kinds_overall(real_kb):
    kinds = Counter()
    for d in real_kb.values():
        for s in d.sections:
            for t in s.tables:
                kinds[t.kind] += len(t.rows)
    assert kinds == {"products": 972, "side_effects": 202, "otc": 87, "other": 1}


def _section(doc, name):
    return next(s for s in doc.sections if s.section.startswith(name))


def _table_after(section, label):
    items = section.items
    i = next(k for k, it in enumerate(items) if isinstance(it, Line) and it.text == label)
    assert isinstance(items[i + 1], Table)
    return items[i + 1]


def test_sublabels_tie_to_the_right_table(real_kb):
    insulins = _section(real_kb["KB-07"], "Insulins")
    rapid = _table_after(insulins, "A. Rapid-acting Insulin")
    assert rapid.rows[0].text == "NovoRapid – Insulin Aspart" and len(rapid.rows) == 5
    premixed = _table_after(insulins, "E. Premixed Insulin")
    assert premixed.rows[0].cells[0] == "Mixtard 30" and rapid is not premixed
    abx = _section(real_kb["KB-04"], "Antibiotics")
    linco = _table_after(abx, "D. Lincosamides")
    assert [r.text for r in linco.rows][0] == "Dalacin C – Clindamycin" and len(linco.rows) == 2
    macro = _table_after(abx, "C. Macrolides")
    assert macro.rows[0].text == "Zithromax – Azithromycin"


def test_reversed_tables_are_typed_with_active_ingredient_first(real_kb):
    secretagogues = _section(real_kb["KB-09"], "Secretagogues").tables[0]
    assert secretagogues.header.cells == ("Generic Name", "Examples of International Trade Names")
    assert (secretagogues.kind, secretagogues.trade_name_col, secretagogues.active_ingredient_col) == (
        "products", 1, 0)
    assert len(secretagogues.rows) == 4
    bdmards = _section(real_kb["KB-09"], "Biologic DMARDs").tables[0]
    assert bdmards.header.cells == ("Drug Class / Generic", "Trade Names")
    assert (bdmards.kind, bdmards.trade_name_col, bdmards.active_ingredient_col) == ("products", 1, 0)
    assert len(bdmards.rows) == 9


def test_citation_headings_come_from_the_source(real_kb):
    for entry in load_manifest():
        paragraphs = {
            strip_numbering(p.text) for p in Document(str(kb_dir() / entry.file)).paragraphs
        }
        for s in real_kb[entry.kb_id].sections:
            assert s.title in paragraphs, (s.section_id, s.title)
            assert s.section in paragraphs, (s.section_id, s.section)


def test_headingless_blocks_cite_their_disease_title(real_kb):
    headingless = [
        s for d in real_kb.values() for s in d.sections if not s.heading_present
    ]
    assert sorted((s.kb_id, s.kind) for s in headingless) == [
        ("KB-01", "overview"), ("KB-02", "otc"), ("KB-05", "overview")
    ]
    assert all(s.section == s.title for s in headingless)


def test_omega3_keeps_its_otc_note_inside_the_section(real_kb):
    dys = [s for s in real_kb["KB-08"].sections if s.title.endswith("Dyslipidemia")]
    assert not any(s.kind == "otc" for s in dys)
    omega = _section(real_kb["KB-08"], "Omega-3")
    assert "OTC ولا Prescription؟" in omega.lines


def test_every_source_paragraph_is_accounted_for_exactly_once(real_kb):
    for entry in load_manifest():
        raw = Counter(
            strip_numbering(p.text)
            for p in Document(str(kb_dir() / entry.file)).paragraphs
            if p.text.strip()
        )
        doc = real_kb[entry.kb_id]
        stored = Counter(strip_numbering(l) for s in doc.sections for l in s.lines)
        stored.update(s.section for s in doc.sections if s.heading_present)
        stored.update(strip_numbering(h) for h in entry.diseases)
        assert stored == raw, entry.kb_id  # equality in both directions


def test_render_and_section_id_contain_only_source_text(real_kb):
    ids = []
    for entry in load_manifest():
        paragraphs = {
            strip_numbering(p.text) for p in Document(str(kb_dir() / entry.file)).paragraphs
        }
        for s in real_kb[entry.kb_id].sections:
            ids.append(s.section_id)
            first = s.render().splitlines()[0]
            assert "no heading in source" not in s.render()
            assert all(part in paragraphs for part in first.split(" › ")), first
            assert "no heading in source" not in s.section_id
    assert len(set(ids)) == len(ids)
