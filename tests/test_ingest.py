import json

import pytest
from docx import Document

from medrag.ingest import IngestError, load_knowledge_base, read_blocks
from medrag.manifest import ManifestEntry, load_manifest, parse_manifest
from medrag.splitter import ParagraphBlock, TableBlock


def make_docx(path):
    doc = Document()
    doc.add_heading("1. Asthma", level=2)
    doc.add_heading("1. SABA", level=2)
    t = doc.add_table(rows=3, cols=2)
    rows = [("Trade name", "Active ingredient"), ("Ventolin", "Salbutamol"), ("Odd", "")]
    for r, cells in enumerate(rows):
        for c, text in enumerate(cells):
            t.cell(r, c).text = text
    doc.add_paragraph("Common side effects:")
    doc.save(str(path))


def test_read_blocks_keeps_body_order(tmp_path):
    make_docx(tmp_path / "x.docx")
    blocks = read_blocks(tmp_path / "x.docx")
    kinds = [type(b).__name__ for b in blocks]
    assert kinds == ["ParagraphBlock", "ParagraphBlock", "TableBlock", "ParagraphBlock"]
    assert blocks[0] == ParagraphBlock(text="1. Asthma", style="Heading 2")
    assert isinstance(blocks[2], TableBlock) and blocks[2].rows[2] == ("Odd", "")


def test_kb_id_comes_from_manifest_not_filename(tmp_path):
    # The file name claims KB-07; the manifest says KB-02.
    make_docx(tmp_path / "KB-07.docx")
    entries = (ManifestEntry("KB-02", "KB-07.docx", ("1. Asthma",)),)
    (doc,) = load_knowledge_base(tmp_path, entries)
    assert doc.kb_id == "KB-02"
    (sec,) = doc.sections
    assert sec.kb_id == "KB-02"
    assert sec.path == "Asthma › SABA"
    assert [r.text for r in sec.rows] == ["Ventolin – Salbutamol", "Odd"]
    assert sec.lines == ("Common side effects:",)


def test_ingestion_is_deterministic(tmp_path):
    make_docx(tmp_path / "a.docx")
    entries = (ManifestEntry("KB-01", "a.docx", ("1. Asthma",)),)
    runs = [load_knowledge_base(tmp_path, entries) for _ in range(2)]
    assert runs[0] == runs[1]
    dumps = [json.dumps([d.to_dict() for d in r], ensure_ascii=False) for r in runs]
    assert dumps[0] == dumps[1]


def test_missing_files_raise_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="MEDRAG_KB_DIR"):
        load_knowledge_base(tmp_path)


def test_manifest_has_nine_unique_documents():
    entries = load_manifest()
    assert [e.kb_id for e in entries] == [f"KB-0{i}" for i in range(1, 10)]
    assert len({e.file for e in entries}) == 9


def test_manifest_rejects_bad_or_duplicate_ids():
    doc = {"kb_id": "KB-01", "file": "a.docx", "diseases": ["1. X"]}
    with pytest.raises(ValueError):
        parse_manifest({"documents": [{**doc, "kb_id": "01"}]})
    with pytest.raises(ValueError):
        parse_manifest({"documents": [doc, {**doc, "file": "b.docx"}]})


def test_non_docx_file_raises_ingest_error_naming_kb_and_file(tmp_path):
    (tmp_path / "bad.docx").write_text("not a docx")
    entries = (ManifestEntry("KB-04", "bad.docx", ("1. Asthma",)),)
    with pytest.raises(IngestError, match=r"KB-04 \(bad\.docx\)"):
        load_knowledge_base(tmp_path, entries)


def test_split_problems_raise_ingest_error_naming_kb_and_file(tmp_path):
    make_docx(tmp_path / "a.docx")
    entries = (ManifestEntry("KB-05", "a.docx", ("1. Asthma", "2. COPD")),)
    with pytest.raises(IngestError, match=r"KB-05 \(a\.docx\)"):
        load_knowledge_base(tmp_path, entries)


def test_default_kb_dir_is_relative_to_repo_root(monkeypatch):
    from medrag.ingest import kb_dir

    monkeypatch.delenv("MEDRAG_KB_DIR", raising=False)
    monkeypatch.chdir("/")
    assert kb_dir().name == "data" and (kb_dir().parent / "pyproject.toml").is_file()


# ------------------------------------------------------------------ KB file names: the manifest's or the KB ID's
def test_files_named_by_kb_id_are_found(tmp_path):
    # The brief names the starter files by ID; the author's copies have descriptive names. Both work.
    make_docx(tmp_path / "KB-01.docx")
    make_docx(tmp_path / "kb-02 Asthma.docx")
    entries = (ManifestEntry("KB-01", "asthma w copd.docx", ("1. Asthma",)),
               ManifestEntry("KB-02", "gerd& vomiting.docx", ("1. Asthma",)))
    docs = load_knowledge_base(tmp_path, entries)
    assert [d.kb_id for d in docs] == ["KB-01", "KB-02"]


def test_the_manifest_name_wins_over_an_id_name(tmp_path):
    from medrag.ingest import resolve_file
    make_docx(tmp_path / "asthma w copd.docx")
    make_docx(tmp_path / "KB-01.docx")
    entry = ManifestEntry("KB-01", "asthma w copd.docx", ("1. Asthma",))
    assert resolve_file(entry, tmp_path).name == "asthma w copd.docx"


def test_a_decomposed_arabic_name_is_the_same_file(tmp_path):
    import unicodedata
    from medrag.ingest import resolve_file
    name = "أدوية الالتهاب الرئوي.docx"
    decomposed = unicodedata.normalize("NFD", name)
    assert decomposed != name
    make_docx(tmp_path / decomposed)
    if not any(p.name == decomposed for p in tmp_path.iterdir()):
        pytest.skip("this file system normalises names itself")
    assert resolve_file(ManifestEntry("KB-05", name, ("1. Asthma",)), tmp_path).name == decomposed


@pytest.mark.parametrize("names", [
    ["KB-012.docx"],                       # another ID, not KB-01
    ["KB-01.0123456789ab.docx"],           # the KB store's old version of a Document
    ["KB-01.txt"],                         # not a .docx
])
def test_id_names_that_do_not_count(tmp_path, names):
    for n in names:
        make_docx(tmp_path / n)
    entries = (ManifestEntry("KB-01", "asthma w copd.docx", ("1. Asthma",)),)
    with pytest.raises(FileNotFoundError, match="asthma w copd"):
        load_knowledge_base(tmp_path, entries)


def test_two_id_candidates_are_named_and_the_exact_id_name_wins(tmp_path):
    from medrag.ingest import resolve_file
    entry = ManifestEntry("KB-01", "asthma w copd.docx", ("1. Asthma",))
    make_docx(tmp_path / "KB-01 a.docx")
    make_docx(tmp_path / "KB-01 b.docx")
    with pytest.raises(FileNotFoundError, match="more than one file could be it.*KB-01 a.docx.*KB-01 b.docx"):
        resolve_file(entry, tmp_path)
    make_docx(tmp_path / "KB-01.docx")
    assert resolve_file(entry, tmp_path).name == "KB-01.docx"
