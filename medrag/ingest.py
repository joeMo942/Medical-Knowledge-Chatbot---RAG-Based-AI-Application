"""Load KB Documents from .docx files into Sections."""
from __future__ import annotations

import hashlib
import os
import unicodedata
from pathlib import Path

from docx import Document
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph

from .manifest import ManifestEntry, load_manifest
from .models import KBDocument
from .splitter import Block, ParagraphBlock, SplitError, TableBlock, split_sections

KB_DIR_ENV = "MEDRAG_KB_DIR"
# Relative to the repository root, not the working directory.
DEFAULT_KB_DIR = Path(__file__).resolve().parents[1] / "data"


class IngestError(ValueError):
    """A KB file could not be read or split; the message names KB ID and file."""


def kb_dir() -> Path:
    """Directory holding the KB .docx files (not committed to git)."""
    return Path(os.environ.get(KB_DIR_ENV) or DEFAULT_KB_DIR)


def read_blocks(path: Path) -> list[Block]:
    """Paragraphs and tables of a .docx in document (body) order.

    python-docx is used for the zip/XML handling; walking the body element's
    children (instead of `doc.paragraphs` / `doc.tables`) keeps each table
    next to the paragraphs around it.
    """
    doc = Document(str(path))
    blocks: list[Block] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            style = p.style.name if p.style is not None else ""
            blocks.append(ParagraphBlock(text=p.text, style=style))
        elif tag == "tbl":
            t = DocxTable(child, doc)
            blocks.append(
                TableBlock(rows=tuple(tuple(c.text for c in r.cells) for r in t.rows))
            )
    return blocks


def _name_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _named_by_id(stem: str, kb_id: str) -> bool:
    """True for the stems "KB-01", "kb-01", "KB-01 Asthma" or "KB-01_asthma": the KB ID, then nothing or a space, "_" or
    "-". Not "KB-012", and not the store's old versions "KB-01.<hash>"."""
    stem, kb_id = stem.casefold(), kb_id.casefold()
    return stem.startswith(kb_id) and (len(stem) == len(kb_id) or stem[len(kb_id)] in " _-")


def resolve_file(entry: ManifestEntry, directory: Path) -> Path:
    """The file of a KB Document in `directory`.

    First the manifest's name ("asthma w copd.docx"), also when it differs only in case or Unicode normal form (an
    Arabic name copied through macOS can arrive decomposed). Then the one .docx named by the KB ID ("KB-01.docx",
    as the brief says the starter files are named), preferring exactly "KB-01.docx"; two other candidates
    ("KB-01 a.docx", "KB-01 b.docx") raise FileNotFoundError naming both, so the right file is never guessed.
    If nothing matches, the manifest's path, which does not exist."""
    exact = directory / entry.file
    if exact.is_file():
        return exact
    try:
        files = sorted(p for p in directory.iterdir() if p.is_file())
    except OSError:
        return exact
    want = _name_key(entry.file)
    same = [p for p in files if _name_key(p.name) == want]
    if len(same) == 1:
        return same[0]
    by_id = [p for p in files if p.suffix.casefold() == ".docx" and _named_by_id(p.stem, entry.kb_id)]
    exact_id = [p for p in by_id if p.stem.casefold() == entry.kb_id.casefold()]
    if len(exact_id) == 1:
        return exact_id[0]  # "KB-01.docx" next to a stray "KB-01 - Copy.docx"
    if len(by_id) == 1:
        return by_id[0]
    if by_id:
        raise FileNotFoundError(f"{entry.kb_id}: more than one file could be it, keep only one: "
                                + ", ".join(p.name for p in by_id))
    return exact


def missing_files(entries: tuple[ManifestEntry, ...], directory: Path) -> list[Path]:
    return [directory / e.file for e in entries if not resolve_file(e, directory).is_file()]


def missing_message(missing: list[Path]) -> str:
    return (", ".join(str(m) for m in missing)
            + " (each file may also be named by its KB ID, for example KB-01.docx)")


def load_document(entry: ManifestEntry, directory: Path, lookup: bool = True) -> KBDocument:
    """The KB ID comes from the manifest entry, never from the file name. `lookup=False` reads exactly the manifest's
    file (the KB store, whose own manifest names its files)."""
    path = resolve_file(entry, directory) if lookup else directory / entry.file
    try:
        blocks = read_blocks(path)
        diseases, sections = split_sections(entry.kb_id, entry.diseases, blocks)
    except Exception as exc:  # PackageNotFoundError, BadZipFile, KeyError, SplitError...
        raise IngestError(
            f"{entry.kb_id} ({path.name}): {type(exc).__name__}: {exc}"
        ) from exc
    return KBDocument(
        kb_id=entry.kb_id, file=entry.file, diseases=diseases, sections=sections
    )


def load_knowledge_base(
    directory: Path | None = None, entries: tuple[ManifestEntry, ...] | None = None
) -> tuple[KBDocument, ...]:
    directory = Path(directory) if directory is not None else kb_dir()
    entries = entries if entries is not None else load_manifest()
    missing = missing_files(entries, directory)
    if missing:
        raise FileNotFoundError(
            f"KB files not found (set {KB_DIR_ENV}): " + missing_message(missing)
        )
    return tuple(load_document(e, directory) for e in entries)


def kb_files_hash(directory: Path | None = None, entries: tuple[ManifestEntry, ...] | None = None) -> str:
    """SHA-256 over the KB .docx files (name and bytes, in manifest order): changes when any Document changes."""
    directory = Path(directory) if directory is not None else kb_dir()
    h = hashlib.sha256()
    for e in entries if entries is not None else load_manifest():
        h.update(e.file.encode())
        h.update(resolve_file(e, directory).read_bytes())
    return h.hexdigest()
