"""Load KB Documents from .docx files into Sections."""
from __future__ import annotations

import hashlib
import os
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


def missing_files(entries: tuple[ManifestEntry, ...], directory: Path) -> list[Path]:
    return [directory / e.file for e in entries if not (directory / e.file).is_file()]


def load_document(entry: ManifestEntry, directory: Path) -> KBDocument:
    """The KB ID comes from the manifest entry, never from the file name."""
    try:
        blocks = read_blocks(directory / entry.file)
        diseases, sections = split_sections(entry.kb_id, entry.diseases, blocks)
    except Exception as exc:  # PackageNotFoundError, BadZipFile, KeyError, SplitError...
        raise IngestError(
            f"{entry.kb_id} ({entry.file}): {type(exc).__name__}: {exc}"
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
            f"KB files not found (set {KB_DIR_ENV}): "
            + ", ".join(str(m) for m in missing)
        )
    return tuple(load_document(e, directory) for e in entries)


def kb_files_hash(directory: Path | None = None, entries: tuple[ManifestEntry, ...] | None = None) -> str:
    """SHA-256 over the KB .docx files (name and bytes, in manifest order): changes when any Document changes."""
    directory = Path(directory) if directory is not None else kb_dir()
    h = hashlib.sha256()
    for e in entries if entries is not None else load_manifest():
        h.update(e.file.encode())
        h.update((directory / e.file).read_bytes())
    return h.hexdigest()
