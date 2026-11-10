"""The writable KB store: the KB Documents an admin can upload, replace and delete.

Without `MEDRAG_KB_STORE` the KB is the read-only `MEDRAG_KB_DIR` plus the bundled `kb_manifest.json`, as before. With
it, the store is the source of truth: a directory holding `manifest.json` (KB ID, file, disease headings, the name the
file was uploaded under, when it changed) and the KB Documents' .docx files. The first start seeds it once from the
read-only folder; after that the folder is not read any more.

File names come from the KB ID and the content, never from the client: a seeded file is `<KB ID>.docx`, an uploaded one
`<KB ID>.<12 hex of its SHA-256>.docx`. A change writes the new file first, then `manifest.json` (both atomically:
temp file + os.replace), then removes the file the manifest no longer names, so a crash leaves either the old KB or the
new one, never a mix. Files the manifest does not name are removed at the next start.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .ingest import IngestError, load_document, read_blocks
from .manifest import KB_ID_PATTERN, ManifestEntry, load_manifest, parse_manifest
from .models import KBDocument
from .splitter import ParagraphBlock, strip_numbering

MANIFEST = "manifest.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_STORED = re.compile(r"^KB-\d{2}(\.[0-9a-f]{12})?\.docx$")


class InvalidDocument(ValueError):
    """An upload that cannot become a KB Document; the message says why (HTTP 400)."""


@dataclass(frozen=True)
class StoredDocument:
    entry: ManifestEntry
    uploaded_as: str  # the file name it came with ("" when unknown)
    updated: str  # ISO 8601, UTC


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def check_kb_id(kb_id: object) -> str:
    # fullmatch on a str: `$` in the pattern would also accept a trailing newline.
    if not isinstance(kb_id, str) or not KB_ID_PATTERN.fullmatch(kb_id):
        raise InvalidDocument("'kb_id' must look like KB-01 (KB- and two digits)")
    return kb_id


def check_diseases(diseases: object) -> tuple[str, ...]:
    if (not isinstance(diseases, (list, tuple)) or not diseases
            or not all(isinstance(d, str) and d.strip() for d in diseases)):
        raise InvalidDocument("'diseases' must be a list of heading texts, exactly as they appear in the file")
    return tuple(d.strip() for d in diseases)


class KBStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    # ------------------------------------------------------------------ reading
    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST

    def seeded(self) -> bool:
        return self.manifest_path.is_file()

    def seed(self, source_dir: Path) -> None:
        """Copy the read-only KB folder into the store, once (a no-op when the store has a manifest)."""
        if self.seeded():
            return
        entries = load_manifest()
        missing = [e.file for e in entries if not (source_dir / e.file).is_file()]
        if missing:
            raise FileNotFoundError(f"cannot seed the KB store: KB files not found in {source_dir}: {', '.join(missing)}")
        self.dir.mkdir(parents=True, exist_ok=True)
        docs = []
        for e in entries:
            stored = f"{e.kb_id}.docx"
            _atomic_write(self.dir / stored, (source_dir / e.file).read_bytes())
            docs.append(StoredDocument(ManifestEntry(e.kb_id, stored, e.diseases), e.file, _now()))
        self._write_manifest(docs)  # last: without it the store counts as unseeded and is seeded again

    def documents(self) -> tuple[StoredDocument, ...]:
        data = json.loads(self.manifest_path.read_text("utf-8"))
        entries = parse_manifest(data)
        return tuple(StoredDocument(e, raw.get("uploaded_as", ""), raw.get("updated", ""))
                     for e, raw in zip(entries, data["documents"]))

    def load_documents(self) -> tuple[KBDocument, ...]:
        """Every KB Document of the store, ingested (and stray files of an interrupted change removed)."""
        stored = self.documents()
        self.clean(stored)
        return tuple(load_document(s.entry, self.dir) for s in stored)

    def clean(self, stored: tuple[StoredDocument, ...]) -> None:
        keep = {s.entry.file for s in stored}
        for p in self.dir.iterdir():
            if p.is_file() and p.name not in keep and (_STORED.match(p.name) or p.name.startswith(".tmp-")):
                p.unlink(missing_ok=True)

    # ------------------------------------------------------------------ changing
    def _write_manifest(self, docs: list[StoredDocument] | tuple[StoredDocument, ...]) -> None:
        payload = {"documents": [
            {"kb_id": d.entry.kb_id, "file": d.entry.file, "diseases": list(d.entry.diseases),
             "uploaded_as": d.uploaded_as, "updated": d.updated}
            for d in sorted(docs, key=lambda d: d.entry.kb_id)]}
        _atomic_write(self.manifest_path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    def validate(self, kb_id: str, diseases: tuple[str, ...], content: bytes) -> KBDocument:
        """Run the real ingest on the upload, in a temporary folder; the store is not touched."""
        if not content:
            raise InvalidDocument("the file is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise InvalidDocument(f"the file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.docx"
            path.write_bytes(content)
            try:
                blocks = read_blocks(path)
            except Exception as exc:
                raise InvalidDocument(f"not a readable .docx file ({type(exc).__name__}: {exc})") from exc
            present = {strip_numbering(b.text) for b in blocks if isinstance(b, ParagraphBlock) and b.text.strip()}
            for heading in diseases:
                if strip_numbering(heading) not in present:
                    raise InvalidDocument(f"disease heading not found in the file: {heading!r}")
            try:
                doc = load_document(ManifestEntry(kb_id, path.name, diseases), Path(tmp))
            except IngestError as exc:
                raise InvalidDocument(str(exc)) from exc
        if not doc.sections:
            raise InvalidDocument("the file has no Sections (no drug-class headings under the disease headings)")
        return doc

    def put(self, kb_id: str, diseases: tuple[str, ...], content: bytes, uploaded_as: str) -> None:
        """Store an already validated upload as the KB Document `kb_id` (new or replacing)."""
        stored = list(self.documents())
        name = f"{kb_id}.{hashlib.sha256(content).hexdigest()[:12]}.docx"
        old = next((s for s in stored if s.entry.kb_id == kb_id), None)
        _atomic_write(self.dir / name, content)
        entry = StoredDocument(ManifestEntry(kb_id, name, diseases), uploaded_as, _now())
        self._write_manifest([s for s in stored if s.entry.kb_id != kb_id] + [entry])
        if old is not None and old.entry.file != name:
            (self.dir / old.entry.file).unlink(missing_ok=True)

    def remove(self, kb_id: str) -> None:
        stored = list(self.documents())
        gone = next((s for s in stored if s.entry.kb_id == kb_id), None)
        if gone is None:
            raise KeyError(kb_id)
        self._write_manifest([s for s in stored if s is not gone])
        (self.dir / gone.entry.file).unlink(missing_ok=True)
