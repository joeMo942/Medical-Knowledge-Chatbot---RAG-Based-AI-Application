"""The live KB: an immutable snapshot of everything built from the Sections, and the admin changes that replace it.

A `KBSnapshot` holds the KB Documents, their Sections and what depends on them (the retriever with its search index and
Trade Name index, the support check, the Arabic-letter names). The request path reads `manager.snapshot` once per request
and uses that one object throughout, so a request in flight keeps the KB it started with while an admin change builds the
next snapshot and swaps the reference (one assignment, atomic). Changes run one at a time under a lock.

Order of a change: validate the upload (the real ingest, in a temporary folder) -> build the next snapshot in memory
(re-embedding only the changed Sections) -> save to the store (new file, then the manifest) -> swap. A failure before the
save leaves the store and the live KB as they were.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from .arabicnames import ArabicNames, arabic_common_words
from .drugs import DrugIndex
from .kbstore import InvalidDocument, KBStore, check_diseases, check_kb_id
from .models import KBDocument, Section
from .retrieval import LexicalRetriever, Retriever
from .support import SupportCheck

log = logging.getLogger("medrag")


class NotFound(KeyError):
    pass


class Busy(RuntimeError):
    """Another change is running."""


class NotReady(RuntimeError):
    """The search index is still loading; a change would have nothing to derive from."""


@dataclass(frozen=True)
class KBSnapshot:
    documents: tuple[KBDocument, ...]
    sections: tuple[Section, ...]
    retriever: Retriever
    support: SupportCheck
    names: ArabicNames

    @classmethod
    def build(cls, documents: tuple[KBDocument, ...], retriever: Retriever,
              sections: tuple[Section, ...] | None = None) -> "KBSnapshot":
        sections = tuple(sections) if sections is not None else tuple(s for d in documents for s in d.sections)
        drugs = getattr(retriever, "drugs", None) or DrugIndex(sections)
        return cls(tuple(documents), sections, retriever, SupportCheck(sections),
                   ArabicNames(drugs, arabic_common_words(sections)))

    @property
    def ready(self) -> bool:
        return bool(getattr(self.retriever, "ready", True))


def derive_retriever(old: Retriever, sections: tuple[Section, ...]) -> Retriever:
    """The retriever for changed Sections: the hybrid one reuses its model and the unchanged vectors; any other is BM25."""
    derive = getattr(old, "derive", None)
    return derive(sections) if derive else LexicalRetriever(sections)


class KBManager:
    def __init__(self, snapshot: KBSnapshot, store: KBStore | None = None,
                 derive: Callable[[Retriever, tuple[Section, ...]], Retriever] = derive_retriever):
        self.snapshot = snapshot
        self.store = store
        self._derive = derive
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.store is not None

    # ------------------------------------------------------------------ listing
    def listing(self) -> list[dict]:
        assert self.store is not None
        snap = self.snapshot
        count = Counter(s.kb_id for s in snap.sections)
        titles = {d.kb_id: list(d.diseases) for d in snap.documents}
        out = []
        for s in self.store.documents():
            kb_id = s.entry.kb_id
            out.append({"kb_id": kb_id, "title": " / ".join(titles.get(kb_id, s.entry.diseases)),
                        "diseases": list(s.entry.diseases), "sections": count.get(kb_id, 0),
                        "file_name": s.uploaded_as, "updated": s.updated})
        return out

    @staticmethod
    def _summary(doc: KBDocument) -> dict:
        return {"kb_id": doc.kb_id, "title": " / ".join(doc.diseases), "sections": len(doc.sections)}

    # ------------------------------------------------------------------ changes (blocking: run in a worker thread)
    def _begin(self) -> None:
        if self.store is None:
            raise RuntimeError("no KB store")
        if not self._lock.acquire(blocking=False):
            raise Busy("another KB change is in progress; try again in a moment")
        if not self.snapshot.ready:
            self._lock.release()
            raise NotReady("the search index is still loading; try again shortly")

    def upload(self, kb_id: object, diseases: object, content: bytes, file_name: str = "") -> dict:
        """Add or replace the KB Document `kb_id`. Raises InvalidDocument (nothing changed), Busy, NotReady."""
        kb_id = check_kb_id(kb_id)
        self._begin()
        try:
            assert self.store is not None
            existing = {s.entry.kb_id: s for s in self.store.documents()}
            if diseases is None:
                if kb_id not in existing:
                    raise InvalidDocument("'diseases' is required for a new KB ID: the heading texts, exactly as in the file")
                headings = existing[kb_id].entry.diseases
            else:
                headings = check_diseases(diseases)
            doc = self.store.validate(kb_id, headings, content)
            old = self.snapshot
            docs = tuple(sorted([d for d in old.documents if d.kb_id != kb_id] + [doc], key=lambda d: d.kb_id))
            sections = tuple(s for d in docs for s in d.sections)
            snap = KBSnapshot.build(docs, self._derive(old.retriever, sections), sections)
            self.store.put(kb_id, headings, content, file_name)
            self.snapshot = snap
            log.info("KB Document %s uploaded: %d Sections", kb_id, len(doc.sections))
            return self._summary(doc)
        finally:
            self._lock.release()

    def delete(self, kb_id: str) -> None:
        """Remove the KB Document `kb_id`. Raises NotFound, Busy, NotReady."""
        self._begin()
        try:
            assert self.store is not None
            if not any(s.entry.kb_id == kb_id for s in self.store.documents()):
                raise NotFound(kb_id)
            old = self.snapshot
            docs = tuple(d for d in old.documents if d.kb_id != kb_id)
            sections = tuple(s for d in docs for s in d.sections)
            snap = KBSnapshot.build(docs, self._derive(old.retriever, sections), sections)
            self.store.remove(kb_id)
            self.snapshot = snap
            log.info("KB Document %s deleted", kb_id)
        finally:
            self._lock.release()


def open_store(path: str, source_dir) -> tuple[KBStore, tuple[KBDocument, ...]]:
    """The store at `path`, seeded from `source_dir` when it has no manifest yet, and its ingested KB Documents."""
    store = KBStore(path)
    store.seed(source_dir)
    return store, store.load_documents()
