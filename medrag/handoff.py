"""Hand-offs (#12): a user whose question was Escalated leaves a short summary and contact details so a pharmacist
or physician can follow up. One table, `handoffs`, in the Sessions database file (created if missing, so an existing
volume keeps working; Session pruning never touches it, like the analytics log).

Only what the user types is stored, plus the Session ID and, from the Session's latest Escalated turn, the escalation
kind (emergency | dose | diagnosis | treatment_decision) and the Answer Language label. The kind is recovered from the
stored reply: an Escalated reply is fixed server text (escalation.py), so the text identifies its trigger and nothing in
/chat had to change. A hand-off is accepted only for a Session that has an Escalated turn in its stored history, and at
most MAX_PER_SESSION per Session, so the endpoint is no free-text drop box.

Unlike the conversation log, a failed write here is raised: the user must know the hand-off was not saved."""
from __future__ import annotations

import time
import unicodedata

from .analytics import Window, _iso
from .escalation import ESCALATION_TEXT
from .sessions import SessionStore

MAX_PER_SESSION = 5
SUMMARY_MAX, CONTACT_MIN, CONTACT_MAX = 2000, 3, 200
MAX_BODY_BYTES = 16 * 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS handoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    summary TEXT NOT NULL,
    contact TEXT NOT NULL,
    kind TEXT,
    language TEXT
);
CREATE INDEX IF NOT EXISTS handoffs_by_time ON handoffs (created_at);
CREATE INDEX IF NOT EXISTS handoffs_by_session ON handoffs (session_id);
"""

_KIND_OF_TEXT = {text: (kind, lang) for kind, langs in ESCALATION_TEXT.items() for lang, text in langs.items()}


class NotEscalated(Exception):
    """The Session has no Escalated turn, or it has used up its hand-offs."""


class InvalidHandoff(ValueError):
    pass


def _clean(body: dict, name: str, low: int, high: int, multiline: bool) -> str:
    value = body.get(name)
    if not isinstance(value, str):
        raise InvalidHandoff(f"'{name}' is required and must be text")
    value = value.strip()
    if len(value) < low:
        raise InvalidHandoff(f"'{name}' is required" if low <= 1 else f"'{name}' is too short (minimum {low} characters)")
    if len(value) > high:
        raise InvalidHandoff(f"'{name}' is too long (maximum {high} characters)")
    try:
        value.encode("utf-8")  # a lone surrogate cannot be encoded
    except UnicodeEncodeError:
        raise InvalidHandoff(f"'{name}' contains invalid characters")
    allowed = "\n\r\t" if multiline else ""
    if any(unicodedata.category(c) == "Cc" and c not in allowed for c in value):
        raise InvalidHandoff(f"'{name}' contains control characters")
    return value


def validate(body: dict) -> tuple[str, str]:
    return (_clean(body, "summary", 1, SUMMARY_MAX, True), _clean(body, "contact", CONTACT_MIN, CONTACT_MAX, False))


class HandoffLog:
    def __init__(self, store: SessionStore):
        self._store = store
        with store.connection() as db:
            db.executescript(SCHEMA)

    def add(self, session_id: str, summary: str, contact: str, now: float | None = None) -> dict:
        """Stores one hand-off; raises NotEscalated (nothing stored) or whatever the database raises."""
        escalated = [t for t in self._store.turns(session_id) if t.status == "escalated"]
        if not escalated:
            raise NotEscalated("no escalated question in this session")
        kind, language = _KIND_OF_TEXT.get(escalated[-1].answer, (None, None))
        created = time.time() if now is None else now
        with self._store.connection() as db:
            if language is not None:  # the label of that turn (en | ar | mixed) as /chat logged it, else what its text says
                row = db.execute("SELECT language FROM turn_log WHERE session_id = ? AND status = 'escalated' "
                                 "ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
                language = row[0] if row and row[0] else language
            db.execute("BEGIN IMMEDIATE")
            try:
                (n,) = db.execute("SELECT COUNT(*) FROM handoffs WHERE session_id = ?", (session_id,)).fetchone()
                if n >= MAX_PER_SESSION:
                    raise NotEscalated(f"this session already has {MAX_PER_SESSION} hand-offs")
                hid = db.execute("INSERT INTO handoffs (session_id, created_at, summary, contact, kind, language) "
                                 "VALUES (?, ?, ?, ?, ?, ?)", (session_id, created, summary, contact, kind, language)).lastrowid
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return {"id": hid, "session_id": session_id, "created_at": _iso(created)}

    def listing(self, window: Window = Window(), limit: int = 10) -> dict:
        """The hand-offs in the window: their count and the most recent `limit`, newest first."""
        where, args = window.where("created_at")
        with self._store.connection() as db:
            (count,) = db.execute(f"SELECT COUNT(*) FROM handoffs WHERE {where}", args).fetchone()
            rows = db.execute(f"SELECT id, session_id, created_at, kind, language, summary, contact FROM handoffs "
                              f"WHERE {where} ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
        return {"count": count, "items": [
            {"id": i, "session_id": s, "created_at": _iso(c), "kind": k, "language": lang, "summary": sm, "contact": ct}
            for i, s, c, k, lang, sm, ct in rows]}
