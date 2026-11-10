"""Sessions and their turns, stored in SQLite (stdlib `sqlite3`; the file lives on a Docker volume).

Per the glossary, a Session's earlier turns only resolve references like "it": they are handed to the
routing call (routing.py) so it can rewrite a follow-up into a standalone question, and are never a
source of facts. This module only stores and returns them.

- One connection guarded by a lock (the pipeline calls in through asyncio.to_thread, so several
  threads may arrive); WAL mode so a reader never blocks the writer; the schema is created on open.
- Bounded: a Session keeps its last MAX_TURNS turns and the store its MAX_SESSIONS most recently
  used Sessions; each stored text is cut to MAX_TEXT_CHARS.
- A Session ID comes from the client, so it is untrusted: only 1-128 of A-Z a-z 0-9 . _ : - are
  accepted (it is only ever used as a bound SQL parameter). An unusable ID is not an error: the
  request gets a fresh Session and the ID in its X-Session-Id header. An unknown but valid ID starts
  a Session under that ID.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path

_VALID_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
MAX_TURNS = 20  # turns kept per Session (only the last few are ever used)
MAX_SESSIONS = 10_000  # least recently used Sessions are dropped beyond this
MAX_TEXT_CHARS = 4000  # a message is at most 2000 characters; an answer is far shorter than this

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    answer TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns (session_id, id);
CREATE INDEX IF NOT EXISTS sessions_by_use ON sessions (updated_at);
"""


@dataclass(frozen=True)
class Turn:
    message: str  # what the user wrote
    answer: str  # the reply text (fixed text for an escalated / out_of_scope reply)
    status: str  # answered | out_of_scope | escalated


class SessionStore:
    def __init__(self, path: str | Path = ":memory:", max_sessions: int = MAX_SESSIONS, max_turns: int = MAX_TURNS):
        self._max_sessions, self._max_turns = max_sessions, max_turns
        self._lock = threading.Lock()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), timeout=5.0, check_same_thread=False, isolation_level=None)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")  # stays "memory" for :memory:, which is fine
            self._db.execute("PRAGMA synchronous=NORMAL")  # durable across a process restart; WAL's usual setting
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """The one connection, under its lock: the analytics log (analytics.py) lives in the same database file."""
        with self._lock:
            yield self._db

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def resolve(session_id: object) -> str:
        """The given ID if usable (and so reused), else a fresh one."""
        if isinstance(session_id, str) and _VALID_ID.fullmatch(session_id):
            return session_id
        return uuid.uuid4().hex

    def add_turn(self, session_id: str, message: str, answer: str, status: str) -> None:
        now = time.time()
        with self._lock:
            db = self._db
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("INSERT INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?) "
                           "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at", (session_id, now, now))
                db.execute("INSERT INTO turns (session_id, message, answer, status, created_at) VALUES (?, ?, ?, ?, ?)",
                           (session_id, message[:MAX_TEXT_CHARS], answer[:MAX_TEXT_CHARS], status, now))
                db.execute("DELETE FROM turns WHERE session_id = ? AND id NOT IN "
                           "(SELECT id FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
                           (session_id, session_id, self._max_turns))
                db.execute("DELETE FROM sessions WHERE id IN (SELECT id FROM sessions ORDER BY updated_at DESC "
                           "LIMIT -1 OFFSET ?)", (self._max_sessions,))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        """The last `limit` turns (all kept ones when None), oldest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT message, answer, status FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, -1 if limit is None else limit)).fetchall()
        return [Turn(*r) for r in reversed(rows)]
