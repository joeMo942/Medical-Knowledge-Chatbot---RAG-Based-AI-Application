"""In-memory Sessions (SQLite persistence comes in #8). Turns are stored but,
per the glossary, only ever resolve references; they are never a source of facts."""
from __future__ import annotations

import re
import uuid
from collections import OrderedDict

_VALID_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class SessionStore:
    def __init__(self, max_sessions: int = 1000, max_turns: int = 20):
        self._data: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        self._max_sessions, self._max_turns = max_sessions, max_turns

    def resolve(self, session_id: object) -> str:
        """The given ID if usable (and so reused), else a fresh one."""
        if isinstance(session_id, str) and _VALID_ID.match(session_id):
            return session_id
        return uuid.uuid4().hex

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        turns = self._data.setdefault(session_id, [])
        turns.append((question, answer))
        del turns[: -self._max_turns]
        self._data.move_to_end(session_id)
        while len(self._data) > self._max_sessions:
            self._data.popitem(last=False)

    def turns(self, session_id: str) -> list[tuple[str, str]]:
        return list(self._data.get(session_id, []))
