"""Checks the test set against the real KB (needs the .docx files; `python -m evaluation.kb_check`).

- every expected KB ID of an answered turn must contain at least one of its `anchors`;
- the documents that contain an anchor but are not listed are printed, so each omission is deliberate;
- every `absent_terms` entry of an out-of-scope trap must occur nowhere in the KB.

Matching is case-insensitive and ignores Arabic diacritics, tatweel and alef/yeh/teh-marbuta variants."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from .scoring import Case, load_cases

DEFAULT_CASES = Path(__file__).with_name("cases.json")
_DIACRITICS = re.compile("[ً-ٰٟـ]")
_ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})


def norm(text: str) -> str:
    return _DIACRITICS.sub("", text).translate(_ARABIC_FOLD).casefold()


def doc_texts(documents: list[Any]) -> dict[str, str]:
    """kb_id -> the normalized text of all its Sections (heading lines, body lines, table rows)."""
    return {d.kb_id: norm("\n".join(s.render() for s in d.sections)) for d in documents}


def anchor_problems(cases: list[Case], texts: dict[str, str]) -> tuple[list[str], list[str]]:
    """(errors, notes). An error: an expected KB ID whose document has none of the anchors.
    A note: a document that has an anchor but is not in the expected IDs."""
    errors: list[str] = []
    notes: list[str] = []
    for case in cases:
        for i, t in enumerate(case.turns, 1):
            if t.expected_status != "answered":
                continue
            where = f"{case.id}" + (f" turn {i}" if case.multi_turn else "")
            hits = {kb for kb, text in texts.items() if any(norm(a) in text for a in t.anchors)}
            for kb in t.expected_kb_ids:
                if kb not in hits:
                    errors.append(f"{where}: {kb} contains none of the anchors {list(t.anchors)}")
            extra = sorted(hits - set(t.expected_kb_ids))
            if extra:
                notes.append(f"{where}: also contain an anchor but are not expected: {extra}")
    return errors, notes


def absent_problems(cases: list[Case], texts: dict[str, str]) -> list[str]:
    """Errors: an `absent_terms` entry that does occur in the KB (so the trap is not a trap)."""
    errors = []
    for case in cases:
        for i, t in enumerate(case.turns, 1):
            for term in t.absent_terms:
                found = sorted(kb for kb, text in texts.items() if norm(term) in text)
                if found:
                    errors.append(f"{case.id} turn {i}: {term!r} occurs in {found}")
    return errors


def main(argv: list[str] | None = None) -> int:
    from medrag.ingest import kb_dir, load_knowledge_base  # only here: the checks above need no medrag

    p = argparse.ArgumentParser(prog="python -m evaluation.kb_check", description=__doc__.split("\n")[0])
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--kb-dir", type=Path, help="KB .docx folder (default: $MEDRAG_KB_DIR or ./data)")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cases = load_cases(args.cases)
    texts = doc_texts(load_knowledge_base(args.kb_dir or kb_dir()))
    errors, notes = anchor_problems(cases, texts)
    errors += absent_problems(cases, texts)
    for n in notes:
        print("note:", n)
    for e in errors:
        print("ERROR:", e)
    print(f"{len(cases)} cases checked, {len(errors)} errors, {len(notes)} notes")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
