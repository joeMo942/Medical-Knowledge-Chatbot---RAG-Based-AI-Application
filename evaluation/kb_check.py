"""Checks the test set against the real KB (needs the .docx files; `python -m evaluation.kb_check`).

- every expected KB ID of an answered turn must contain at least one of its `anchors`;
- the documents that contain an anchor but are not listed are printed, so each omission is deliberate;
- every `expected_sections` entry must be a Section of one of the expected Documents;
- at least one `answer_must_mention` term must occur in one of the expected Documents;
- every `answer_must_mention_all` term must occur in one of the expected Documents;
- every `absent_terms` entry of an out-of-scope trap must occur nowhere in the KB.

Matching is case-insensitive and ignores Arabic diacritics, tatweel and alef/yeh/teh-marbuta variants."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .scoring import PATH_SEPARATOR, Case, load_cases, norm

DEFAULT_CASES = Path(__file__).with_name("cases.json")
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


def section_names(documents: list[Any]) -> dict[str, set[str]]:
    """kb_id -> normalized Section names and "title › name" paths."""
    return {d.kb_id: {x for s in d.sections for x in (norm(s.section), norm(f"{s.title}{PATH_SEPARATOR}{s.section}"))}
            for d in documents}


def relevance_problems(cases: list[Case], texts: dict[str, str], sections: dict[str, set[str]]) -> list[str]:
    """Errors: an expected_sections entry that is no Section of any expected Document, or an
    answer_must_mention list none of whose terms occurs in any expected Document."""
    errors = []
    for case in cases:
        for i, t in enumerate(case.turns, 1):
            where = f"{case.id}" + (f" turn {i}" if case.multi_turn else "")
            for name in t.expected_sections:
                if not any(norm(name) in sections.get(kb, set()) for kb in t.expected_kb_ids):
                    errors.append(f"{where}: expected section {name!r} is in none of {list(t.expected_kb_ids)}")
            if t.answer_must_mention and not any(norm(term) in texts.get(kb, "")
                                                 for kb in t.expected_kb_ids for term in t.answer_must_mention):
                errors.append(f"{where}: none of {list(t.answer_must_mention)} occurs in {list(t.expected_kb_ids)}")
            for term in t.answer_must_mention_all:
                if not any(norm(term) in texts.get(kb, "") for kb in t.expected_kb_ids):
                    errors.append(f"{where}: {term!r} (answer_must_mention_all) occurs in none of {list(t.expected_kb_ids)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    from medrag.ingest import kb_dir, load_knowledge_base  # only here: the checks above need no medrag

    p = argparse.ArgumentParser(prog="python -m evaluation.kb_check", description=__doc__.split("\n")[0])
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--kb-dir", type=Path, help="KB .docx folder (default: $MEDRAG_KB_DIR or ./data)")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cases = load_cases(args.cases)
    documents = load_knowledge_base(args.kb_dir or kb_dir())
    texts = doc_texts(documents)
    errors, notes = anchor_problems(cases, texts)
    errors += absent_problems(cases, texts)
    errors += relevance_problems(cases, texts, section_names(documents))
    for n in notes:
        print("note:", n)
    for e in errors:
        print("ERROR:", e)
    print(f"{len(cases)} cases checked, {len(errors)} errors, {len(notes)} notes")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
