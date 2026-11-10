"""The KB manifest: which file is which KB Document.

The source files are not named by KB ID, so the mapping is metadata kept in
`kb_manifest.json`. It also lists each KB Document's disease headings as they
appear in the file, because Word heading styles cannot be trusted to tell a
disease heading from a drug-class heading.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources

KB_ID_PATTERN = re.compile(r"^KB-[0-9]{2}$")  # [0-9], not \d: that also accepts Arabic-Indic digits


@dataclass(frozen=True)
class ManifestEntry:
    kb_id: str
    file: str
    diseases: tuple[str, ...]  # headings exactly as they appear in the file


def parse_manifest(data: dict) -> tuple[ManifestEntry, ...]:
    entries = []
    for raw in data["documents"]:
        entry = ManifestEntry(
            kb_id=raw["kb_id"], file=raw["file"], diseases=tuple(raw["diseases"])
        )
        if not KB_ID_PATTERN.match(entry.kb_id):
            raise ValueError(f"bad KB ID in manifest: {entry.kb_id!r}")
        if not entry.diseases:
            raise ValueError(f"{entry.kb_id}: manifest lists no disease headings")
        entries.append(entry)
    for field in ("kb_id", "file"):
        values = [getattr(e, field) for e in entries]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {field} in manifest")
    return tuple(entries)


def load_manifest() -> tuple[ManifestEntry, ...]:
    text = resources.files("medrag").joinpath("kb_manifest.json").read_text("utf-8")
    return parse_manifest(json.loads(text))
