"""Command line: `python -m medrag dump|ingest`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dump import render_dump
from .ingest import kb_dir, load_knowledge_base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medrag")
    parser.add_argument(
        "--kb-dir", type=Path, help=("KB .docx folder (default: $MEDRAG_KB_DIR or <repo root>/data; "
              "a non-editable install, e.g. Docker, must set MEDRAG_KB_DIR)")
    )
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="print the Section structure (headings and row counts)")
    d.add_argument("--out", type=Path, help="write to this file instead of stdout")
    sub.add_parser("ingest", help="print every Section, with its content, as JSON")
    args = parser.parse_args(argv)

    docs = load_knowledge_base(args.kb_dir or kb_dir())
    if args.command == "dump":
        text = render_dump(docs)
    else:
        text = json.dumps([d.to_dict() for d in docs], ensure_ascii=False, indent=2) + "\n"
    if getattr(args, "out", None):
        args.out.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
