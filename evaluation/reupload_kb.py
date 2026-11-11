"""Deletes every KB Document through the admin API and uploads the originals again (headings from medrag/kb_manifest.json).

    python -m evaluation.reupload_kb [--base-url http://localhost:8000] [--data-dir data] [--keep]

Checks the admin endpoints end to end: after the delete a question must be Out of Scope; then each upload is timed.
`--keep` skips the delete (a plain replace of every document)."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What is the active ingredient of Controloc?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--keep", action="store_true", help="do not delete first")
    args = ap.parse_args(argv)
    manifest = json.loads((ROOT / "medrag" / "kb_manifest.json").read_text("utf-8"))["documents"]
    with httpx.Client(base_url=args.base_url, timeout=300) as c:
        if not args.keep:
            for d in c.get("/admin/documents").json()["documents"]:
                r = c.delete(f"/admin/documents/{d['kb_id']}")
                print("delete", d["kb_id"], r.status_code)
            status = c.post("/chat", json={"message": QUESTION}).json().get("status")
            print(f"after deleting everything: {QUESTION!r} -> {status}")
        for e in manifest:
            body = {"kb_id": e["kb_id"], "diseases": e["diseases"], "filename": e["file"],
                    "content_base64": base64.b64encode((args.data_dir / e["file"]).read_bytes()).decode()}
            t0 = time.perf_counter()
            r = c.post("/admin/documents", json=body)
            print(f"upload {e['kb_id']}: {r.status_code} in {time.perf_counter() - t0:.1f}s {r.text[:120]}")
            if r.status_code != 200:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
