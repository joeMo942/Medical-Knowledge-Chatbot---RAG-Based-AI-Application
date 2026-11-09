"""The embedding model's files on disk: find them, or fetch them once (first start) and verify them.

BAAI/bge-m3 is pinned to one revision; the files the code needs (the XLM-R encoder, the sparse head and
the tokenizer) are listed with their size and SHA-256 (git blob SHA-1 for the small non-LFS files, as the Hub
publishes them), so a truncated or swapped download is refused. Nothing here is in git or baked into the image.

Where the files come from, in order:
  1. already in the model directory (a marker file records the verified revision): nothing to do;
  2. a seed directory (MEDRAG_EMBED_SEED, mounted at /seed in compose): copied in, e.g. from a local download;
  3. the Hugging Face Hub, file by file, resumable, each file retried with a growing pause (the network this
     was developed on drops long downloads).
`python -m medrag.modelstore` runs it (the `embedder-pull` compose service); the API calls `ensure_model` too.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from pathlib import Path

log = logging.getLogger("medrag")

MODEL_ID = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MARKER = ".verified-revision"
# file -> (size in bytes, "sha256:<hex>" for LFS files or "sha1:<git blob sha1>" for the others)
FILES: dict[str, tuple[int, str]] = {
    "pytorch_model.bin": (2271145830, "sha256:b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"),
    "tokenizer.json": (17098108, "sha256:21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"),
    "sentencepiece.bpe.model": (5069051, "sha256:cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "sparse_linear.pt": (3516, "sha256:45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9"),
    "config.json": (687, "sha1:e6eda1c72da8f9dc30fdd9b69c73d35af3b7a7ad"),
    "special_tokens_map.json": (964, "sha1:b1879d702821e753ffe4245048eee415d54a9385"),
    "tokenizer_config.json": (444, "sha1:dc69ac559dcba2694012009aaa108c614541789a"),
}
DOWNLOAD_ATTEMPTS = 8


def _digest(path: Path, algo: str) -> str:
    if algo == "sha1":  # git blob hash: sha1("blob <size>\0" + content)
        h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    else:
        h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_ok(path: Path, name: str) -> bool:
    size, want = FILES[name]
    if not path.is_file() or path.stat().st_size != size:
        return False
    algo, hexsum = want.split(":")
    return _digest(path, algo) == hexsum


def is_ready(directory: Path) -> bool:
    marker = directory / MARKER
    return marker.is_file() and marker.read_text().strip() == REVISION and all((directory / n).is_file() for n in FILES)


def _download(name: str, directory: Path) -> None:
    from huggingface_hub import hf_hub_download  # lazy: the CI install has no huggingface_hub

    last: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            hf_hub_download(MODEL_ID, name, revision=REVISION, local_dir=str(directory))
            if file_ok(directory / name, name):
                return
            raise OSError(f"{name} failed its size/hash check")
        except Exception as exc:  # network trouble of any kind: wait and resume
            last = exc
            pause = min(60, 5 * attempt)
            log.warning("download of %s failed (attempt %d/%d): %s; retrying in %ds",
                        name, attempt, DOWNLOAD_ATTEMPTS, exc, pause)
            time.sleep(pause)
    raise RuntimeError(f"could not download {name} of {MODEL_ID}@{REVISION}") from last


def ensure_model(directory: str | Path, seed: str | Path | None = None) -> Path:
    """Make `directory` hold the verified model files; returns it. Safe to call on every start."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if is_ready(directory):
        return directory
    seed_dir = Path(seed) if seed else None
    for name in FILES:
        target = directory / name
        if file_ok(target, name):
            continue
        if seed_dir and file_ok(seed_dir / name, name):
            log.info("copying %s from the seed directory %s", name, seed_dir)
            tmp = target.with_name(target.name + ".part")
            shutil.copyfile(seed_dir / name, tmp)
            os.replace(tmp, target)
        else:
            log.info("downloading %s from the Hugging Face Hub (%s@%s)", name, MODEL_ID, REVISION[:8])
            _download(name, directory)
    (directory / MARKER).write_text(REVISION + "\n")
    log.info("model %s@%s is in %s", MODEL_ID, REVISION[:8], directory)
    return directory


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    directory = os.environ.get("MEDRAG_EMBED_PATH") or "/models/bge-m3"
    seed = os.environ.get("MEDRAG_EMBED_SEED") or "/seed"
    ensure_model(directory, seed if Path(seed).is_dir() else None)


if __name__ == "__main__":
    main()
