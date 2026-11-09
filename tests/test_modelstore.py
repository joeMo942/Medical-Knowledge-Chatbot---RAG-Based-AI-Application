"""Model files: verification, seeding from a local folder, and the download with retries (huggingface_hub faked)."""
import hashlib
import sys
import types

import pytest

from medrag import modelstore as ms


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


@pytest.fixture
def tiny(monkeypatch):
    """Two tiny 'model files': one LFS-style (sha256), one plain (git blob sha1)."""
    big, small = b"weights" * 10, b'{"a": 1}'
    files = {"pytorch_model.bin": (len(big), "sha256:" + hashlib.sha256(big).hexdigest()),
             "config.json": (len(small), "sha1:" + blob_sha1(small))}
    monkeypatch.setattr(ms, "FILES", files)
    monkeypatch.setattr(ms.time, "sleep", lambda s: None)
    return {"pytorch_model.bin": big, "config.json": small}


def test_file_ok_checks_size_and_hash(tmp_path, tiny):
    for name, data in tiny.items():
        (tmp_path / name).write_bytes(data)
        assert ms.file_ok(tmp_path / name, name)
    (tmp_path / "config.json").write_bytes(b'{"a": 2}')  # same size, other content
    assert not ms.file_ok(tmp_path / "config.json", "config.json")
    assert not ms.file_ok(tmp_path / "missing", "config.json")


def test_seed_directory_is_copied_instead_of_downloaded(tmp_path, tiny, monkeypatch):
    seed, dest = tmp_path / "seed", tmp_path / "models"
    seed.mkdir()
    for name, data in tiny.items():
        (seed / name).write_bytes(data)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # any download attempt would fail
    assert ms.ensure_model(dest, seed) == dest
    assert ms.is_ready(dest) and (dest / "pytorch_model.bin").read_bytes() == tiny["pytorch_model.bin"]
    assert ms.ensure_model(dest, None) == dest  # a verified directory is left alone


def test_a_corrupt_seed_file_is_not_used(tmp_path, tiny, monkeypatch):
    seed, dest = tmp_path / "seed", tmp_path / "models"
    seed.mkdir()
    (seed / "config.json").write_bytes(b"x" * len(tiny["config.json"]))
    (seed / "pytorch_model.bin").write_bytes(tiny["pytorch_model.bin"])
    calls = []

    def fake_download(repo, name, revision, local_dir):
        calls.append((repo, name, revision))
        (dest / name).write_bytes(tiny[name])

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    ms.ensure_model(dest, seed)
    assert calls == [(ms.MODEL_ID, "config.json", ms.REVISION)]  # only the bad one was fetched, at the pinned revision


def test_download_is_retried_and_a_bad_download_is_refused(tmp_path, tiny, monkeypatch):
    dest = tmp_path / "models"
    attempts = {"n": 0}

    def flaky(repo, name, revision, local_dir):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("reset by peer")
        (dest / name).write_bytes(tiny[name])

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=flaky))
    ms.ensure_model(dest)
    assert ms.is_ready(dest) and attempts["n"] == 2 + 2  # config.json failed twice; each file then succeeded once

    bad = tmp_path / "bad"
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        hf_hub_download=lambda repo, name, revision, local_dir: (bad / name).write_bytes(b"corrupt")))
    with pytest.raises(RuntimeError):
        ms.ensure_model(bad)
    assert not ms.is_ready(bad)
