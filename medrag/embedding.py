"""Embedders: text -> a dense vector and a sparse (token weight) vector.

`Embedder` is the small interface the retriever needs, so another model (or an Ollama-served bge-m3 that
only has the dense half) can be swapped in. `BGEM3Embedder` is the real one: BAAI/bge-m3 on the CPU with
torch and transformers (imported lazily, so the rest of the package and CI never load them).

BGE-M3's two outputs, as in the FlagEmbedding reference implementation:
  dense  = the L2-normalised hidden state of the first token ([CLS]);
  sparse = relu(Linear(hidden state)) per token, the largest weight per distinct token id, special tokens left out.
Dense similarity is the dot product of the vectors, sparse similarity the sum of weight products over shared tokens.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .modelstore import MODEL_ID, REVISION


@dataclass(frozen=True)
class Encoded:
    dense: tuple[float, ...]
    sparse: dict[int, float]


class Embedder(Protocol):
    model_id: str
    revision: str

    def encode(self, texts: list[str], max_length: int = 512) -> list[Encoded]:
        """One Encoded per text, in order. May block for seconds (CPU): call it off the event loop."""
        ...


def dense_score(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sparse_score(a: dict[int, float], b: dict[int, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(w * b[t] for t, w in a.items() if t in b)


BULK_TEXTS = 4  # more texts than this is a bulk job (the index), fewer is a question


class BGEM3Embedder:
    model_id = MODEL_ID
    revision = REVISION

    def __init__(self, model_dir: str | Path, threads: int | None = None, batch_size: int = 8):
        """`threads`: CPU threads for a question (a few short texts). The index build (many long texts) uses all
        cores: measured in the container, a one-question forward pass is about 4x slower on 24 threads than on 4
        (the work is too small to share out), while the index build gains from every core."""
        import os

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._bulk_threads = os.cpu_count() or 1
        self._query_threads = threads or self._bulk_threads
        directory = str(model_dir)
        self._tok = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self._model = AutoModel.from_pretrained(directory, local_files_only=True).eval()
        head = torch.load(Path(model_dir) / "sparse_linear.pt", map_location="cpu", weights_only=True)
        self._sparse = torch.nn.Linear(self._model.config.hidden_size, 1)
        self._sparse.load_state_dict(head)
        self._special = set(self._tok.all_special_ids)
        self._batch = batch_size
        self._lock = threading.Lock()  # one forward pass at a time; the CPU threads are already busy with it

    def encode(self, texts: list[str], max_length: int = 512) -> list[Encoded]:
        torch = self._torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths share a batch: less padding
        out: list[Encoded | None] = [None] * len(texts)
        with self._lock, torch.inference_mode():
            torch.set_num_threads(self._bulk_threads if len(texts) > BULK_TEXTS else self._query_threads)
            for start in range(0, len(order), self._batch):
                idx = order[start:start + self._batch]
                enc = self._tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                                return_tensors="pt")
                hidden = self._model(**enc).last_hidden_state
                dense = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
                weights = torch.relu(self._sparse(hidden)).squeeze(-1)
                for row, i in enumerate(idx):
                    sparse: dict[int, float] = {}
                    for tid, w in zip(enc["input_ids"][row].tolist(), weights[row].tolist()):
                        if tid in self._special or w <= 0:
                            continue
                        if w > sparse.get(tid, 0.0):
                            sparse[tid] = w
                    out[i] = Encoded(tuple(dense[row].tolist()), sparse)
        return out  # type: ignore[return-value]
