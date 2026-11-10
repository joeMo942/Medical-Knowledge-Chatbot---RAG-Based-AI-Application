"""The real BGE-M3 on the CPU. Needs torch, transformers and the model files (MEDRAG_EMBED_PATH, default
~/.cache/medrag/bge-m3, or /models/bge-m3 in the image); skipped where they are absent, so CI never downloads anything."""
import os
from pathlib import Path

import pytest

from medrag import modelstore

CANDIDATES = [os.environ.get("MEDRAG_EMBED_PATH"), str(Path.home() / ".cache" / "medrag" / "bge-m3"), "/models/bge-m3"]
MODEL_DIR = next((Path(p) for p in CANDIDATES if p and all((Path(p) / n).is_file() for n in modelstore.FILES)), None)

pytestmark = pytest.mark.skipif(MODEL_DIR is None, reason="BGE-M3 files not found (set MEDRAG_EMBED_PATH)")


@pytest.fixture(scope="module")
def embedder():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from medrag.embedding import BGEM3Embedder

    return BGEM3Embedder(MODEL_DIR)


def test_dense_and_sparse_outputs(embedder):
    from medrag.embedding import dense_score, sparse_score

    a, b, c = embedder.encode(["What are the trade names of Ibuprofen?", "Brufen, Nurofen: Ibuprofen tablets",
                               "Salbutamol inhaler for asthma"])
    assert len(a.dense) == 1024 and abs(dense_score(a.dense, a.dense) - 1.0) < 1e-3
    assert dense_score(a.dense, b.dense) > dense_score(a.dense, c.dense)
    assert sparse_score(a.sparse, b.sparse) > sparse_score(a.sparse, c.sparse)
    assert a.sparse and all(w > 0 for w in a.sparse.values())
