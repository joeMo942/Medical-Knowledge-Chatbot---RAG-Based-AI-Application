"""Embedder benchmark: BGE-M3 in-process on the CPU (shipped) vs bge-m3 served by Ollama (GPU), on the same KB.

Run inside the api container (it has torch, /models/bge-m3 and reaches ollama:11434):
    PYTHONPATH=/tmp/bench python /tmp/bench/embed_bench.py
Measures: index build time, query embedding latency, dense-vector parity (cosine per Section), and search recall at k=4
with evaluation.recall's own scoring for three retrievers:
  local-hybrid      shipped: local dense + local sparse + BM25 (weights 3/1/1)
  local-no-sparse   local dense + BM25 (3/0/1): what is lost by dropping the sparse list alone
  ollama-hybrid     Ollama dense + BM25 (3/0/1): Ollama's bge-m3 returns dense vectors only
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
import urllib.request

from evaluation.recall import DEFAULT_CASES, DEFAULT_QUERIES, build_names, evaluate, items_from, make_query_of, report
from evaluation.scoring import load_cases
from medrag.embedding import BGEM3Embedder, Encoded
from medrag.hybrid import DOC_MAX_LENGTH, HybridRetriever
from medrag.ingest import kb_dir, load_knowledge_base

OLLAMA = "http://ollama:11434"


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


class OllamaEmbedder:
    model_id, revision = "ollama:bge-m3", ""

    def __init__(self):
        self.calls = []

    def encode(self, texts, max_length=512):
        body = json.dumps({"model": "bge-m3", "input": list(texts), "keep_alive": "24h", "truncate": True,
                           "options": {"num_ctx": max(max_length, 512)}}).encode()
        req = urllib.request.Request(f"{OLLAMA}/api/embed", body, {"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        self.calls.append(time.monotonic() - t0)
        n = len(texts) or 1
        return [Encoded(unit(e), {}, data.get("prompt_eval_count", 0) // n) for e in data["embeddings"]]


def main():
    sections = [s for d in load_knowledge_base(kb_dir()) for s in d.sections]
    texts = [s.render() for s in sections]
    out = {"sections": len(sections)}

    t0 = time.monotonic(); local = BGEM3Embedder("/models/bge-m3"); out["local_load_s"] = round(time.monotonic() - t0, 1)
    oll = OllamaEmbedder()
    oll.encode(["warm-up"])  # loads bge-m3 into Ollama
    t0 = time.monotonic(); L = local.encode(texts, max_length=DOC_MAX_LENGTH); out["local_index_s"] = round(time.monotonic() - t0, 1)
    t0 = time.monotonic(); O = []
    for i in range(0, len(texts), 8):
        O += oll.encode(texts[i:i + 8], max_length=DOC_MAX_LENGTH)
    out["ollama_index_s"] = round(time.monotonic() - t0, 1)
    cos = [sum(a * b for a, b in zip(unit(l.dense), o.dense)) for l, o in zip(L, O)]
    out["parity_cosine"] = {"mean": round(statistics.mean(cos), 4), "min": round(min(cos), 4),
                            "below_0.99": sum(c < 0.99 for c in cos)}

    queries = json.loads(DEFAULT_QUERIES.read_text(encoding="utf-8"))
    qs = [q["message"] for q in queries][:30]
    lt, ot = [], []
    for q in qs:
        t0 = time.monotonic(); local.encode([q]); lt.append(time.monotonic() - t0)
        t0 = time.monotonic(); oll.encode([q]); ot.append(time.monotonic() - t0)
    out["query_ms_median"] = {"local_cpu": round(1000 * statistics.median(lt), 1), "ollama": round(1000 * statistics.median(ot), 1)}
    qcos = []
    for q in qs[:10]:
        a, b = local.encode([q])[0], oll.encode([q])[0]
        qcos.append(sum(x * y for x, y in zip(unit(a.dense), b.dense)))
    out["query_parity_cosine_min"] = round(min(qcos), 4)
    print(json.dumps(out, indent=1), flush=True)

    cases = [c for p in DEFAULT_CASES for c in load_cases(p)]
    items = items_from(cases, queries)
    query_of = make_query_of(build_names(sections), True)
    lines = []
    for name, factory, weights in [("local-hybrid", lambda: local, (3.0, 1.0, 1.0)),
                                   ("local-no-sparse", lambda: local, (3.0, 0.0, 1.0)),
                                   ("ollama-hybrid", lambda: oll, (3.0, 0.0, 1.0))]:
        r = HybridRetriever(sections, factory, name, min_dense=0.5, weights=weights)
        r.load()
        t0 = time.monotonic()
        rows, quiet = evaluate(r, items, cases, 4, query_of)
        lines += report(name, 4, rows, quiet, None) + [f"(search time {time.monotonic() - t0:.1f}s)", ""]
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
