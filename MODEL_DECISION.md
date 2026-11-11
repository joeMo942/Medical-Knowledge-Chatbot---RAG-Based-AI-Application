# Model decision record

| | Chosen (open weights) | Runs on |
|---|---|---|
| **LLM** | `qwen3.5:4b` (4.2 B, Q4_K_M, Apache 2.0) via Ollama | GPU |
| **Embedding** | `BAAI/bge-m3` (revision `5617a9f`), dense + sparse, fused with BM25 | CPU, in the `api` container ([ADR 0001](docs/adr/0001-target-the-authors-laptop-gpu.md)) |

`GET /health` reports both names. **Test machine:** RTX 5070 Laptop GPU (8 GB VRAM), Intel Core Ultra 9 275HX, 31 GB RAM, Windows 11 Pro with Docker Desktop (WSL 2), 1 TB SSD. Docker's data uses 59 GB of the system drive, which has about 23 GB free. All numbers below were measured on this machine, plugged in, in November 2026. Full results: [docs/test-results/](docs/test-results/README.md).

## LLM: qwen3.5:4b, compared with gemma4:e4b

Both ran on the same code, one model loaded at a time, over all five test sets (195 turns):

| | **qwen3.5:4b (chosen)** | gemma4:e4b |
|---|---|---|
| Status correct | **193/195** | 191/195 |
| Latency p50 / p95 | 3.81 / 7.62 s | **2.03 / 5.12 s** |
| VRAM / download | **4.2 / 3.3 GB** | 5.0 / 6.6 GB |
| Contract violations | 0 | 0 |

On the main 90-turn set (final run, 2026-11-11), Qwen got:
- **status 90/90**, including the 10 dev questions, and language 90/90;
- **citations 44/44 valid**, and 53/53 snippets are verbatim text from the cited Section, so **no invented sources**;
- latency p50 3.47 s and p95 6.65 s, far inside the 30 s limit, with output at 58–73 tokens/s;
- 100% on the GPU.

**Why not Gemma**, although it is about 40% faster:
- It answered a question nobody asked: after "What is the capital of France?", it answered "What are its side effects?" with Misoprostol, a drug the user never named. Qwen refused.
- It missed a Franco-Arabic metformin question and an escalation case.
- It needs more VRAM and twice the download.
- The prompts were tuned on Qwen, so Gemma would need its own tuning round.

Caveat: the main set was also used to tune the pipeline, so its score describes this set, not the hidden one.

## Embedding: BGE-M3 on the CPU

Recall at k=4 on 75 questions:

| | All (of 75) | Everyday wording, no drug name (of 26) |
|---|---:|---:|
| BM25 keyword search (first version) | 62 | 17 |
| **BGE-M3 hybrid (chosen)** | **75** | **26** |

- One model covers Arabic, dialect, Franco-Arabic and English.
- With the name mapping from #7, brand names written in Arabic letters went from 1/5 to 5/5.

**Compared with `bge-m3` served by Ollama on the GPU:**

| | **CPU in the api (chosen)** | Ollama on the GPU |
|---|---|---|
| Recall | 75/75 | 75/75 (no sparse vectors) |
| Vectors | | the same: cosine ≥ 0.9997 on all 113 Sections |
| One question | 52 ms quiet, about 133 ms average under eval load | 54 ms |
| Building the index (113 Sections) | 48 s, then cached | **10 s** |
| Extra VRAM next to the LLM | **0** | about 0.8 GB |

The results are the same, so I kept the GPU free for the LLM. Moving the embedder to the GPU would not make answers faster: embedding is about 1% of an answer's time. Writing the answer is about 2.4 s of the 3.2 s answer call.

## Without a GPU (`docker-compose.cpu.yml`)

The same models also run on the CPU only. In that mode the time limits are raised (budget 240 s, routing call 90 s). Five questions, one at a time:

| Question (CPU wording) | Status | CPU only | GPU, closest dev question (final run) |
|---|---|---:|---:|
| Panadol ingredient (warm-up) | answered | 81.8 s | - |
| Cataflam ingredient | answered | 78.0 s | 3.75 s |
| ايه البدائل التجارية لـ Brufen؟ | answered (ar) | 71.5 s | 4.28 s |
| cover letter | out_of_scope | 20.4 s | 2.35 s |
| Voltaren mg | escalated | 21.1 s | 3.14 s |

The GPU times are for the dev questions on the same topic, which are worded a little differently. All five got the right status. On the CPU, an answer takes 70–80 s (two LLM calls) and a refusal about 20 s (one call). That is fine for trying the system, but it breaks the 30 s limit, so the GPU is required for the contract.

## Not tested

Larger models, for lack of disk space. The largest candidate, Qwen3.6-35B-A3B, needs about 20 GB on its own.
