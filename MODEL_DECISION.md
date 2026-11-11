# Model decision record

| | Chosen | Runs on |
|---|---|---|
| **LLM** | `qwen3.5:4b` (4.2 B parameters, Q4_K_M, Apache 2.0) via Ollama | GPU |
| **Embedding** | `BAAI/bge-m3` (pinned revision `5617a9f`), dense + sparse, fused with BM25 | CPU, inside the `api` container ([ADR 0001](docs/adr/0001-target-the-authors-laptop-gpu.md)) |

Both are open-weights models. `GET /health` reports both names.

## Test hardware

| Part | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop, **8 GB VRAM** |
| CPU | Intel Core Ultra 9 275HX |
| RAM | 31 GB |
| Disk | 1 TB SSD; the system drive C: has **about 23 GB free**, and Docker's data (images, models, volumes) already uses 59 GB of it |
| OS | Windows 11 Pro, Docker Desktop (WSL 2) |

## Why qwen3.5:4b: what I measured

On the 90-turn test set (final run `final`, 2026-11-11, the 10 dev questions plus my own cases in Arabic, Egyptian dialect, Franco-Arabic and English):

- **Status: 90/90 correct**, including all 10 dev questions. Language label: 90/90.
- **Citations:** 44/44 valid, and 53/53 snippets are verbatim text from the cited Section. **No invented sources.**
- **0 API contract violations.**
- **Speed:** latency per `/chat` is p50 3.47 s and p95 6.65 s in the final run (2026-11-11; laptop plugged in, Windows Balanced power mode, the GPU sometimes power-capped; the earlier clean-clone run had 3.68 / 6.04 s), far inside the 30 s limit. Output runs at 58–73 tokens/s.
- **Memory:** 3.4 GB loaded and 100% on the GPU, so it fits the 8 GB card with room for its context.

Caveat: these 90 turns were also used to tune the pipeline, so the score describes this test set, not the hidden one. All five test sets and their results: [docs/test-results/](docs/test-results/README.md).

## Why BGE-M3: what I measured

Recall test with 75 questions (re-run 2026-11-11, results in [docs/test-results/](docs/test-results/README.md)), counting a question as found when the right Section is among the 4 retrieved:

| Search | Found (of 75) | Everyday wording, no drug name (of 26) |
|---|---:|---:|
| BM25 keyword search (first version) | 62 | 17 |
| **BGE-M3 hybrid (chosen)** | **75** | **26** |

- **Multilingual:** it handles Arabic, dialect, Franco-Arabic and English in one model. Questions with brand names written in Arabic letters went from 1/5 to 5/5 correct (with the name mapping added in #7).
- **Speed:** about 133 ms per question on the CPU, which keeps the GPU free for the LLM.

## What else I tried

A small benchmark on 2026-11-11, with the laptop plugged in. Each alternative was run against the chosen model on the same code, one model loaded at a time. The full tables are in [docs/test-results/benchmark/](docs/test-results/benchmark/README.md).

**LLM: `gemma4:e4b` against `qwen3.5:4b`**, both through Ollama, on all five test sets (195 turns):

| | qwen3.5:4b (chosen) | gemma4:e4b |
|---|---|---|
| Status correct | **193/195** | 191/195 |
| Latency p50 / p95 | 3.81 / 7.62 s | **2.03 / 5.12 s** |
| VRAM | **4.2 GB** | 5.0 GB |
| Download | **3.3 GB** | 6.6 GB |
| Contract violations | 0 | 0 |

Gemma is about 40% faster. I kept Qwen for these reasons:
- **Gemma answered something nobody asked.** After "What is the capital of France?", it answered "What are its side effects?" with the side effects of Misoprostol, a drug the user never named. Qwen refused it.
- **Gemma missed some questions.** It also missed a Franco-Arabic metformin question and an escalation case.
- **Fit.** Qwen needs less VRAM and half the download.
- **Tuning.** The prompts were tuned on Qwen, so Gemma would need its own tuning round.

**Embedding: `bge-m3` served by Ollama on the GPU against BGE-M3 in the api on the CPU:**
- **Same results.** Recall was the same (75/75), the dense vectors were practically identical (cosine of at least 0.9997 on all 113 Sections), and a question took about the same time to embed (54 ms against 52 ms).
- **Faster start.** Ollama builds the index in 10 s instead of 48 s.
- **The cost.** It adds about 0.8 GB of VRAM next to the LLM, and Ollama returns no sparse vectors.

I kept the CPU model so the GPU stays free for the LLM.

I did not test larger models because of disk space. The two benchmark models took the free space on the drive Docker uses down to about 23 GB, and the largest candidate I considered, Qwen3.6-35B-A3B, needs about 20 GB on its own.
