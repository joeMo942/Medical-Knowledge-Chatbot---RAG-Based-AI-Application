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
| Disk | 1 TB SSD; the system drive C: has **27 GB free**, and Docker's data (images, models, volumes) already uses 59 GB of it |
| OS | Windows 11 Pro, Docker Desktop (WSL 2) |

## Why qwen3.5:4b: what I measured

On the 90-turn test set (the 10 dev questions plus my own cases in Arabic, Egyptian dialect, Franco-Arabic and English):

- **Status: 90/90 correct**, including all 10 dev questions. Language label: 90/90.
- **Citations:** 44/44 valid, and 53/53 snippets are verbatim text from the cited Section. **No invented sources.**
- **0 API contract violations.**
- **Speed:** latency per `/chat` is p50 3.68 s and p95 6.04 s, far inside the 30 s limit. Output runs at 60–73 tokens/s.
- **Memory:** 3.4 GB loaded and 100% on the GPU, so it fits the 8 GB card with room for its context.

Caveat: these 90 turns were also used to tune the pipeline, so the score describes this test set, not the hidden one.

## Why BGE-M3: what I measured

Recall test with 75 questions, counting a question as found when the right Section is among the 4 retrieved:

| Search | Found (of 75) | Everyday wording, no drug name (of 26) |
|---|---:|---:|
| BM25 keyword search (first version) | 62 | 17 |
| **BGE-M3 hybrid (chosen)** | **75** | **26** |

- **Multilingual:** it handles Arabic, dialect, Franco-Arabic and English in one model. Questions with brand names written in Arabic letters went from 1/5 to 5/5 correct (with the name mapping added in #7).
- **Speed:** about 133 ms per question on the CPU, which keeps the GPU free for the LLM.

## Why I did not try other models

1. **The results already met the brief.** The system scored 90/90 with no invented citations, at p95 6 s against a 30 s limit, and fits the 8 GB GPU. Nothing failed that another model was needed to fix.
2. **Storage.** The project already takes about 16 GB on disk. Only 27 GB is free on the drive Docker uses. The other candidates I considered would each add several GB, and the largest (Qwen3.6-35B-A3B) about 20 GB on its own. Downloading and testing them would have meant freeing disk space that this machine does not have.

So **no other LLM or embedding model was tested, and none was rejected on measured evidence.** The only comparison made is BGE-M3 hybrid search against the BM25 search it replaced (the table above). Comparisons that are still open: Gemma 4 E4B as the LLM, and BGE-M3 served through Ollama instead of CPU torch.
