# Small benchmark (2026-11-11)

Two questions, measured on the author's laptop (RTX 5070 Laptop 8 GB, on AC power, one request at a time):

1. **LLM:** `qwen3.5:4b` (shipped) against `gemma4:e4b`, both through Ollama 0.40.15, on all five test sets.
2. **Embedder:** BGE-M3 in the api process on the CPU (shipped) against `bge-m3` served by Ollama on the GPU.

Nothing in the shipped defaults changed. Both LLM arms ran on the same api image (branch `nogpu-startup`, commit `4025bc7`, whose GPU settings equal `main`'s) with `MEDRAG_USAGE_HEADER=1`. The model was swapped only at start-up: `MEDRAG_LLM_MODEL=gemma4:e4b docker compose up -d --wait`, after `ollama stop qwen3.5:4b`, so only one LLM was loaded at a time. Both ran at `100% GPU` (`ollama ps`).

## 1. LLM: qwen3.5:4b against gemma4:e4b

| set | model | status | language | citations (expected doc) | snippets valid | relevance | contract | p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | qwen | **90/90** | 90/90 | 44/44 | 53/53 | 42/44 | 0 | 3.81 / 7.15 |
| main | gemma | 88/90 | 90/90 | 43/44 | 93/93 | 43/44 | 0 | **2.32 / 4.65** |
| issue5 | qwen | 16/17 | 17/17 | 6/7 | 7/7 | 6/7 | 0 | 4.75 / 5.98 |
| issue5 | gemma | 16/17 | 17/17 | 6/7 | 19/19 | 6/7 | 0 | **2.29 / 3.75** |
| issue7 | qwen | 20/21 | 21/21 | 10/11 | 12/12 | 10/11 | 0 | 4.11 / 9.46 |
| issue7 | gemma | 20/21 | 21/21 | 11/11 | 38/38 | 11/11 | 0 | **2.45 / 5.73** |
| issue8 | qwen | 9/9 | 9/9 | 5/5 | 5/5 | 5/5 | 0 | 4.04 / 5.36 |
| issue8 | gemma | 9/9 | 9/9 | 5/5 | 6/6 | 5/5 | 0 | **2.58 / 3.64** |
| issue32 | qwen | 58/58 | 58/58 | 18/19 | 50/50 | 12/19 | 0 | 0.87 / 7.69 |
| issue32 | gemma | 58/58 | 58/58 | 18/19 | 69/69 | 10/19 | 0 | **0.46 / 5.76** |

All 195 turns:

| | qwen3.5:4b | gemma4:e4b |
| --- | --- | --- |
| status right | **193/195** | 191/195 |
| latency p50 / p95 / max | 3.81 / 7.62 / 14.55 s | **2.03 / 5.12 / 7.46 s** |
| Answered turns p50 / p95 | 5.57 / 8.32 s | **3.38 / 5.78 s** |
| tokens per turn (prompt / completion, main set) | 3,673 / 139 | 3,705 / 161 |
| VRAM, model loaded alone (`nvidia-smi`) | **4.2 GB** | 5.0 GB |
| non-200 responses, contract violations | 0, 0 | 0, 0 |
| download size | **3.3 GB** | 6.6 GB |

Gemma cites more Sections per answer (snippets 93 against 53 on the main set, all valid).

### The four turns where the two differ on status

| turn | expected | qwen | gemma |
| --- | --- | --- | --- |
| "What is the capital of France?" then **"What are its side effects?"** | out_of_scope | out_of_scope | **answered**, with Misoprostol's side effects: a drug nobody named |
| "Can I take Panadol at 3pm?" (a timing question) | escalated | escalated | out_of_scope |
| "el metformin mamnoo3 emta?" (Franco-Arabic: when is metformin contraindicated) | answered | answered | out_of_scope |
| "dawa lel daghT el 3aly" (Franco-Arabic: a medicine for high blood pressure) | answered | out_of_scope | answered |

Gemma also wrote the citation path into the answer text once ("... (KB-08 › أدوية الضغط – Antihypertensive Drugs › ACE Inhibitors)").

**Reading:** Gemma is about 40% faster on the GPU, but it answered one unanchored follow-up with a drug the user never named, which is a faithfulness failure. The prompts and word lists were tuned on Qwen, so a fair Gemma result would need its own tuning round. Qwen uses less VRAM and half the download.

## 2. Embedder: BGE-M3 on the CPU against bge-m3 through Ollama

Run inside the api container (`scripts/embed_bench.py`), with `qwen3.5:4b` loaded on the GPU. Raw output: `2026-11-11-embedder.txt`.

| | BGE-M3 in-process, CPU (shipped) | bge-m3 through Ollama, GPU |
| --- | --- | --- |
| recall at k=4, 75 turns (`evaluation.recall` scoring) | **75/75**, dense + sparse + BM25 | **75/75**, dense + BM25 (Ollama returns no sparse vectors) |
| same, local dense + BM25 without sparse | 75/75 | - |
| embedding one question (median of 26) | 52 ms (4 threads, the shipped setting) | 54 ms |
| building the index of 113 Sections | 47.6 s (all cores) | **9.8 s** |
| dense vectors, cosine between the two (113 Sections) | - | mean 1.0000, min 0.9997 |
| extra VRAM next to qwen3.5:4b | **0** | about 0.8 GB (4.2 → 5.1 GB total; both stayed at `100% GPU`) |
| extra download | 2.3 GB (Hugging Face) | 1.2 GB (Ollama) |

**Reading:** the two produce the same dense vectors and the same recall, and a question embeds in the same time. Ollama builds the index about 5 times faster, but it costs about 0.8 GB of VRAM on an 8 GB card, puts the embedder in the LLM's queue, and drops the sparse list (no recall loss on this test set).

## Re-running

```
sh scripts/run_sets.sh bench-qwen                                  # stack on qwen3.5:4b, MEDRAG_USAGE_HEADER=1
docker exec ollama-container ollama stop qwen3.5:4b
MEDRAG_USAGE_HEADER=1 MEDRAG_LLM_MODEL=gemma4:e4b docker compose up -d --wait
sh scripts/run_sets.sh bench-gemma
python scripts/compare.py <results dir> bench-qwen bench-gemma
docker cp evaluation <api>:/tmp/bench/ && docker cp scripts/embed_bench.py <api>:/tmp/bench/
docker exec -e PYTHONPATH=/tmp/bench -w /tmp/bench <api> python embed_bench.py
```

`run_sets.sh` and `compare.py` have the author's scratch folder and the date written in; edit them before reuse.
