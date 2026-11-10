# Model decision record

**Chosen:** LLM `qwen3.5:4b` (Ollama, Apache 2.0 per `ollama show qwen3.5:4b --license`), embedding model `BAAI/bge-m3` (pinned revision
`5617a9f61b028005a4858fdac845db406aefb181`, open weights). Both are open-weights models, as the brief requires. `GET /health` reports both names.

## Summary (one page)

**Why qwen3.5:4b.** It fits the 8 GB laptop GPU with room to spare and is fast enough for the 30 s limit: 3.4 GB loaded, 100% on the GPU (`ollama ps`), decode 60 to 73 tokens/s, latency per `/chat` p50 3.68 s and p95 6.04 s over the 90-turn test set. It is Apache 2.0 and one `ollama pull`. The argument is "sufficient and measured", not "shown best".

**What the data shows.** With this model fixed, status accuracy on the 90 turns went from 55/90 (BM25 plus one answer call) to 90/90 as the pipeline grew: rules and a routing call (83/90), the support check (85/90), Sessions (86/90), hybrid search (86/90), Arabic and Franco-Arabic work (90/90); the clean-clone run of the submission also gave 90/90. At the baseline the model never escalated (0 of 28), so the 90/90 is a property of the system, not of the model: fixed rules and server-side checks keep the safety decisions, and a 4B model is enough under that arrangement. The 90/90 is on the set the pipeline was tuned against, not a forecast for the hidden set.

**Why BGE-M3.** Questions come in Arabic, dialect, Franco-Arabic and English; it is multilingual open weights and gives dense and sparse vectors in one pass, fused with BM25. On the offline recall set (75 questions, k=4): hybrid 75, improved BM25 71, original BM25 62; on 26 everyday-wording questions 26, 23 and 17. It runs on the CPU so the GPU stays free for the LLM (ADR 0001): one question embeds in about 50 ms (133 ms mean in the eval run), the index of 113 Sections builds in 42.7 s on the first start.

**What was not compared, and why.** No head-to-head benchmark of other LLMs was run (issue #10 was skipped), so no other model was rejected on evidence. No other embedding model was compared; only BGE-M3 against BM25. Open comparison, **not done:** BGE-M3 served through Ollama instead of in-process CPU torch. Open-weights only, as the brief requires.

## Appendix: measurements

### Read this first: what was and was not compared

- **No head-to-head benchmark of other LLMs was run.** The benchmark (issue #10) was skipped. Nothing here says qwen3.5:4b beats any other model; it says what
  this model did on this test set inside this pipeline, and why it fits the hardware. No other LLM was tried, so none was rejected on evidence.
- **No other embedding model was compared either.** What was measured is BGE-M3 hybrid search against the BM25 search that came before it (below).
- **Open comparison, not done:** serving BGE-M3 through Ollama instead of in-process CPU `torch`. It might free the CPU and simplify the image (no torch, 1.65 GB down to
  something near the 258 MB image of the first version), at a cost nobody measured: GPU memory next to the LLM, load/unload when the two models swap, and whether
  Ollama can return the sparse vectors the hybrid search uses (as far as I know it returns dense embeddings only; not checked).
- The 90-turn test set was used to tune the pipeline (rules, router examples, support check), so the 90/90 below says the pipeline passes the cases it was developed against, not that it
  will pass the hidden question set. `cases_issue5.json` and `cases_issue7.json` are small held-out sets, written by the same author.

### The constraints that decided it

- **Hardware (ADR 0001):** one RTX 5070 Laptop GPU with 8 GB of VRAM, 31 GB RAM, Core Ultra 9 275HX. Docker Desktop on Windows.
- **Time:** the brief allows 30 s per `/chat`; the server keeps to a 25 s budget (`MEDRAG_BUDGET_S`).
- **Languages:** Arabic (formal and Egyptian dialect), English, Franco-Arabic, mixed.
- **Rule:** open weights only; no API key anywhere.

### LLM: qwen3.5:4b

Measured facts (sources are the local eval results, `evaluation/results/`, and `ollama show`, `ollama ps`, `nvidia-smi`):

- 4.2 B parameters, Q4_K_M, 3.3 GB on disk, **3.4 GB loaded, 100% on the GPU** (`ollama ps`), context 8192, so the model, its KV cache and the desktop fit in 8 GB (4.2 GB of the card's 8 GB in use during a run, all processes).
- Speed on that GPU: decode **60 to 73 tokens/s**, prefill 9,000 to 15,000 tokens/s (TOKEN_USAGE.md). Latency per `/chat` over the 90-turn set: **p50 3.68 s, p95 6.04 s, max 7.23 s** (`issue13-clean`), far inside 30 s.
- Tokens per turn: 3,807.5 on average across routing, answer and embedding calls; per call kind in TOKEN_USAGE.md.
- Thinking is switched off (`"think": false`) and the temperature is 0, so the same question gives the same tokens; the model's reasoning mode would add latency the budget cannot afford.
- It writes Arabic and English acceptably for the lookup task: the replies in the 90-turn set are in the expected Answer Language 90/90 (`issue7e`, `issue13-clean`), counting the label, and 83/90 before the Arabic work.

#### How the eval score moved with this model held fixed

Same model (`qwen3.5:4b`), same 90 turns of `evaluation/cases.json`, status accuracy per turn; each row is the pipeline as it stood at that run (from the `.md` files of the runs):

| run (results file) | pipeline at that point | status accuracy | latency p50 / p95 | LLM prompt tokens per turn | LLM calls per turn |
| --- | --- | ---: | --- | ---: | ---: |
| `baseline-r5` (2026-11-09, `2d30f05`) | walking skeleton: BM25 search and one answer call, no routing, no escalation rules | **55/90** | 2.97 / 6.12 s | 1,403 | 0.96 |
| `issue4c` | Escalation rules + routing call (#4) | **83/90** | 3.62 / 7.07 s | 2,722 | 1.57 |
| `issue5b` | support check, Out of Scope, partly covered questions (#5) | **85/90** | 4.13 / 7.25 s | 2,794 | 1.57 |
| `issue8b` | Sessions and rewrite validation (#8) | **86/90** | 2.89 / 5.09 s | 3,261 | 1.57 |
| `issue6e` (2026-11-10) | BGE-M3 hybrid search (#6) | **86/90** | 3.07 / 5.38 s | 3,295 | 1.54 |
| `issue7e` | Arabic, dialect, Franco-Arabic (#7) | **90/90** | 3.52 / 5.87 s | 3,646 | 1.58 |
| `issue9a`, `issue11a` | logging and analytics (#9), admin page (#11); no change to answers | 90/90, 90/90 | 3.65 / 5.87 s, 3.80 / 6.20 s | 3,646 | 1.58 |
| `issue13-clean` (2026-11-10, `95f009b`, fresh clone, fresh volumes) | the submission | **90/90** | 3.68 / 6.04 s | 3,646 | 1.58 |

(No separate results file for the hand-off work (#12) exists on this machine; it does not touch answers. The earlier `baseline` file of 2026-11-06 is of the first test set, 88 turns, and is not comparable.)

What this table does and does not say:

- At the baseline the model on its own **never escalated** (the confusion matrix of `baseline-r5`: of the 28 turns that must be Escalated, 11 came back Answered, 17 Out of Scope, 0 Escalated),
  and it answered Arabic-letter Trade Names Out of Scope (1/5). The 4B model cannot be trusted with the safety decisions alone, and the pipeline does not ask it to: fixed rules decide Emergency and Dose wording first, a
  routing call may only raise a status, and the server checks Citations, snippets, doses and support after the model speaks (README, Safety policy).
- So **the 90/90 is a property of the system, not of the model**, and the model decision cannot be separated from the pipeline in this data. The data does support "a 4B model is enough when the server keeps the safety decisions".
- Failure patterns still seen with this model, recorded in the known-limitations issue (#16): the router can classify a rare wording differently from run to run; a grounded answer can contain one invented list item
  (a Metformin answer listed "hypersensitivity" that its Section does not say); a Trade Name can be misspelled in the answer text; it sometimes names the wrong Sections in prose while its Citations are right.
  The faithfulness review is where a person catches these; the sheets for the runs are generated by `python -m evaluation.review` and are judged by hand, never by an LLM.

#### Why this model, in terms of what was observed

- It is the model the whole project was built and measured with from the first run; its decode speed and VRAM use leave the 30 s limit and the 8 GB card comfortable (p95 6 s, 3.4 GB), so there was no latency or memory
  pressure that would justify a trial of another model. That is the whole argument: **sufficient and measured, not shown best.**
- A much larger model would leave little of the 8 GB for the KV cache at 8192 context (reasoning, not a measurement), and a smaller one would be faster but this one was already well inside the limit. Neither was tried.
- Open-weights Apache 2.0, one `ollama pull`, no key; swapping the model is one variable (`MEDRAG_LLM_MODEL`), so the comparison below can be run later without code changes.

#### What a proper benchmark would be (not done)

The same 90 turns plus the held-out sets (`--cases`), the same pipeline, `MEDRAG_LLM_MODEL=<other tag>`, comparing status accuracy, relevance, language accuracy, invented Citations (the runner already checks
snippets and doc IDs), latency p95 and tokens, plus a faithfulness review of the same sample. At least one smaller and one larger open model.

### Embedding model: BAAI/bge-m3

- **Why a multilingual embedding at all:** questions come in Arabic, dialect, Franco-Arabic and English, and the KB is in Arabic and English mixed; a lexical search alone misses everyday wording
  ("عندي سخونية" against a KB that says "الحرارة").
- **Why this one:** it is multilingual (Arabic and English in one vector space), open weights, and gives a **dense and a sparse (token-weight) vector** from one forward pass, so
  the search is dense + sparse + BM25 fused by reciprocal rank (docs/search-and-arabic.md). That choice was made on those properties; **no other embedding model was run for comparison.**
- **Where it runs:** on the CPU inside the `api` container, so the GPU belongs to the LLM (ADR 0001): model load 7.7 s, one question in about 50 ms on 4 threads (133 ms mean during the eval run), whole search under 0.4 s,
  index of 113 Sections built in 42.7 s on the first start and cached after that. It needs about 2.5 GB of RAM. The cost is the image (CPU `torch`, 1.65 GB) and the 2.3 GB weight download on the first start.

Measured recall (README Search section and docs/search-and-arabic.md; `python -m evaluation.recall`, relevant Sections among the top k, 75 questions of which 26 are everyday wording with no drug name):

| retriever | all 75, k=1 | k=2 | k=4 | 26 everyday, k=4 |
| --- | ---: | ---: | ---: | ---: |
| BM25, walking skeleton tokenizer | 47 | 55 | 62 | 17 |
| BM25 with Arabic folding and frame words, Arabic-letter names mapped, Franco rewrite (#7) | 61 | 71 | 71 | 23 |
| hybrid BGE-M3 + BM25 (#7) | **66** | **73** | **75** | **26** |

At the API's k=4 the hybrid search finds a relevant Section for all 75 (BM25 alone: 71 after the Arabic work, 62 before) and for all 26 everyday questions (BM25: 23, 17). The gain over the improved BM25 is
modest (4 of 75 at k=4, 5 at k=1) and the retriever weights were chosen on the same grid (differences of 2 to 4 questions out of 225), so this is a mild preference, not a sharp optimum.

### Open items

- LLM comparison (above). Not done.
- BGE-M3 through Ollama instead of in-process CPU torch. **Not done.**
- A second embedding model (for instance a smaller multilingual one) against BGE-M3 on `evaluation/recall.py`. Not done.
