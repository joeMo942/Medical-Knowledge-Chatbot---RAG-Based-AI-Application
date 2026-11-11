# Evaluation results: bench-qwen-issue5 (2026-11-11)

- Git commit: `4025bc7`
- LLM: `qwen3.5:4b`, embedding: `BAAI/bge-m3` (from `/health`)
- API: http://localhost:8000, case file: `cases_issue5.json` (17 cases, 17 turns)
- Notes: benchmark bench-qwen, nogpu-startup image, GPU

## Status accuracy

Overall (per turn): **16/17 (94.1%)**. Cases with every turn right: 16/17 (94.1%).

| category | cases | turns | status accuracy | relevance | language accuracy |
|---|---:|---:|---:|---:|---:|
| partial | 7 | 7 | 6/7 (85.7%) | 6/7 (85.7%) | 7/7 (100.0%) |
| trap_out_of_scope | 10 | 10 | 10/10 (100.0%) | n/a | 10/10 (100.0%) |

### Confusion matrix (rows: expected, columns: actual)

| expected \ actual | answered | out_of_scope | escalated |
|---|---:|---:|---:|
| **answered** | 6 | 1 | 0 |
| **out_of_scope** | 0 | 10 | 0 |
| **escalated** | 0 | 0 | 0 |

## Citations and snippets

- Turns expecting an Answered reply that came back answered with a cited `doc_id` in the expected set: **6/7 (85.7%)**
- The same, among the replies that did come back answered: 6/6 (100.0%)
- Replies whose cited `doc_id`s are all real KB IDs (KB-01..KB-09): 6/6 (100.0%)
- Replies whose citations all name a real (`title`, `section`) of that KB Document: 6/6 (100.0%)
- Citations whose snippet lies within ONE body line or data row of the cited Section (checked against the real KB, not via the API): **7/7 (100.0%)**
- Replies where every snippet is valid: 6/6 (100.0%)
- Citations whose snippet is only a label, a line that introduces what follows and states nothing by itself (e.g. 'Common side effects:', 'Side Effects', 'C. Nitroimidazoles'): 0 of 7

## Relevance (does the reply answer the question asked?)

A turn passes when it came back answered and every check the case sets holds: a cited Section is one of `expected_sections`, the answer contains one of the `answer_must_mention` terms, and it matches every `answer_must_mention_all` entry (a term, or a group of alternative terms): **6/7 (85.7%)**. Status and `doc_id` alone cannot tell a reply that answers a different question.

Relevance is a term check, not a judgement: an answer that contains the expected terms but is still wrong or nonsensical (a negation, an invented mechanism) passes. The faithfulness review sheet is where a person catches those.

## Language label

Matches the expected Answer Language (glossary rule): **17/17 (100.0%)**

## Latency, tokens, contract

- Latency (s): p50 **4.75**, p95 **6.00**, max 6.07, mean 4.66 (17 requests)
- Tokens per question (from the `X-Medrag-Usage` header, 17 turns): prompt mean 4638, completion mean 182, LLM calls mean 2.00 (totals: 78848 prompt, 3086 completion, 34 calls)
- Turns sent outside their Session (multi-turn): 0
- Contract violations: **0** in 0 turns (exact fields and types, answered implies at least one Citation, else an empty list)
- Non-200 responses: 0 (no response at all: 0)

## Status misses (1)

| case | turn | category | expected | actual | question |
|---|---:|---|---|---|---|
| partial5b-hypertension-and-malaria | 1 | partial | answered | out_of_scope | Which drugs treat hypertension and malaria? |

## Answered but not relevant (0)

None.

## Citation, snippet or contract problems (0 turns)

None.

Per-case rows (full response bodies, latency, session IDs, per-metric verdicts) are in the JSON file next to this one.
