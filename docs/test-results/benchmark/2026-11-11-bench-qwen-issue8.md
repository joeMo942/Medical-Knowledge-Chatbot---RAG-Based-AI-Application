# Evaluation results: bench-qwen-issue8 (2026-11-11)

- Git commit: `4025bc7`
- LLM: `qwen3.5:4b`, embedding: `BAAI/bge-m3` (from `/health`)
- API: http://localhost:8000, case file: `cases_issue8.json` (4 cases, 9 turns)
- Notes: benchmark bench-qwen, nogpu-startup image, GPU

## Status accuracy

Overall (per turn): **9/9 (100.0%)**. Cases with every turn right: 4/4 (100.0%).

| category | cases | turns | status accuracy | relevance | language accuracy |
|---|---:|---:|---:|---:|---:|
| multi_turn | 4 | 9 | 9/9 (100.0%) | 5/5 (100.0%) | 9/9 (100.0%) |

### Confusion matrix (rows: expected, columns: actual)

| expected \ actual | answered | out_of_scope | escalated |
|---|---:|---:|---:|
| **answered** | 5 | 0 | 0 |
| **out_of_scope** | 0 | 1 | 0 |
| **escalated** | 0 | 0 | 3 |

## Citations and snippets

- Turns expecting an Answered reply that came back answered with a cited `doc_id` in the expected set: **5/5 (100.0%)**
- The same, among the replies that did come back answered: 5/5 (100.0%)
- Replies whose cited `doc_id`s are all real KB IDs (KB-01..KB-09): 5/5 (100.0%)
- Replies whose citations all name a real (`title`, `section`) of that KB Document: 5/5 (100.0%)
- Citations whose snippet lies within ONE body line or data row of the cited Section (checked against the real KB, not via the API): **5/5 (100.0%)**
- Replies where every snippet is valid: 5/5 (100.0%)
- Citations whose snippet is only a label, a line that introduces what follows and states nothing by itself (e.g. 'Common side effects:', 'Side Effects', 'C. Nitroimidazoles'): 0 of 5

## Relevance (does the reply answer the question asked?)

A turn passes when it came back answered and every check the case sets holds: a cited Section is one of `expected_sections`, the answer contains one of the `answer_must_mention` terms, and it matches every `answer_must_mention_all` entry (a term, or a group of alternative terms): **5/5 (100.0%)**. Status and `doc_id` alone cannot tell a reply that answers a different question.

Relevance is a term check, not a judgement: an answer that contains the expected terms but is still wrong or nonsensical (a negation, an invented mechanism) passes. The faithfulness review sheet is where a person catches those.

## Language label

Matches the expected Answer Language (glossary rule): **9/9 (100.0%)**

## Latency, tokens, contract

- Latency (s): p50 **4.04**, p95 **5.35**, max 5.36, mean 3.25 (9 requests)
- Tokens per question (from the `X-Medrag-Usage` header, 9 turns): prompt mean 3425, completion mean 131, LLM calls mean 1.56 (totals: 30823 prompt, 1181 completion, 14 calls)
- Turns sent outside their Session (multi-turn): 0
- Contract violations: **0** in 0 turns (exact fields and types, answered implies at least one Citation, else an empty list)
- Non-200 responses: 0 (no response at all: 0)

## Status misses (0)

None.

## Answered but not relevant (0)

None.

## Citation, snippet or contract problems (0 turns)

None.

Per-case rows (full response bodies, latency, session IDs, per-metric verdicts) are in the JSON file next to this one.
