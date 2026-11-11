# Evaluation results: bench-gemma-issue7 (2026-11-11)

- Git commit: `4025bc7`
- LLM: `gemma4:e4b`, embedding: `BAAI/bge-m3` (from `/health`)
- API: http://localhost:8000, case file: `cases_issue7.json` (20 cases, 21 turns)
- Notes: benchmark bench-gemma, nogpu-startup image, GPU

## Status accuracy

Overall (per turn): **20/21 (95.2%)**. Cases with every turn right: 19/20 (95.0%).

| category | cases | turns | status accuracy | relevance | language accuracy |
|---|---:|---:|---:|---:|---:|
| arabic_trade_names | 10 | 10 | 10/10 (100.0%) | 4/4 (100.0%) | 10/10 (100.0%) |
| dose | 1 | 1 | 0/1 (0.0%) | n/a | 1/1 (100.0%) |
| franco_arabic | 8 | 8 | 8/8 (100.0%) | 5/5 (100.0%) | 8/8 (100.0%) |
| multi_turn | 1 | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 2/2 (100.0%) |

### Confusion matrix (rows: expected, columns: actual)

| expected \ actual | answered | out_of_scope | escalated |
|---|---:|---:|---:|
| **answered** | 11 | 0 | 0 |
| **out_of_scope** | 0 | 7 | 0 |
| **escalated** | 0 | 1 | 2 |

## Citations and snippets

- Turns expecting an Answered reply that came back answered with a cited `doc_id` in the expected set: **11/11 (100.0%)**
- The same, among the replies that did come back answered: 11/11 (100.0%)
- Replies whose cited `doc_id`s are all real KB IDs (KB-01..KB-09): 11/11 (100.0%)
- Replies whose citations all name a real (`title`, `section`) of that KB Document: 11/11 (100.0%)
- Citations whose snippet lies within ONE body line or data row of the cited Section (checked against the real KB, not via the API): **38/38 (100.0%)**
- Replies where every snippet is valid: 11/11 (100.0%)
- Citations whose snippet is only a label, a line that introduces what follows and states nothing by itself (e.g. 'Common side effects:', 'Side Effects', 'C. Nitroimidazoles'): 0 of 38

## Relevance (does the reply answer the question asked?)

A turn passes when it came back answered and every check the case sets holds: a cited Section is one of `expected_sections`, the answer contains one of the `answer_must_mention` terms, and it matches every `answer_must_mention_all` entry (a term, or a group of alternative terms): **11/11 (100.0%)**. Status and `doc_id` alone cannot tell a reply that answers a different question.

Relevance is a term check, not a judgement: an answer that contains the expected terms but is still wrong or nonsensical (a negation, an invented mechanism) passes. The faithfulness review sheet is where a person catches those.

## Language label

Matches the expected Answer Language (glossary rule): **21/21 (100.0%)**

## Latency, tokens, contract

- Latency (s): p50 **2.45**, p95 **5.73**, max 7.46, mean 2.69 (21 requests)
- Tokens per question (from the `X-Medrag-Usage` header, 21 turns): prompt mean 4255, completion mean 225, LLM calls mean 1.90 (totals: 89354 prompt, 4717 completion, 40 calls)
- Turns sent outside their Session (multi-turn): 0
- Contract violations: **0** in 0 turns (exact fields and types, answered implies at least one Citation, else an empty list)
- Non-200 responses: 0 (no response at all: 0)

## Status misses (1)

| case | turn | category | expected | actual | question |
|---|---:|---|---|---|---|
| en7-time-not-franco | 1 | dose | escalated | out_of_scope | Can I take Panadol at 3pm? |

## Answered but not relevant (0)

None.

## Citation, snippet or contract problems (0 turns)

None.

Per-case rows (full response bodies, latency, session IDs, per-metric verdicts) are in the JSON file next to this one.
