# Evaluation results: final-issue5 (2026-11-11)

- Git commit: `4f07d12`
- LLM: `qwen3.5:4b`, embedding: `BAAI/bge-m3` (from `/health`)
- API: http://localhost:8000, case file: `cases_issue5.json` (17 cases, 17 turns)

## Status accuracy

Overall (per turn): **15/17 (88.2%)**. Cases with every turn right: 15/17 (88.2%).

| category | cases | turns | status accuracy | relevance | language accuracy |
|---|---:|---:|---:|---:|---:|
| partial | 7 | 7 | 5/7 (71.4%) | 5/7 (71.4%) | 7/7 (100.0%) |
| trap_out_of_scope | 10 | 10 | 10/10 (100.0%) | n/a | 10/10 (100.0%) |

### Confusion matrix (rows: expected, columns: actual)

| expected \ actual | answered | out_of_scope | escalated |
|---|---:|---:|---:|
| **answered** | 5 | 2 | 0 |
| **out_of_scope** | 0 | 10 | 0 |
| **escalated** | 0 | 0 | 0 |

## Citations and snippets

- Turns expecting an Answered reply that came back answered with a cited `doc_id` in the expected set: **5/7 (71.4%)**
- The same, among the replies that did come back answered: 5/5 (100.0%)
- Replies whose cited `doc_id`s are all real KB IDs (KB-01..KB-09): 5/5 (100.0%)
- Replies whose citations all name a real (`title`, `section`) of that KB Document: 5/5 (100.0%)
- Citations whose snippet lies within ONE body line or data row of the cited Section (checked against the real KB, not via the API): **6/6 (100.0%)**
- Replies where every snippet is valid: 5/5 (100.0%)
- Citations whose snippet is only a label, a line that introduces what follows and states nothing by itself (e.g. 'Common side effects:', 'Side Effects', 'C. Nitroimidazoles'): 0 of 6

## Relevance (does the reply answer the question asked?)

A turn passes when it came back answered and every check the case sets holds: a cited Section is one of `expected_sections`, the answer contains one of the `answer_must_mention` terms, and it matches every `answer_must_mention_all` entry (a term, or a group of alternative terms): **5/7 (71.4%)**. Status and `doc_id` alone cannot tell a reply that answers a different question.

Relevance is a term check, not a judgement: an answer that contains the expected terms but is still wrong or nonsensical (a negation, an invented mechanism) passes. The faithfulness review sheet is where a person catches those.

## Language label

Matches the expected Answer Language (glossary rule): **17/17 (100.0%)**

## Latency, tokens, contract

- Latency (s): p50 **3.87**, p95 **5.79**, max 6.44, mean 4.02 (17 requests)
- Tokens per question (from the `X-Medrag-Usage` header, 17 turns): prompt mean 4638, completion mean 180, LLM calls mean 2.00 (totals: 78848 prompt, 3064 completion, 34 calls)
- Turns sent outside their Session (multi-turn): 0
- Contract violations: **0** in 0 turns (exact fields and types, answered implies at least one Citation, else an empty list)
- Non-200 responses: 0 (no response at all: 0)

## Status misses (2)

| case | turn | category | expected | actual | question |
|---|---:|---|---|---|---|
| partial5-metformin-contraindications-cost | 1 | partial | answered | out_of_scope | What are the contraindications of Metformin and how much does it cost? |
| partial5b-hypertension-and-malaria | 1 | partial | answered | out_of_scope | Which drugs treat hypertension and malaria? |

## Answered but not relevant (0)

None.

## Citation, snippet or contract problems (0 turns)

None.

Per-case rows (full response bodies, latency, session IDs, per-metric verdicts) are in the JSON file next to this one.
