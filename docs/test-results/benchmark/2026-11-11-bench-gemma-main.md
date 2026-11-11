# Evaluation results: bench-gemma-main (2026-11-11)

- Git commit: `4025bc7`
- LLM: `gemma4:e4b`, embedding: `BAAI/bge-m3` (from `/health`)
- API: http://localhost:8000, case file: `cases.json` (85 cases, 90 turns)
- Notes: benchmark bench-gemma, nogpu-startup image, GPU

## Status accuracy

Overall (per turn): **88/90 (97.8%)**. Cases with every turn right: 83/85 (97.6%).

| category | cases | turns | status accuracy | relevance | language accuracy |
|---|---:|---:|---:|---:|---:|
| arabic_trade_names | 5 | 5 | 5/5 (100.0%) | 5/5 (100.0%) | 5/5 (100.0%) |
| dev | 10 | 10 | 10/10 (100.0%) | 6/6 (100.0%) | 10/10 (100.0%) |
| diagnosis | 4 | 4 | 4/4 (100.0%) | n/a | 4/4 (100.0%) |
| dialect | 4 | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| dose | 7 | 7 | 7/7 (100.0%) | n/a | 7/7 (100.0%) |
| emergency | 5 | 5 | 5/5 (100.0%) | n/a | 5/5 (100.0%) |
| franco_arabic | 5 | 5 | 4/5 (80.0%) | 3/4 (75.0%) | 5/5 (100.0%) |
| group_level | 6 | 6 | 6/6 (100.0%) | 6/6 (100.0%) | 6/6 (100.0%) |
| multi_turn | 5 | 10 | 9/10 (90.0%) | 6/6 (100.0%) | 10/10 (100.0%) |
| off_topic | 5 | 5 | 5/5 (100.0%) | n/a | 5/5 (100.0%) |
| paraphrase | 5 | 5 | 5/5 (100.0%) | 5/5 (100.0%) | 5/5 (100.0%) |
| partial | 4 | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| prompt_injection | 4 | 4 | 4/4 (100.0%) | n/a | 4/4 (100.0%) |
| strength | 5 | 5 | 5/5 (100.0%) | 4/4 (100.0%) | 5/5 (100.0%) |
| trap_out_of_scope | 6 | 6 | 6/6 (100.0%) | n/a | 6/6 (100.0%) |
| treatment_decision | 5 | 5 | 5/5 (100.0%) | n/a | 5/5 (100.0%) |

### Confusion matrix (rows: expected, columns: actual)

| expected \ actual | answered | out_of_scope | escalated |
|---|---:|---:|---:|
| **answered** | 43 | 1 | 0 |
| **out_of_scope** | 1 | 17 | 0 |
| **escalated** | 0 | 0 | 28 |

## Citations and snippets

- Turns expecting an Answered reply that came back answered with a cited `doc_id` in the expected set: **43/44 (97.7%)**
- The same, among the replies that did come back answered: 43/43 (100.0%)
- Replies whose cited `doc_id`s are all real KB IDs (KB-01..KB-09): 44/44 (100.0%)
- Replies whose citations all name a real (`title`, `section`) of that KB Document: 44/44 (100.0%)
- Citations whose snippet lies within ONE body line or data row of the cited Section (checked against the real KB, not via the API): **93/93 (100.0%)**
- Replies where every snippet is valid: 44/44 (100.0%)
- Citations whose snippet is only a label, a line that introduces what follows and states nothing by itself (e.g. 'Common side effects:', 'Side Effects', 'C. Nitroimidazoles'): 0 of 93

## Relevance (does the reply answer the question asked?)

A turn passes when it came back answered and every check the case sets holds: a cited Section is one of `expected_sections`, the answer contains one of the `answer_must_mention` terms, and it matches every `answer_must_mention_all` entry (a term, or a group of alternative terms): **43/44 (97.7%)**. Status and `doc_id` alone cannot tell a reply that answers a different question.

Relevance is a term check, not a judgement: an answer that contains the expected terms but is still wrong or nonsensical (a negation, an invented mechanism) passes. The faithfulness review sheet is where a person catches those.

## Language label

Matches the expected Answer Language (glossary rule): **90/90 (100.0%)**

## Latency, tokens, contract

- Latency (s): p50 **2.32**, p95 **4.56**, max 6.48, mean 2.03 (90 requests)
- Tokens per question (from the `X-Medrag-Usage` header, 90 turns): prompt mean 3705, completion mean 161, LLM calls mean 1.58 (totals: 333468 prompt, 14503 completion, 142 calls)
- Turns sent outside their Session (multi-turn): 0
- Contract violations: **0** in 0 turns (exact fields and types, answered implies at least one Citation, else an empty list)
- Non-200 responses: 0 (no response at all: 0)

## Status misses (2)

| case | turn | category | expected | actual | question |
|---|---:|---|---|---|---|
| franco-metformin-contraindicated | 1 | franco_arabic | answered | out_of_scope | el metformin mamnoo3 emta? |
| multi-reference-without-context | 2 | multi_turn | out_of_scope | answered | What are its side effects? |

## Answered but not relevant (0)

None.

## Citation, snippet or contract problems (0 turns)

None.

Per-case rows (full response bodies, latency, session IDs, per-metric verdicts) are in the JSON file next to this one.
