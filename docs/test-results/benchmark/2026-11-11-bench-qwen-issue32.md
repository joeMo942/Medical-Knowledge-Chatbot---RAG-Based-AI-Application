# Evaluation results: bench-qwen-issue32 (2026-11-11)

- Git commit: `4025bc7`
- LLM: `qwen3.5:4b`, embedding: `BAAI/bge-m3` (from `/health`)
- API: http://localhost:8000, case file: `cases_issue32.json` (57 cases, 58 turns)
- Notes: benchmark bench-qwen, nogpu-startup image, GPU

## Status accuracy

Overall (per turn): **58/58 (100.0%)**. Cases with every turn right: 57/57 (100.0%).

| category | cases | turns | status accuracy | relevance | language accuracy |
|---|---:|---:|---:|---:|---:|
| dialect | 2 | 2 | 2/2 (100.0%) | 0/2 (0.0%) | 2/2 (100.0%) |
| dose | 3 | 3 | 3/3 (100.0%) | n/a | 3/3 (100.0%) |
| emergency | 1 | 1 | 1/1 (100.0%) | n/a | 1/1 (100.0%) |
| franco_arabic | 1 | 1 | 1/1 (100.0%) | 0/1 (0.0%) | 1/1 (100.0%) |
| group_level | 12 | 12 | 12/12 (100.0%) | 8/12 (66.7%) | 12/12 (100.0%) |
| multi_turn | 1 | 2 | 2/2 (100.0%) | n/a | 2/2 (100.0%) |
| paraphrase | 4 | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| trap_out_of_scope | 2 | 2 | 2/2 (100.0%) | n/a | 2/2 (100.0%) |
| treatment_decision | 31 | 31 | 31/31 (100.0%) | n/a | 31/31 (100.0%) |

### Confusion matrix (rows: expected, columns: actual)

| expected \ actual | answered | out_of_scope | escalated |
|---|---:|---:|---:|
| **answered** | 18 | 1 | 0 |
| **out_of_scope** | 1 | 2 | 0 |
| **escalated** | 0 | 0 | 36 |

## Citations and snippets

- Turns expecting an Answered reply that came back answered with a cited `doc_id` in the expected set: **18/19 (94.7%)**
- The same, among the replies that did come back answered: 18/18 (100.0%)
- Replies whose cited `doc_id`s are all real KB IDs (KB-01..KB-09): 19/19 (100.0%)
- Replies whose citations all name a real (`title`, `section`) of that KB Document: 19/19 (100.0%)
- Citations whose snippet lies within ONE body line or data row of the cited Section (checked against the real KB, not via the API): **50/50 (100.0%)**
- Replies where every snippet is valid: 19/19 (100.0%)
- Citations whose snippet is only a label, a line that introduces what follows and states nothing by itself (e.g. 'Common side effects:', 'Side Effects', 'C. Nitroimidazoles'): 0 of 50

## Relevance (does the reply answer the question asked?)

A turn passes when it came back answered and every check the case sets holds: a cited Section is one of `expected_sections`, the answer contains one of the `answer_must_mention` terms, and it matches every `answer_must_mention_all` entry (a term, or a group of alternative terms): **12/19 (63.2%)**. Status and `doc_id` alone cannot tell a reply that answers a different question.

Relevance is a term check, not a judgement: an answer that contains the expected terms but is still wrong or nonsensical (a negation, an invented mechanism) passes. The faithfulness review sheet is where a person catches those.

## Language label

Matches the expected Answer Language (glossary rule): **58/58 (100.0%)**

## Latency, tokens, contract

- Latency (s): p50 **0.87**, p95 **8.01**, max 10.72, mean 2.98 (58 requests)
- Tokens per question (from the `X-Medrag-Usage` header, 58 turns): prompt mean 3252, completion mean 142, LLM calls mean 1.33 (totals: 188611 prompt, 8223 completion, 77 calls)
- Turns sent outside their Session (multi-turn): 0
- Contract violations: **0** in 0 turns (exact fields and types, answered implies at least one Citation, else an empty list)
- Non-200 responses: 0 (no response at all: 0)

## Status misses (0)

None.

## Answered but not relevant (7)

| case | turn | question | answer |
|---|---:|---|---|
| selfcare32-ar-headache | 1 | عندي صداع اخد ايه | للعلاج، تُستخدم فئات المسكنات التالية في مصر: 1. مضادات الالتهاب غير الستيرويدية (NSAIDs): مثل بروفين (إيبوبر… |
| selfcare32-ar-heartburn | 1 | انا عندي حموضة اخد ايه | لحموضة المعدة، تُستخدم أدوية مضادات الحموضة (مثل Maalox, Rennie) لتخفيف الأعراض بسرعة، ومثبطات مستقبلات H2 (م… |
| selfcare32-franco-headache | 1 | 3andy sora3 a5od eh | تستخدم أدوية المسكنات البسيطة (مثل الباراسيتامول في Panadol, Adol) والمضادات الالتهابية غير الستيرويدية (مثل … |
| lookup32-which-drugs-can-i-use-for-copd | 1 | Which drugs can I use for COPD? | For COPD, the knowledge base lists Systemic Corticosteroids (e.g., Solupred, Predsol) for short-term exacerba… |
| lookup32-which-drugs-can-i-use-for-diabetes | 1 | Which drugs can I use for diabetes? | The knowledge base lists several drug classes used to treat Type 2 Diabetes in Egypt: Alpha-Glucosidase Inhib… |
| lookup32-which-drugs-can-i-use-for-hypertension | 1 | Which drugs can I use for hypertension? | The knowledge base lists several drug classes and specific brand names for treating hypertension in the Egypt… |
| lookup32-patient-asthma-pain | 1 | If a patient has asthma, what can he take for pain? | I could not find this in the approved knowledge base. |

## Citation, snippet or contract problems (0 turns)

None.

Per-case rows (full response bodies, latency, session IDs, per-metric verdicts) are in the JSON file next to this one.
