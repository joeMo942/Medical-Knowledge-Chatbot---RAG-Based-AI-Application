# Drug Information Assistant

RAG chatbot answering pharmacy staff and doctors strictly from the approved Knowledge Base (see GLOSSARY.md).

## Run

1. Put the nine KB `.docx` files in `./data` (not committed to git).
2. `docker compose up -d --build` (add `--wait` to block until healthy). Needs Docker with an NVIDIA GPU (ADR 0001).
   The first start downloads the model (about 3.3 GB) into the `ollama-models` volume; watch it with
   `docker compose logs -f model-pull`. Later starts skip the download. The API reports healthy only after the model is present.
3. Open http://localhost:8000 (chat page), or `curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"What is the active ingredient in Cataflam?"}'`.
4. `docker compose down` stops it and keeps the model and Sessions volumes (never add `-v` unless you want them gone).

No NVIDIA GPU: `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up` (far slower; answers will exceed the 30 s limit).

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEDRAG_LLM_MODEL` | `qwen3.5:4b` | Ollama model tag; changing it is the only step to swap the model |
| `MEDRAG_LLM_BACKEND` | `ollama` | `openai` selects any OpenAI-compatible endpoint |
| `MEDRAG_OPENAI_BASE_URL`, `MEDRAG_OPENAI_MODEL`, `MEDRAG_OPENAI_API_KEY` | empty | only used when the backend is `openai`; keep the key in your shell or an untracked `.env` |
| `MEDRAG_BUDGET_S` | `25` | total time budget for one `/chat` request |
| `MEDRAG_TOP_K` | `4` | Sections given to the model |
| `MEDRAG_OLLAMA_NUM_CTX` | `8192` | Ollama context size (warm-up and chat use the same) |
| `MEDRAG_MAX_MESSAGE_CHARS` | `2000` | longer messages get HTTP 400 |
| `MEDRAG_LLM_CONCURRENCY` | `1` | simultaneous LLM calls; requests that cannot finish within the budget fail fast with 500 |
| `MEDRAG_LLM_EST_S` | `5` | floor of the expected duration of one LLM call (the estimate is the slowest of the last 5 calls, never below this), used by that queue check; must be below `MEDRAG_BUDGET_S` |
| `MEDRAG_MAX_OUTPUT_TOKENS` | `1200` | hard maximum of one model reply |
| `MEDRAG_MIN_OUTPUT_TOKENS` | `150` | a request with time for fewer output tokens than this is refused (500 busy) instead of calling the model |
| `MEDRAG_DECODE_TOK_S` | `53` | decode speed used to size each reply's cap to the time left: (time left - margin) x speed; the server may lower it from observed speeds, never raise it |
| `MEDRAG_LLM_MARGIN_S` | `2.5` | time kept back for prompt evaluation and overhead when sizing that cap |
| `MEDRAG_USAGE_HEADER` | `0` | `1` sends the `X-Medrag-Usage` response header (JSON: `prompt_tokens`, `completion_tokens`, `llm_calls` of that request, also on 500s); needed for the token metrics of the evaluation runs. The response body never changes |
| `MEDRAG_SESSIONS_DB` | `:memory:` (`/var/lib/medrag/sessions.db` in Docker, on the `sessions` volume) | SQLite file of the Sessions; with `:memory:` they last only until the process ends |
| `MEDRAG_KB_DIR` | `./data` | KB `.docx` folder (fixed to the `./data` mount, `/kb`, in Docker) |

All of these except `MEDRAG_KB_DIR` and `MEDRAG_SESSIONS_DB` are passed through by `docker-compose.yml`, with the defaults above (the KB folder is the fixed `./data` mount, and the Sessions path is set by the Dockerfile to the `sessions` volume).
A reply cut off at its cap is answered as `out_of_scope` (and logged as a warning), never as an error.
`GET /health` returns 503 (`"status":"error"`) when the LLM backend is unreachable or the model is missing.

## Escalation

Every message is checked before retrieval and before the answer call (`medrag/pipeline.py`): fixed rules for Emergency and Dose wording in English, Arabic, dialect and Franco-Arabic (`medrag/rules.py`; they can only raise a status, never treat "how much X is in product Y" as a Dose, and treat a reported overdose or a child who swallowed pills as an Emergency), then one short LLM routing call (`medrag/routing.py`) that also detects Diagnosis and Treatment Decision and rewrites the question. An emergency rule escalates at once; a dose rule is still routed, because the router may upgrade it to an Emergency, never downgrade it. A flagged message gets a fixed server-written reply (`medrag/escalation.py`: no drug facts, no Citations, Arabic for `ar` and `mixed`).

- The routing call queues in the gate ahead of answer calls and has its own duration estimate and a 4 s timeout. It is skipped only when the request's remaining budget cannot afford it; then the rules decide alone and the answer call also classifies the message (a `trigger` backstop in its schema).
- The user text is fenced as data and its angle brackets are replaced, so it cannot close the fence.
- Last check on every Answered reply (`medrag/dosecheck.py`): a dose-like amount (a number with mg, g, ml, tablets, ... or "three times a day") that is not in the cited Sections' text turns the reply into the fixed Dose escalation.
- `/health` is `ok` only after the model has been loaded, so the first request does not pay for the load.
- A rule-flagged emergency makes no LLM call; every other request makes one short extra call, counted in `X-Medrag-Usage`.

## Out of Scope and Citations

After the model answers, the server checks the Citations (`medrag/answering.py`, `medrag/support.py`). Model text is never shown on `out_of_scope`; that reply is a fixed message in the reply language.

- A Citation must point to a Section that was retrieved for this question (others are dropped); `doc_id`, `title` and `section` come from the Section. A snippet must be one body line or data row of it, word for word; a paraphrase is replaced by the closest real line (most words in common with what the model wrote, never a heading, table header or label line such as "Common side effects:").
- Support check (deterministic, no extra LLM call): the question's topic words (the question minus frame words like "which drugs are used to treat", numbers and KB drug names) must be found in the cited Sections' text, with Arabic folding, stemming and a few English/Arabic pairs. "Which drugs treat malaria?" cites real Plaquenil/Doxycycline rows but the KB never says malaria, so it becomes Out of Scope with `citations: []`. A word the whole KB lacks also fails the check unless the question asks an aspect of a covered drug (ingredient, class, alternatives, side effects, contraindications), which is a partly covered question and stays Answered.
- Partly covered questions are Answered: the prompt asks for the covered part with Citations plus one sentence naming what the knowledge base does not contain. Out of Scope only when nothing asked is covered. The prompt forbids outside knowledge, and the model writes its Citations before it picks the status (a status chosen first made it call a partly covered question out of scope).
- Known limits: a question whose only unsupported topic is a generic word the cited text happens to contain (for example "children" in a question about a drug the KB lists, but not for that use) is left to the model; arabic transliterated brand names are not matched until hybrid search (#6).

## Sessions

`POST /chat` takes an optional `session_id`; the reply carries the Session in the `X-Session-Id` header (the chat page sends it back). No `session_id` starts a new Session, an unknown valid one starts a Session under that ID, and an unusable one (not 1-128 characters of `A-Za-z0-9._:-`) gets a fresh Session whose ID is in the header (no 400). Sessions and their turns are stored in SQLite (WAL) on the `sessions` volume, so they survive `docker compose restart api`; a Session keeps its last 20 turns and the store its 10,000 most recently used Sessions.

The last 3 turns are given to the routing call only, fenced and trimmed, to rewrite a follow-up ("and its side effects?") into a standalone question; retrieval, the answer call and the support check use that question, but only if it passes a deterministic check: every number in it is in the new message, and every other word it adds is a KB drug name or Drug Class (a price or symptom from an earlier reply would be refused and the raw message used instead). Earlier turns are never given to the answer call, so they are never a source of facts: a follow-up whose answer is only in an earlier reply is Out of Scope. Every turn is checked for escalation on its own, on the new message and on its rewrite; history can only raise a status. If the routing call is skipped or fails, the raw message is used. `evaluation/cases_issue8.json` adds multi-turn cases (a fact only in an earlier reply, a Dose request in turn 3, an injection in an earlier turn, an Emergency that only the earlier turn reveals).

## Evaluation

The test set (`evaluation/cases.json`: 85 cases, 90 turns; `evaluation/cases_issue5.json` adds 11 out-of-KB drug traps and partly covered cases, run with `--cases`, each with expected status, expected KB IDs, relevance terms and a note) and its runner live in `evaluation/`, outside the `medrag` package: the system never sees them. Start the stack with the usage header on, so tokens are measured (`MEDRAG_USAGE_HEADER=1 docker compose up -d --wait`), then:

```
python -m evaluation.run --label baseline          # every case, one at a time, against http://localhost:8000
python -m evaluation.review evaluation/results/<date>-baseline.json   # manual faithfulness-review sheet
python -m evaluation.kb_check                      # checks expected KB IDs, sections and out-of-scope traps against the real KB files
```

`run` writes `evaluation/results/<date>-<label>.json` (a row per turn with the full response, latency, the Session ID sent and returned) and `.md` (status accuracy overall and per category, confusion matrix, Citation, snippet and relevance checks, language accuracy, latency p50/p95, tokens, contract violations). The label may hold only letters, digits, `.`, `_` and `-` (for example `qwen3.5-4b`, not `qwen3.5:4b`); it is checked before the first request. A name that already exists gets a numeric suffix. The results folder is gitignored: results stay local. Without the usage header the runner warns and the token metrics are marked absent. The review sheet always includes every reply that came back answered although it should have been escalated, every answered Franco-Arabic, partial and multi-turn reply and up to 5 escalated ones, plus at least 5 seeded random answered replies, and never overwrites an existing sheet; faithfulness is judged by a person, never by an LLM.

## Tests

`pip install -e ".[dev]" && pytest`. No Ollama or KB files are needed; tests that need the real KB skip themselves.
`tests/test_e2e_smoke.py` runs against a started stack and skips otherwise.
