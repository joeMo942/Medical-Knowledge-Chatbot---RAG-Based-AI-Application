# Operations detail: Sessions, logging and analytics, Hand-off, Admin page

Detail behind the README's feature sections.

## Sessions

`POST /chat` takes an optional `session_id`; the reply carries the Session in the `X-Session-Id` header (the chat page sends it back). No `session_id` starts a new Session, an unknown valid one starts a Session under that ID, and an unusable one (not 1-128 characters of `A-Za-z0-9._:-`) gets a fresh Session whose ID is in the header (no 400). Sessions and their turns are stored in SQLite (WAL) on the `sessions` volume, so they survive `docker compose restart api`; a Session keeps its last 20 turns and the store its 10,000 most recently used Sessions.

The last 3 turns are given to the routing call only, fenced and trimmed, to rewrite a follow-up ("and its side effects?") into a standalone question; retrieval, the answer call and the support check use that question, but only if it passes a deterministic check: every number in it is in the new message, and every other word it adds is a KB drug name or Drug Class (a price or symptom from an earlier reply would be refused and the raw message used instead). Earlier turns are never given to the answer call, so they are never a source of facts: a follow-up whose answer is only in an earlier reply is Out of Scope. Every turn is checked for escalation on its own, on the new message and on its rewrite; history can only raise a status. If the routing call is skipped or fails, the raw message is used. `evaluation/cases_issue8.json` adds multi-turn cases (a fact only in an earlier reply, a Dose request in turn 3, an injection in an earlier turn, an Emergency that only the earlier turn reveals).

## Logging and analytics

Every `/chat` turn and every model call is logged, so the analytics endpoint and the measured numbers for `TOKEN_USAGE.md` come from the running system.

- **Where.** Two tables, `turn_log` and `call_log`, in the same SQLite file as the Sessions (the `sessions` volume). They are separate from the `sessions` / `turns` tables, so the pruning of old Sessions and turns never deletes a log row, and an existing volume gets the tables on its next start.
- **Per turn:** the Session ID, the message, the status, the `language` label, the cited KB IDs (unique, in order), the total latency (wall time of the `/chat` request) and whether it went unanswered. **Unanswered** means the status is `out_of_scope` or `escalated`. A request that failed (500, or 499 when the client left) is logged as a turn with status `error`: it counts in the volume, the status counts and the latency percentiles, but never as unanswered. Requests rejected with 400 are not logged.
- **Per model call:** the kind (`routing`, `answer`, `embedding`), the model name, prompt and completion tokens, and the latency. The LLM tokens are the counts the model server reports (Ollama `prompt_eval_count` / `eval_count`, or the OpenAI-style `usage`), not estimates; a call that failed and reported nothing is logged with 0 tokens. The embedding (BGE-M3 runs in-process, so there is no server to report) is one call per search, its prompt tokens are the BGE-M3 tokenizer's own count of the query (the tokens the model reads), completion 0, and its model is `BAAI/bge-m3@<revision>`. Building the search index at startup is not a turn's call and is not logged. The `X-Medrag-Usage` header still counts the LLM calls only.
- **The write** happens after the response is decided and runs off the request path; any failure (a locked database, a full disk) is written to the server log and swallowed, so it never makes a `/chat` request fail.

`GET /admin/analytics` needs no authentication (the `/admin` endpoints are open; see the note in the README).

```
curl "localhost:8000/admin/analytics?since=2026-11-10&bucket=hour&limit=10"
```

Query parameters, all optional: `since` (inclusive) and `until` (exclusive), ISO 8601 date or date-time (no zone means UTC); `bucket` = `day` (default) or `hour` for the volume over time; `limit` (default 10) for the top lists. The response has `summary` (turns, unanswered, count per status), `volume` (turns, unanswered and errors per bucket), `top_questions` and `unanswered_questions` (grouped by the message after the project's text normaliser and whitespace collapse; each shows the most recent original text, the count and, for the unanswered, the latest status), `tokens_by_call_type` (calls, prompt, completion and total tokens per kind, a `total` row, and an `llm_total` row of routing plus answer), `tokens_per_turn` (sums over all calls and the mean per turn) and `latency_ms` (p50 and p95). Percentiles use the nearest-rank method: the value at rank ceil(p/100 x n) of the n sorted turn latencies in the window.

`python -m evaluation.run --label x --analytics-check` calls the endpoint for the run's window afterwards and compares turns, unanswered, status counts and, with `MEDRAG_USAGE_HEADER=1`, the token totals against the run's own rows; p50 and p95 latency are shown for the client and the server side and are not expected to be equal.

No real patient data is stored: this project has none, and the logged messages are test and demo questions about the Knowledge Base.

## Hand-off

When a question is Escalated, the chat page shows a form below the reply (never on Answered or Out of Scope replies): a summary pre-filled from what the user typed in the Session (editable, at most 2000 characters), a contact field, and a note that only what they type is stored. For an Emergency the reply still starts with the instruction to call 123 or go to the emergency department; the form comes after it and is not focused. The form is in Arabic for `ar` and `mixed` replies, English otherwise. After sending it is replaced by a confirmation; an error is shown in the form.

`POST /handoff` is separate from `/chat` (whose contract is unchanged). The Session comes from the `X-Session-Id` header, as for `/chat`; the body is `{"summary": "...", "contact": "..."}` (summary 1-2000 characters, contact 3-200, both after trimming; no control characters; the body at most 16 KB, else 413). Answer 201 `{"id", "session_id", "created_at"}`.

```
curl -X POST localhost:8000/handoff -H "Content-Type: application/json" -H "X-Session-Id: $SID"   -d '{"summary": "Synthetic test: child swallowed tablets", "contact": "test@example.invalid"}'
```

- **The gate.** A hand-off is accepted only if the Session has an Escalated turn among its stored turns (409 `no escalated question in this session` otherwise, also for an unknown Session; no or unusable header: 400), and at most 5 per Session (then 409). So the endpoint is not a free-text drop box. A failed write is a 500 to the hand-off request (the user must know it was not saved); `/chat` is not affected.
- **What is stored.** Table `handoffs` in the Sessions SQLite file (created on start if missing; Session pruning never deletes it): Session ID, time, the summary and contact exactly as typed, the escalation kind of the Session's latest Escalated turn (`emergency`, `dose`, `diagnosis`, `treatment_decision`; read from the fixed reply text) and its language label. Nothing else.
- **Where it shows.** `GET /admin/analytics` has `handoffs` (`count` in the window and the most recent `limit`, newest first, honouring `since` / `until` / `limit`); `GET /admin/handoffs` returns the same object. Neither needs authentication. Each item has `id`, `session_id`, `created_at`, `kind`, `language`, `summary`, `contact`.

Test data is synthetic: no real patient data is used or needed.

## Admin page

`/admin` is a page for uploading, replacing and deleting KB Documents without touching code; changes are in the next answer. The page and its three endpoints need no authentication; it opens straight into the dashboard.

**KB store.** With `MEDRAG_KB_STORE` set (`/kb-store` on the `kb-store` volume in compose) the KB lives in that writable folder: on the first start it is seeded once from `MEDRAG_KB_DIR` (`./data`) and the bundled `kb_manifest.json` (each file copied as `<KB ID>.docx`, plus a `manifest.json` with the disease headings); after that the store is the source of truth and `./data` is not read again. Files are named from the KB ID and the file's content, never from a client-supplied name, and every write is atomic (temp file + rename, manifest last), so a crash leaves the old KB. Without `MEDRAG_KB_STORE` (tests, plain local runs) the KB is read-only as before and the document endpoints answer 503.

| Endpoint | |
| --- | --- |
| `GET /admin/documents` | KB ID, title, number of Sections, file name as uploaded, updated time |
| `POST /admin/documents` | JSON `{"kb_id": "KB-03", "diseases": ["1. Pain & Fever - ..."], "filename": "x.docx", "content_base64": "..."}`; adds the KB ID or replaces it. 413 above 10 MB |
| `DELETE /admin/documents/{kb_id}` | removes it (404 if unknown) |

Example:
`curl -X POST localhost:8000/admin/documents -H 'content-type: application/json' -d "{\"kb_id\":\"KB-04\",\"content_base64\":\"$(base64 -w0 pharyngitis.docx)\"}"`.

**Disease headings.** Word heading styles cannot tell a disease heading from a drug-class heading, so the upload carries the disease headings (a list, exactly as they appear in the file). They are required for a new KB ID; a replace without them keeps the current ones. A heading that is not in the file is rejected, naming it.

**Validation.** The file goes through the real ingest in a temporary folder before anything is changed. Not a .docx, unreadable, no Sections, a missing heading, a malformed KB ID (`KB-` and two digits) or more than 10 MB gives 400/413 with the reason, and the store and the running KB are untouched.

**What is rebuilt.** The changed KB Document is ingested again and a new immutable KB snapshot (Sections, search index, Trade Name index, support check, Arabic-letter names) replaces the old one in one step; a /chat request in flight finishes on the snapshot it started with. The hybrid index re-embeds only the changed KB Document's Sections and reuses the other vectors, then writes the index cache. One change at a time (a second gets 409); a change while the index is still loading gets 503. Deleting every KB Document is allowed: every question is then Out of Scope. Escalation does not depend on the KB and keeps working throughout.

Changes survive restarts (the `kb-store` volume). To go back to `./data`: `docker compose down`, `docker volume rm <project>_kb-store`, start again (it is seeded afresh). `python -m evaluation.reupload_kb` deletes every KB Document through the API and uploads the originals again (used to check that the eval set still passes).

