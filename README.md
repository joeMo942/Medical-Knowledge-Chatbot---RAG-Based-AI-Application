# Drug Information Assistant

A chatbot for pharmacy staff and doctors in Egypt. It answers **only** from an approved Knowledge Base of nine drug-information documents, shows the
document, section and verbatim line each answer came from, and declines or escalates everything else. Questions can be in Arabic (formal or Egyptian dialect), English,
Franco-Arabic (Arabic in Latin letters) or a mix. Everything runs in Docker with open-weights models (`qwen3.5:4b` through Ollama, `BAAI/bge-m3` for search); no API key is needed.

| Read | |
| --- | --- |
| [Run it](#run-it) | prerequisites, the one start command, first-download sizes |
| [API](#api) | `POST /chat`, `GET /health`, examples |
| [Safety policy](#safety-policy) | Answered / Out of Scope / Escalated, Dose vs Strength, why this suits pharmacists and doctors |
| [Architecture](#architecture) | the pipeline, services, volumes |
| [Evaluation](#evaluation) and [Test results](#test-results) | how to run the eval; the recorded results ([docs/test-results/](docs/test-results/README.md)) |
| [`MODEL_DECISION.md`](MODEL_DECISION.md), [`TOKEN_USAGE.md`](TOKEN_USAGE.md) | why these models; tokens and speed per model call |
| [Known limitations](#known-limitations) | what a user will notice |
| [`docs/adr/`](docs/adr), [`GLOSSARY.md`](GLOSSARY.md) | decisions and the domain vocabulary used here (Section, Citation, Dose, Strength, ...) |
| [`docs/submission-checks.md`](docs/submission-checks.md) | secrets scan and clean-run check, what was and was not verified |

## What it does

- Answers a question about a drug, a drug class, trade names, ingredients, side effects, contraindications and alternatives from the Knowledge Base, with **Citations** (KB ID, title, section, snippet).
- Says so when the Knowledge Base does not cover it (**Out of Scope**), and answers the covered half of a partly covered question.
- **Escalates** emergencies and requests for a dose, a diagnosis or a treatment decision about a specific person: a fixed reply says what to do next and gives no medical advice. A plain first-person **self-care question** ("I have a headache, what can I take?") with no risk qualifier is answered with what the Knowledge Base lists for the condition, plus a note to ask a pharmacist (see the Safety policy).
- **Tolerates typos:** a word one edit away from exactly one Knowledge Base word is corrected, and the answer starts with "Did you mean: headache?".
- Keeps conversations (Sessions) so a follow-up like "and its side effects?" works; remembers nothing as a source of facts.
- Extras: a chat page (`/`), an admin page to upload, replace and delete KB documents (`/admin`), conversation logging and analytics, a hand-off form after an escalation, and its own evaluation harness.

## Run it

### Prerequisites

- **Docker** with Compose v2 (tested with Docker 29.4 and Compose 5.1 on Windows 11, Docker Desktop).
- **An NVIDIA GPU with 8 GB of VRAM** and the NVIDIA Container Toolkit (on Windows, Docker Desktop with GPU support), because the default compose file requests the GPU for the LLM (ADR 0001).
  The author's test machine: RTX 5070 Laptop GPU, 8 GB. **No GPU:** use the CPU override `docker-compose.cpu.yml` (below); it works but is far slower and answers will exceed the 30 s limit.
- About **16 GB of free disk** for the images and models, and about 2.5 GB of RAM for the search model.
- **The Knowledge Base files.** They are not in git. Put the nine `.docx` files in `./data` with exactly these file names (they are listed in `medrag/kb_manifest.json`):

  | KB ID | file in `./data` |
  | --- | --- |
  | KB-01 | `asthma w copd.docx` |
  | KB-02 | `gerd& vomiting.docx` |
  | KB-03 | `pain headache ibs arthritis.docx` |
  | KB-04 | `pharyngitis.docx` |
  | KB-05 | `أدوية الالتهاب الرئوي.docx` |
  | KB-06 | `ادويه الحساسيه و الجيوب الانفيه.docx` |
  | KB-07 | `ادويه السكر.docx` |
  | KB-08 | `ادويه الضغط و الكوليسترول.docx` |
  | KB-09 | `ادويه القولون و التهابات المفاصل.docx` |

  A missing file stops the API at start with the list of the names it did not find.
- **Python 3.12+ on the host**, only to run the evaluation (not needed to run the system).

### Start

```
git clone <this repository> && cd <this repository>
# put the nine .docx files in ./data  (see above)
echo "MEDRAG_ADMIN_TOKEN=$(openssl rand -hex 16)" > .env    # optional: enables the admin dashboard; .env is gitignored and read by docker compose
docker compose up -d --build --wait
```

That is the whole start: one command builds the `api` image, starts Ollama, downloads the models on the first run, and (with `--wait`) returns when `GET /health` is `ok`.

### Open it

When `docker compose up` returns, everything is on <http://localhost:8000>:

| What | URL | Notes |
| --- | --- | --- |
| **Chat page** | <http://localhost:8000> | Ask in Arabic, Egyptian dialect, Franco-Arabic or English. Escalated replies show the hand-off form. |
| **Admin dashboard** | <http://localhost:8000/admin> | Sign in with the admin token (below). Tabs: Documents (upload, replace, delete KB Documents), Analytics, Hand-offs. |
| Health | <http://localhost:8000/health> | `{"status":"ok", ...}` when the models and the index are ready (503 before). |
| Interactive API docs | <http://localhost:8000/docs> | FastAPI's Swagger page for every endpoint. |
| Chat API | `POST http://localhost:8000/chat` | See [API](#api). |

**The admin token** is the value of `MEDRAG_ADMIN_TOKEN`. To see it: `cat .env` (or `echo $MEDRAG_ADMIN_TOKEN` if you exported it in your shell instead). The dashboard keeps it only for the browser tab (sessionStorage); "Sign out" forgets it.
Without a token the admin dashboard says "admin disabled: set MEDRAG_ADMIN_TOKEN": add the line to `.env` and run `docker compose up -d` again (no rebuild needed). The chat page never needs a token.

From a terminal:

```
curl localhost:8000/health
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"What is the active ingredient in Cataflam?"}'
curl -H "Authorization: Bearer $(sed -n 's/^MEDRAG_ADMIN_TOKEN=//p' .env)" localhost:8000/admin/documents
```

Port 8000 busy? The API listens on host port 8000 (`docker-compose.yml`, `api` → `ports`); change the left side of `"8000:8000"` and use that port in the URLs.

### Stop, restart, logs

- `docker compose down` stops it and keeps the models, the KB store, the Sessions, the analytics and the hand-offs. Never add `-v` unless you want them gone.
- `docker compose up -d --wait` starts it again; a restart is healthy in about 10 s (the models and the search index are cached).
- Logs: `docker compose logs -f api` (and `ollama`, `model-pull`, `embedder-pull`). Status: `docker compose ps`.

**No NVIDIA GPU:** `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build --wait`.

### What the first start downloads

| what | size | where | how long |
| --- | ---: | --- | --- |
| `qwen3.5:4b` (4.2 B parameters, Q4_K_M) | 3.3 GB | Ollama, into the `ollama-models` volume (`docker compose logs -f model-pull`) | download time of 3.3 GB |
| BGE-M3, pinned revision, each file checked by size and SHA-256 | 2.3 GB | Hugging Face, into the `embed-models` volume (`docker compose logs -f embedder-pull`) | download time of 2.3 GB; the copy and check alone took 26 s from a local folder |
| `api` image (CPU `torch`, `transformers`, ...) | 1.65 GB on disk (`docker images`) | built from the `Dockerfile` | build plus pip downloads |
| `ollama/ollama` image | 3.39 GB compressed download (from the registry manifest), 8.5 GB on disk (`docker images`) | Docker Hub | download time of 3.39 GB |
| search index of the 113 Sections | small | built by the API on the first start | **42.7 s** measured (embedder load 7.7 s), cached afterwards |

How the times were measured: **the downloads themselves were not timed on a clean machine** (the author's connection is slow and flaky; see `docs/submission-checks.md`), so the time of the three big downloads is just
size divided by your bandwidth (3.3 GB + 2.3 GB at 100 Mbit/s, about 12 MB/s, is roughly 8 minutes; the images come on top). What was measured, on the test machine, from a fresh clone with fresh volumes, the images already built, and both models
seeded from local copies (so nothing large was downloaded): **121 s** from `docker compose up -d --build --wait` to `healthy`, of which the model copy and check was 26 s and the one-time index build 43 s.
A later restart is healthy in about 8 s, because the index is cached in the `index` volume and the models are already there. The API reports healthy only when both models are present, BGE-M3 is loaded and the index is ready; until then `/health` is 503.

Slow or offline network (optional; both default to an empty folder, which means "download"):
`MEDRAG_WHEELS_DIR=<folder of wheels>` makes the image build install torch (CPU), transformers and their dependencies from that folder instead of downloading them (the pins of `constraints.txt` found in the folder are installed first, with no network access), and
`MEDRAG_EMBED_SRC=<folder with the BGE-M3 files>` makes `embedder-pull` copy (and verify) them into the volume instead of downloading. Example:
`MEDRAG_WHEELS_DIR=~/.cache/medrag/wheelhouse/linux-py312 MEDRAG_EMBED_SRC=~/.cache/medrag/bge-m3 docker compose up -d --build --wait`.

### Windows and Git Bash notes

- Docker Desktop with the WSL 2 backend and GPU support. In Git Bash, put `MSYS_NO_PATHCONV=1` in front of any `docker ... exec/run` command that passes a Linux path (`/tmp/x.py`), or Git Bash rewrites it into a Windows path.
- **Arabic or other non-ASCII text in a `curl` command line is mangled on Windows** (it arrives as `?`): the server then sees question marks and answers something else. Put the JSON in a UTF-8 file and send it with
  `curl -X POST localhost:8000/chat -H 'content-type: application/json' --data-binary @question.json`. The chat page and the Python eval runner are not affected.
- The `curl` and `export` examples are for Git Bash / Linux / macOS. In PowerShell use `curl.exe` (not the `curl` alias), `$env:MEDRAG_ADMIN_TOKEN = '...'`, and a file for the JSON body.

## API

Exactly the brief's contract; an extra field in the `/chat` body would break a strict schema check, so the Session ID is in a header (ADR 0002).

### `POST /chat`

Request body (JSON): `message` (required, plain text, at most 2000 characters) and `session_id` (optional; a missing value starts a new Session).

Response, HTTP 200:

```json
{
  "answer": "The active ingredient in Cataflam is Diclofenac potassium.",
  "citations": [
    {"doc_id": "KB-04", "title": "أدوية التهاب الحلق واللوز – Pharyngitis & Tonsillitis",
     "section": "NSAIDs – Non-Steroidal Anti-Inflammatory Drugs", "snippet": "Cataflam – Diclofenac potassium"}
  ],
  "status": "answered",
  "language": "en"
}
```

| field | rule |
| --- | --- |
| `answer` | text in the answer language |
| `citations` | at least one entry when `status` is `answered`; an empty list otherwise. `doc_id` is the KB ID (`KB-01` ... `KB-09`), `snippet` is one line or table row of the Section, word for word |
| `status` | `answered`, `out_of_scope` or `escalated` (see [Safety policy](#safety-policy)) |
| `language` | `ar`, `en` or `mixed` (`mixed` means Franco-Arabic): the language of the answer |

- **Response header `X-Session-Id`**: the Session of this conversation. Send it back as `session_id` in the next request to continue the thread (the chat page does).
  An unusable `session_id` (not 1 to 128 characters of `A-Za-z0-9._:-`) gets a fresh Session, not an error.
- **Errors:** HTTP 400 `{"error": "..."}` for a bad request (no message, too long, not JSON); HTTP 500 `{"error": "..."}` on any failure, including the server being busy, the search index still loading, or a request that cannot finish within the time budget (25 s).
- **Timing:** every request is answered within 30 s (the server stops at 25 s); on the test GPU the p95 is about 6 s.

Examples (real replies from the test run; Arabic example in a file, see the Windows notes):

```
$ curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"How many mg of Voltaren can an adult take per day?"}'
{"answer":"I cannot give doses. Please check the approved leaflet of the product, or ask the prescriber or a pharmacist.","citations":[],"status":"escalated","language":"en"}

$ curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"Write me a cover letter for a job application."}'
{"answer":"I could not find this in the approved knowledge base.","citations":[],"status":"out_of_scope","language":"en"}

$ curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"He took 30 pills of panadol and is vomiting and confused"}'
{"answer":"This may be an emergency. Call 123 (ambulance in Egypt) or go to the nearest emergency department now. Do not wait for medication advice from this assistant.","citations":[],"status":"escalated","language":"en"}

# a conversation: read the Session from the header, send it back
$ SID=$(curl -si -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"What is the active ingredient in Controloc?"}' | tr -d '\r' | grep -i '^x-session-id' | cut -d' ' -f2)
$ curl -s -X POST localhost:8000/chat -H 'content-type: application/json' -d "{\"session_id\":\"$SID\",\"message\":\"Which drug class is it in?\"}"
{"answer":"Pantoprazole belongs to the Proton Pump Inhibitors (PPIs) drug class.","citations":[{"doc_id":"KB-02","title":"أدوية قرحة المعدة والحموضة – Peptic Ulcer & GERD","section":"Proton Pump Inhibitors (PPIs)","snippet":"الاستخدام: تقليل إفراز حمض المعدة، وعلاج GERD وPeptic Ulcers، وتدخل ضمن أنظمة علاج H. pylori."}],"status":"answered","language":"en"}
```

### `GET /health`

```
$ curl -s localhost:8000/health
{"status":"ok","llm":"qwen3.5:4b","embedding":"BAAI/bge-m3","kb_documents":9,"embedding_revision":"5617a9f61b028005a4858fdac845db406aefb181"}
```

HTTP 200 when ready. 503 `{"status":"error", ...}` when the LLM backend is unreachable or the model is missing, or the embedding model and index are not loaded yet.

### Other endpoints (not part of the scored contract)

| endpoint | what | detail |
| --- | --- | --- |
| `GET /`, `GET /admin` | chat page, admin page | [docs/operations.md](docs/operations.md) |
| `GET /admin/analytics`, `GET /admin/handoffs` | analytics and hand-offs, `Authorization: Bearer $MEDRAG_ADMIN_TOKEN` | [Logging and analytics](#logging-and-analytics), [Hand-off](#hand-off) |
| `GET/POST /admin/documents`, `DELETE /admin/documents/{kb_id}` | list, upload or replace, delete a KB Document | [Admin page](#admin-page) |
| `POST /handoff` | leave a summary and contact after an escalation | [Hand-off](#hand-off) |

The `/admin` endpoints answer 503 until `MEDRAG_ADMIN_TOKEN` is set, and 401 for a wrong token.

## Safety policy

The users are pharmacists and doctors, and the system is a lookup in approved text, not a clinician. The terms are defined in [`GLOSSARY.md`](GLOSSARY.md).

| status | when | what the reply is |
| --- | --- | --- |
| **`answered`** | the Knowledge Base supports what was asked, or part of it | the answer, with at least one Citation; a partly covered question gets the covered part plus one sentence naming what the Knowledge Base does not contain |
| **`out_of_scope`** | nothing that was asked is covered, or the question is not about drug information (a cover letter, a recipe, a drug the Knowledge Base does not list) | a fixed server-written message in the reply language, no Citations; the model's own text is never shown |
| **`escalated`** | an **Emergency** (red-flag symptoms, an acutely unwell person, a reported overdose or a child who swallowed pills), or a request for a **Dose**, a **Diagnosis** or a **Treatment Decision** about a specific person | a fixed server-written reply that says what to do next and gives no medical advice (an Emergency: call 123 or go to the emergency department; a Dose: check the approved leaflet or ask the prescriber or a pharmacist; a Diagnosis or Treatment Decision: ask the treating doctor); no drug facts, no Citations. Takes precedence over the other two |

**Dose vs Strength.** A **Strength** is the amount of an Active Ingredient in one unit of a product, as the Knowledge Base writes it ("Paracetamol 500 mg in a Panadol tablet"): that is a fact about the product and may be answered.
A **Dose** is how much to give, how often, or to whom ("how many tablets a day", "units of Lantus at night", "a child's dose"): the assistant **never states one**, even when the question is wrapped in "ignore your rules", even
for a drug the Knowledge Base does not list. "How much caffeine is in Panadol Extra?" is a Strength question; "how many mg of Voltaren can an adult take per day?" is a Dose request and is escalated. As a last check on every Answered reply, a dose-like amount
("500 mg", "three times a day", "2 tablets") that is not in the cited Sections' own text turns the reply into the fixed Dose escalation, so a model that invents a number cannot get it to the user.
A question about what the Knowledge Base says for a group of patients ("are NSAIDs allowed in pregnancy?", "is Doxycycline suitable for children under 8?") is answered; deciding for one named patient is a Treatment Decision and is escalated.

**Self-care questions and lookups (a deliberate exception, ADR 0003).** Asking what is used FOR a condition is a normal drug-information **lookup** even in the first person: "Which drugs can I use for asthma?", "What can I take for migraine?", "What can a pregnant woman take for her heartburn?" are **answered** (or Out of Scope when the Knowledge Base lacks the condition), never escalated; the condition asked about is not a risk qualifier. A **self-care question** states the user's OWN symptom and asks what to take ("I have a headache, what can I take?", "عندي صداع اخد ايه", "3andy sora3 a5od eh"): with no risk qualifier it is **answered** with general Knowledge Base information (the drug classes and drugs it lists for that condition, with Citations, never a Dose) and a fixed last sentence, added by the server, that a pharmacist can say which one suits the person. It is **escalated** when the message states a risk qualifier about the person: pregnancy or breastfeeding, a child or an age under 18, an elderly person, kidney or liver disease or another chronic condition they HAVE ("I have asthma and a headache"; not the condition being asked about), other medicines they take, an allergy, a severity or red-flag word ("worst", "severe", a stiff neck, a weak arm, a high fever, alcohol, "جامد"), how long it has lasted ("for 3 days", "من ايام"), a Dose ("and how many?": the Dose text), or an Emergency; a qualifier in an earlier turn or in the router's rewrite escalates too, misspelled qualifier words ("pregnent", "sever") count, and a question about another specific person ("my wife", "my son", "my patient", a pronoun with no generic noun before it) stays a Treatment Decision ("a patient ... he" is a group). Migraine, arthritis, IBS and heartburn are everyday indications, not chronic qualifiers. The rule is fixed code (`medrag/selfcare.py`); the only status it ever lowers is the router's `treatment_decision`, and it never lowers an emergency, a dose or a diagnosis.

**Typos.** A Latin-letter question word of five letters or more that is not in the Knowledge Base but is ONE edit away from exactly one drug name (Trade Name or Active Ingredient) or condition word of it (the words of its Section titles and headings: "headace" -> "headache", "Brufin" -> "Brufen") is corrected before search and the support check, and an Answered reply then starts with `Did you mean: headache?` (Arabic: `هل تقصد: headache؟`). An ambiguous word is never corrected, a drug the Knowledge Base knows is never turned into another, ordinary English words are left alone, and Arabic-script and Franco-Arabic words are not corrected. Escalated and Out of Scope replies keep their fixed texts.

**Why this policy suits pharmacists and doctors.**

- Their questions are lookups ("what is in this product, what are the contraindications, which brands share this ingredient"); a wrong confident answer costs more than a refusal, so **the answer must come from the approved text and show where**: each Citation is a real Section that was retrieved for this question and a snippet that is a line of it, word for word, so the professional can check the source in seconds.
- Doses, diagnoses and treatment choices for a person are the professional's own judgement and need information the Knowledge Base does not have (weight, history, other drugs); a 4B model must not guess them, and the Knowledge Base itself holds no doses. The assistant therefore routes them to the treating doctor instead.
- **Failing safe:** an error, a timeout, a cut-off reply, a Citation that does not check out, or a topic the cited text does not support all end as Out of Scope, never as an invented answer; Escalated is decided by fixed rules first and the model can only raise a status, never lower one; Emergency and Dose wording is caught by fixed rules that need neither search nor the model, so those are escalated even while the search index is still loading.
- An escalation can be followed up by a human: the chat page offers a **hand-off** (summary and contact) that a pharmacist or physician can read through the admin endpoints.
- No real patient data is used anywhere; the test data is synthetic.

Detail and the checks behind it: [docs/safety-pipeline.md](docs/safety-pipeline.md).

## Architecture

```
 browser / script
      |  POST /chat {message, session_id?}
      v
 +---------------------------------- api  (FastAPI, one container) --------------------------------------+
 | 1 rules          fixed Emergency / Dose wording rules (EN, AR, dialect, Franco)  --> Escalated, no LLM   |
 | 2 routing call   LLM, one short JSON call: trigger, language, standalone question  --> may raise to Escalated |
 | 3 escalation     fixed server-written reply (no drug facts, no Citations)                                |
 | 4 hybrid search  BGE-M3 dense + sparse + BM25, fused; top 4 Sections          (CPU, in the api process)  |
 | 5 answer call    LLM, JSON: answer, citations (Section ID + snippet), status                             |
 | 6 citations      keep only retrieved Sections; snippet must be a real line of the Section                |
 | 7 dose check     an amount not in the cited text  --> Escalated (Dose)                                    |
 | 8 support check  the question's topic words must be in the cited Sections --> else Out of Scope           |
 +------------------------------------------------------------------------------------------------------+
      |  {answer, citations, status, language}  + X-Session-Id header
      v
 browser / script

 api  --HTTP-->  ollama  (qwen3.5:4b, on the GPU)         volumes: ollama-models
 api  reads      embed-models (BGE-M3 files, read-only), kb-store (KB Documents), index (search index)
 api  writes     sessions (SQLite: Sessions, turn log, call log, hand-offs)
```

Services in `docker-compose.yml`, in start order: **`ollama`** (healthy) then **`model-pull`** (downloads the LLM once, exits) and **`embedder-pull`** (puts the BGE-M3 files in the `embed-models` volume once, exits), then **`api`**
(healthy only when both models are present and the search index is ready). `./data` is mounted read-only at `/kb` and copied into the `kb-store` volume on the first start; after that the store is the source of truth
(the admin page changes it). The embedding model runs on the CPU so the GPU stays free for the LLM (ADR 0001). The LLM has a concurrency of one: a request that cannot be served within its time budget fails fast (HTTP 500 "busy") rather than queueing.
Why each choice: [MODEL_DECISION.md](MODEL_DECISION.md), [docs/adr/](docs/adr).

## Features

### Search

Hybrid search over whole Sections: BGE-M3 gives a dense and a sparse vector per Section and per question, BM25 (with Arabic folding) is a third scorer, and the three rankings are fused by weighted reciprocal rank. A question that names a
Trade Name is also searched with its Active Ingredient and the other way round; an Alternatives question ("bedail Brufen", "alternatives to Panadol") gets the Sections that list every other Trade Name of the same Active Ingredient.
Offline recall (`python -m evaluation.recall`; relevant Section among the top 4, 75 questions): BM25 62, improved BM25 71, **hybrid 75**; on the 26 everyday questions with no drug name, 17, 23 and **26**. The index (113 Sections) builds in about 43 s on the first start and is cached.
Detail, the full recall table and the known limits: [docs/search-and-arabic.md](docs/search-and-arabic.md).

### Arabic, dialect and Franco-Arabic

One normaliser is used everywhere (alef, yaa and taa marbuta variants, harakat, tatweel, Arabic-Indic digits). The answer language is decided once from the message: Arabic script is `ar`, English is `en`, Franco-Arabic is `mixed`
(Arabic script with the English drug names and clinical terms the KB uses). Franco-Arabic questions are rewritten into Arabic letters before search, but the server deletes any word of the rewrite that is not the user's own word, so the model cannot
swap a word for another one; Trade Names written in Arabic letters ("بروفين") are mapped to the KB's Latin spelling. Detail and limits: [docs/search-and-arabic.md](docs/search-and-arabic.md).

### Sessions

`POST /chat` takes an optional `session_id`; the reply returns it in `X-Session-Id`. Sessions live in SQLite on the `sessions` volume and survive restarts (last 20 turns per Session, 10,000 Sessions). The last 3 turns are given to the routing call only, to rewrite a follow-up
("and its side effects?") into a standalone question that must pass a deterministic check; earlier turns are never a source of facts and every turn is checked for escalation on its own. Detail: [docs/operations.md](docs/operations.md).

### Logging and analytics

Every `/chat` turn and every model call is logged (status, language, cited KB IDs, latency, prompt and completion tokens per call kind). `GET /admin/analytics` (Bearer `MEDRAG_ADMIN_TOKEN`) returns volume over time, top and unanswered questions,
tokens per call type and per turn, and latency p50 and p95; `python -m evaluation.run ... --analytics-check` cross-checks it against a run. The log is written off the request path and a write failure never fails `/chat`. No real patient data is stored.

```
export MEDRAG_ADMIN_TOKEN=$(openssl rand -hex 16)   # before docker compose up
curl -H "Authorization: Bearer $MEDRAG_ADMIN_TOKEN" "localhost:8000/admin/analytics?since=2026-11-10&bucket=hour&limit=10"
```

Parameters and fields: [docs/operations.md](docs/operations.md). The measured token numbers are in [TOKEN_USAGE.md](TOKEN_USAGE.md).

### Hand-off

After an Escalated reply the chat page shows a form (a summary pre-filled from what the user typed, a contact field, a note that only what they type is stored); it never delays the escalation advice. `POST /handoff` stores it with the Session
(accepted only if the Session has an Escalated turn, at most 5 per Session) and `GET /admin/handoffs` lists them. Detail: [docs/operations.md](docs/operations.md).

```
curl -X POST localhost:8000/handoff -H "Content-Type: application/json" -H "X-Session-Id: $SID" -d '{"summary": "Synthetic test: child swallowed tablets", "contact": "test@example.invalid"}'
```

### Admin page

`/admin` uploads, replaces and deletes KB Documents without touching code; the change is in the next answer. The page asks for `MEDRAG_ADMIN_TOKEN`. The KB lives in the `kb-store` volume (seeded once from `./data`); every upload is validated by the real ingest
before anything changes and applied atomically. To go back to `./data`: `docker compose down`, `docker volume rm <project>_kb-store`, start again. Detail and the three endpoints: [docs/operations.md](docs/operations.md).

## Configuration (environment variables)

No secrets are stored in the repository or the images: the only secret-like values, `MEDRAG_ADMIN_TOKEN` and the optional `MEDRAG_OPENAI_API_KEY`, come from the operator's shell at start (or an untracked `.env`, which is gitignored). `docker-compose.yml` passes them through with no default.

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
| `MEDRAG_ADMIN_TOKEN` | unset | Bearer token of the `/admin` endpoints (see Logging and analytics); unset or empty makes every `/admin` endpoint answer 503. Set it in your shell, never in a file in the repo |
| `MEDRAG_RETRIEVER` | `hybrid` | `bm25` searches without the embedding model (a quick start, or a baseline); `hybrid` is BGE-M3 + BM25, see Search |
| `MEDRAG_MIN_DENSE` | `0.5` | with no query word in any Section (BM25 finds nothing), a best dense similarity below this means nothing matches (Out of Scope without an LLM call). On the test set the 5 such questions, all not answerable, reach 0.36-0.49 |
| `MEDRAG_EMBED_THREADS` | `4` | CPU threads for embedding one question (measured: 4 threads take about 50 ms, 24 take about 200 ms); the index build always uses all cores; `0` uses all cores for questions too |
| `MEDRAG_EMBED_PATH`, `MEDRAG_INDEX_DIR`, `MEDRAG_EMBED_SEED` | `/models/bge-m3`, `/index`, empty (set by the Dockerfile and compose) | where the model files and the search index cache live, and an optional folder to copy the model files from |
| `MEDRAG_KB_DIR` | `./data` | KB `.docx` folder (fixed to the `./data` mount, `/kb`, in Docker); with a KB store it is only the one-time seed |
| `MEDRAG_KB_STORE` | unset (`/kb-store` in Docker, on the `kb-store` volume) | writable KB store of the admin page, see Admin page; unset makes the KB read-only and the document endpoints 503 |

All of these except `MEDRAG_KB_DIR` and `MEDRAG_SESSIONS_DB` are passed through by `docker-compose.yml`, with the defaults above (`MEDRAG_ADMIN_TOKEN` from your shell, with no default) (the KB folder is the fixed `./data` mount, and the Sessions path is set by the Dockerfile to the `sessions` volume).
A reply cut off at its cap is answered as `out_of_scope` (and logged as a warning), never as an error.
`GET /health` returns 503 (`"status":"error"`) when the LLM backend is unreachable or the model is missing, or the embedding model and index are not loaded yet; when ready it reports `"embedding":"BAAI/bge-m3"` and `embedding_revision`. A `/chat` that needs search answers 500 `{"error": ...}` until then; Emergency and Dose messages are still escalated (that needs no search).

## Evaluation

The test set and its runner live in `evaluation/`, outside the `medrag` package: the system never sees them.

- `evaluation/cases.json`: 85 cases, 90 turns (the 10 public dev questions, paraphrases, dialect, Franco-Arabic, Arabic-letter Trade Names, Strength, group-level questions, partly covered questions, multi-turn, prompt injection, off-topic traps, Emergency, Dose, Diagnosis, Treatment Decision), each with the expected status, KB IDs, relevance terms and a note.
- `--cases` runs a smaller set: `cases_issue5.json` (17: drugs outside the KB and partly covered cases), `cases_issue7.json` (20: Franco-Arabic and Arabic-letter Trade Name cases), `cases_issue8.json` (4 multi-turn cases), `cases_issue32.json` (57: self-care wording and lookups).

Run it against a started stack. It needs the usage header on (so tokens are measured) and, for `--analytics-check`, the same admin token the server was started with. The runner needs Python 3.12+ and the package's small dependencies on the host, and it reads the KB files in `./data` (for the snippet check):

```
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
export MEDRAG_ADMIN_TOKEN=$(openssl rand -hex 16) MEDRAG_USAGE_HEADER=1
docker compose up -d --build --wait                  # (re)start with both set
python -m evaluation.run --label final --analytics-check                               # every case, one at a time, against http://localhost:8000
python -m evaluation.run --label final-held-out --cases evaluation/cases_issue5.json   # a held-out set (also cases_issue7.json, cases_issue8.json, cases_issue32.json)
python -m evaluation.review evaluation/results/<date>-final.json                       # the manual faithfulness-review sheet
python -m evaluation.kb_check                                                          # checks expected KB IDs, sections and out-of-scope traps against the real KB files
python -m evaluation.recall                                                            # offline search recall (needs torch and the BGE-M3 files, see the script)
```

`run` writes `evaluation/results/<date>-<label>.json` (a row per turn: full response, latency, Session IDs) and `.md` (status accuracy overall and per category, confusion matrix, Citation, snippet and relevance checks, language accuracy, latency p50 and p95, tokens, contract violations).
The label may hold only letters, digits, `.`, `_` and `-` (`qwen3.5-4b`, not `qwen3.5:4b`); it is checked before the first request, and a name that already exists gets a numeric suffix. Without the usage header the runner warns and marks the token metrics absent. The results folder is gitignored; the final runs were copied to [docs/test-results/](docs/test-results/README.md).
The review sheet always includes every reply that came back answered although it should have been escalated, every answered Franco-Arabic, partial and multi-turn reply, up to 5 escalated ones and at least 5 seeded random answered replies; the sheet is filled in a first pass by an AI reviewer (Claude Opus) that checks each claim against the cited Section; the author then checks it and signs it off, and the sheet says which is which. The sheet never overwrites an existing one.
`--analytics-check` compares `GET /admin/analytics` with the run's own numbers (turns, unanswered, status counts and token totals must match exactly; latency is shown from both sides). Send no other traffic to the API while it runs: any other request in the window shows up as a DIFF.

## Test results

Recorded final run, 2026-11-11 (laptop plugged in, Windows Balanced power mode, the GPU sometimes power-capped), one request at a time. Full reports, raw results, the analytics-check outputs, the recall output and the harness description (test sets, scoring, how to re-run, every failing case, caveats) are in [docs/test-results/](docs/test-results/README.md).

| set | status | language | Citation KB ID | snippets verbatim | contract violations | latency p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cases.json` (85 cases, 90 turns, includes the 10 dev questions) | **90/90** | 90/90 | 44/44 | 53/53 | 0 | 3.47 / 6.65 s |
| `cases_issue5.json` (17: outside the KB, partly covered) | 15/17 | 17/17 | 5/7 | 6/6 | 0 | 3.87 / 5.79 s |
| `cases_issue7.json` (20 cases, 21 turns: Franco-Arabic, Arabic-letter Trade Names) | 21/21 | 21/21 | 11/11 | 16/16 | 0 | 3.64 / 6.17 s |
| `cases_issue8.json` (4 cases, 9 turns: multi-turn) | 9/9 | 9/9 | 5/5 | 5/5 | 0 | 3.20 / 4.65 s |
| `cases_issue32.json` (57 cases, 58 turns: self-care wording, lookups) | 58/58 | 58/58 | 18/19 | 50/50 | 0 | 0.82 / 7.43 s |

Search recall (75 questions, relevant Section among the 4 retrieved): hybrid 75, BM25 71.
The two misses are partly covered questions that came back Out of Scope instead of answering the covered half (`partial5-metformin-contraindications-cost`, `partial5b-hypertension-and-malaria`); they fail every time. The sets were written by the author and used while tuning, so these are development numbers, not a forecast for the hidden set.
The faithfulness review sheets are generated but not signed off yet; see the results folder.

## Known limitations

What a user is most likely to notice (the full list is in the follow-up issue #16 of the issue tracker):

- **It is a 4B model.** The router can classify an unusual wording differently from run to run; a grounded answer can contain one item the cited Section does not say; a Trade Name can be misspelled, or the wrong Section named in the prose while the Citations are right. The server checks Citations, snippets, doses and support, but not every sentence: this is what the faithfulness review is for.
- **It refuses more than it should in some cases.** An English synonym of a word the KB writes only in Arabic counts as unsupported unless it is in the small alias list (hypertension, pregnancy, fever, ...); an Arabic comparison question with no drug named is refused; an Arabic-letter brand the mapper does not know (short or vowel-poor names such as اكتوس, Cozaar) is not recognised; "which drugs treat hypertension and malaria" is refused by the model itself although half of it is covered.
- **It can answer more than it should for one kind of question.** "Cancer drugs and their side effects" (an aspect of a condition's drugs) can be Answered by citing a contraindication line that mentions the disease; a generic word that the cited text happens to contain can slip through the support check. Diagnosis and Treatment Decision are caught by the router alone (and the self-care rule can lower a router Treatment Decision for a first-person question it judges to have no risk qualifier: a wording of a risk that its word lists do not know is answered with the general information and the pharmacist note instead of escalated) (the answer-call backstop applies only when routing is skipped or fails); the dose check does no unit conversion ("0.5 g" in the KB against "500 mg" in the answer becomes a Dose escalation, which errs on the safe side).
- **Unasked sentences in answers.** The faithfulness review found answers that add a sentence nobody asked for, such as "pregnancy guidance is not in the knowledge base" on a condition lookup, or "the price or an alternative is not in the knowledge base". Some of these are false. The model copies example words from the answer prompt. A prompt fix removed them but broke 4 other cases, so it was not kept (see [docs/test-results](docs/test-results/README.md#faithfulness-review)).
- **Franco-Arabic edge cases.** With no digit-letter word and only one Franco word, the message is labelled `en` unless the router agrees; the router's transliteration is not always word for word, in which case the raw message is searched.
- **Sessions.** Two concurrent requests in the same Session can each miss the other's turn; clients that pick the same simple `session_id` share history; if the routing call fails, an emergency that only the history reveals comes back Out of Scope (never answered) rather than Escalated.
- **Capacity.** One LLM call at a time: a request that cannot finish within the 25 s budget gets HTTP 500 "busy". Without an NVIDIA GPU (the CPU override) answers exceed the 30 s limit.
- **Operations.** An out-of-range `since` or `until` on `/admin/analytics` answers 500 instead of 400; the log tables grow without limit; a few admin-page edge cases on Windows (a read racing a replace) are noted in the issue.
- **Tested on one machine and one KB.** The 90/90 above is on the test set the pipeline was tuned against (the held-out sets were written by the same author), so it is evidence that the pipeline works, not a forecast for the hidden question set. The real internet download path of the models was not run in the final check (see [docs/submission-checks.md](docs/submission-checks.md)).

## Tests

`pip install -e ".[dev]" && pytest`: 1,277 tests pass with a stack running (that also runs the stack-dependent test files; 1 skips without the embedding libraries). CI-style, as GitHub Actions runs it, with no KB files: `MEDRAG_KB_DIR=/nonexistent pytest` gives 1,181 passed and 97 skipped (the tests that need the real `.docx` files).
No Ollama, torch or model is needed for the unit tests (the search tests use a fake embedder); `tests/test_embedding_real.py` runs only where torch (`pip install -e ".[embed]"`) and the BGE-M3 files (`MEDRAG_EMBED_PATH`) exist.

## Submission checks

[docs/submission-checks.md](docs/submission-checks.md) records, with the exact commands: the secrets scan of the repository, its full git history and the built image; the clean-clone run (what was verified and what was not); and the commands for the full from-scratch check that downloads everything.
