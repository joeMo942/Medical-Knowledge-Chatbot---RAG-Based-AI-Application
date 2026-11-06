# Drug Information Assistant

RAG chatbot answering pharmacy staff and doctors strictly from the approved Knowledge Base (see GLOSSARY.md).

## Run

1. Put the nine KB `.docx` files in `./data` (not committed to git).
2. `docker compose up -d --build` (add `--wait` to block until healthy). Needs Docker with an NVIDIA GPU (ADR 0001).
   The first start downloads the model (about 3.3 GB) into the `ollama-models` volume; watch it with
   `docker compose logs -f model-pull`. Later starts skip the download. The API reports healthy only after the model is present.
3. Open http://localhost:8000 (chat page), or `curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"What is the active ingredient in Cataflam?"}'`.
4. `docker compose down` stops it and keeps the model volume.

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
| `MEDRAG_LLM_EST_S` | `5` | initial guess of one LLM call, used by that queue check |
| `MEDRAG_MAX_OUTPUT_TOKENS` | `600` | cap on one model reply; a reply cut off at the cap is answered as out_of_scope |
| `MEDRAG_KB_DIR` | `./data` | KB `.docx` folder (fixed to the `./data` mount, `/kb`, in Docker) |

All of these except `MEDRAG_KB_DIR` are passed through by `docker-compose.yml`, with the defaults above.
`GET /health` returns 503 (`"status":"error"`) when the LLM backend is unreachable or the model is missing.

## Tests

`pip install -e ".[dev]" && pytest`. No Ollama or KB files are needed; tests that need the real KB skip themselves.
`tests/test_e2e_smoke.py` runs against a started stack and skips otherwise.
