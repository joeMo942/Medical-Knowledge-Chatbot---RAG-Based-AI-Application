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
| `MEDRAG_KB_DIR` | `./data` (`/kb` in Docker) | KB `.docx` folder |

## Tests

`pip install -e ".[dev]" && pytest`. No Ollama or KB files are needed; tests that need the real KB skip themselves.
`tests/test_e2e_smoke.py` runs against a started stack and skips otherwise.
