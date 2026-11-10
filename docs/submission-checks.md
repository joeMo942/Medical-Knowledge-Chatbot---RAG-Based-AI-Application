# Submission checks

Done on 2026-11-10 on the test machine (RTX 5070 Laptop, Windows 11, Docker Desktop 29.4, Compose 5.1), branch `issue-13-submission`.

## 1. Secrets

Method:

- Working tree and every commit of every branch (125 commits): `git grep` / `git log --all -p -G` for `ghp_`, `github_pat_`, `hf_...`, `sk-...`, `AKIA...`, `BEGIN ... PRIVATE KEY`, `xox[abp]-`, `AIza...`, and generic `password|secret|api_key|token = "<value>"` assignments; file names in history matching `.env`, `.pem`, `.key`, `id_rsa`, `credentials`, `secret`; long base64/hex strings in tracked files.
- Built `medrag-api` image: `docker history --no-trunc` (all `ENV` and `RUN` lines), `docker image inspect` (environment, labels, user), and a throwaway container listing `.env*`, `*.pem`, `*.key`, `id_rsa*`, `credentials*`, `.netrc` and grepping `/app` and the installed package for the token patterns.

Result: **no secret found.** The only hits are test fixtures (`TOKEN = "s3cret-token"` in tests, used against an in-process app), the pinned model file hashes in `medrag/modelstore.py`, the public Python release key ID (`GPG_KEY` of the base image) and the CA certificate bundles of `certifi`/the OS. The image environment holds only `MEDRAG_*` paths and pip settings; `MEDRAG_ADMIN_TOKEN` and the optional API key are passed at run time from the operator's shell, never baked in. `.env` and `.env.*` are gitignored and dockerignored; no KB `.docx` is in git or in the image.

Noted, not a secret: the assessment brief PDF (`docs/MTI_AI_SOLUTIONS_AI_ENGINEER_TECHNICAL_ASSESSMENT.pdf`, marked confidential) is in git history (commits `204dc6f` and later removed by `551db6c`). The repository is private; decide whether to rewrite history before sharing it.

## 2. Clean-clone run

Verified: `git clone` of the branch into an empty folder, `./data` copied in (the KB is not in git), a separate compose project (`docker compose -p medragclean ...`) so all volumes were new, then only the README's commands:

- Start: `docker compose up -d --build --wait` with `MEDRAG_WHEELS_DIR` and `MEDRAG_EMBED_SRC` pointing at local copies and the Ollama models volume seeded by a local copy, so nothing large was downloaded. **Healthy after 121 s** (model copy and check 26 s, embedder load 7.7 s, index build 42.7 s).
- `python -m evaluation.run --label issue13-clean --analytics-check` in a fresh venv (`pip install -e .`): **90/90**, relevance 42/44, language 90/90, citations 44/44, snippets 53/53, 0 contract violations, latency p50 3.68 s and p95 6.04 s, analytics check "all exact numbers match" (turns 90, unanswered 46, 328,180 prompt and 12,476 completion tokens, 142 LLM calls).
- `evaluation.kb_check` (85 cases, 0 errors), `evaluation.review`, and a second `--only dev` run with the analytics check: all as the README says.
- Forced README fixes: the KB file names and where they go, the host Python and `pip install -e .` for the runner, the Arabic-in-curl problem on Windows, and "no other traffic during `--analytics-check`".

**Not verified here:** the real internet download of `qwen3.5:4b` (Ollama) and of BGE-M3 (Hugging Face), the pull of the `ollama/ollama` and Python base images, and the torch CPU wheel download from `download.pytorch.org` (a build with no wheelhouse). Their sizes are in the README; the times are not measured.

## 3. Full from-scratch check (run it yourself when you accept the downloads, about 16 GB)

**Destructive:** `docker compose down -v --rmi all` deletes this project's volumes (the models, the KB store, Sessions, the search index) and the images it uses, including the shared `medrag-api` tag. To keep your caches, use a throwaway project name instead: `docker compose -p fresh ...` (the `medrag-api` image tag is shared, so skip `--rmi` then).

```
docker compose down -v --rmi all                      # WARNING: deletes volumes and images
git clone <repository> fresh && cd fresh
mkdir data   # copy the nine KB .docx files in, with the names from the README
unset MEDRAG_WHEELS_DIR MEDRAG_EMBED_SRC              # no local seeds: everything downloads
export MEDRAG_ADMIN_TOKEN=$(openssl rand -hex 16) MEDRAG_USAGE_HEADER=1
time docker compose up -d --build --wait
python -m venv .venv && . .venv/bin/activate && pip install -e .
python -m evaluation.run --label fresh --analytics-check
```

Record the time to healthy and the score here when done.
