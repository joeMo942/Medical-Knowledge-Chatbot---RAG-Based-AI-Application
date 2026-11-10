# Token usage and performance

What each model call of the pipeline costs in tokens and time, measured on the test hardware, and what could be
cut. Every number says whether it could feed an admin panel in production or is only useful while developing, and why.

Raw results (`evaluation/results/`) are gitignored, so the numbers below are copied here with their source.

## How it was measured

- **Test hardware:** the author's laptop (ADR 0001): NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM (8,151 MiB reported by
  `nvidia-smi`), 31 GB RAM, Core Ultra 9 275HX (24 threads), Windows 11 with Docker Desktop.
- **Models:** `qwen3.5:4b` (Ollama 0.40.15, 4.2 B parameters, Q4_K_M, context 8192, temperature 0, thinking off). `ollama ps`
  during the run showed it 100% on the GPU (3.4 GB loaded). BGE-M3 runs on the CPU inside the `api` container (4 threads per question).
- **Workload:** the 90 turns of `evaluation/cases.json` (85 cases, one request at a time), start to finish through `POST /chat`:
  `MEDRAG_USAGE_HEADER=1 docker compose up -d --wait`, then `python -m evaluation.run --label <x> --analytics-check`.
- **Token counts are what the model server reports, not estimates:** Ollama's `prompt_eval_count` and `eval_count` for the two LLM calls;
  for the embedding call, the BGE-M3 tokenizer's own count of the question (BGE-M3 runs in-process, so there is no server to report).
  They are logged per call in the `call_log` table and summed by `GET /admin/analytics`.
- **Reproducible:** the model runs at temperature 0, so token counts repeat exactly. The run `issue9a` (#9, 2026-11-10) and the clean-clone run
  `issue13-clean` (this issue, 2026-11-10, commit `95f009b`) have identical totals, call for call. Latencies move by a few percent between runs.
- **`--analytics-check`** compares the server's analytics with the runner's own rows: turns 90, unanswered 46, status counts, and prompt
  tokens 328,180, completion tokens 12,476, LLM calls 142 all matched exactly in both runs.

## Tokens per model call

From `GET /admin/analytics` after the run (`tokens_by_call_type`, `tokens_per_turn`), 90 turns:

| call | calls | prompt tokens, total | completion tokens, total | per call (mean) | latency per call (mean, server side) |
| --- | ---: | ---: | ---: | --- | ---: |
| routing (LLM) | 83 | 225,159 | 3,037 | 2,713 prompt + 37 completion | 1,031 ms |
| answer (LLM) | 59 | 103,021 | 9,439 | 1,746 prompt + 160 completion | 3,194 ms |
| embedding (BGE-M3, CPU) | 62 | 2,022 | 0 | 33 prompt | 133 ms |
| all three | 204 | 330,202 | 12,476 | | |

Per turn (mean over the 90 turns, all call types): **3,669 prompt + 139 completion = 3,807.5 tokens** (analytics `tokens_per_turn`).
The LLM calls alone are 3,646 prompt + 139 completion per turn, 1.58 LLM calls per turn (the `X-Medrag-Usage` figures of the run report).

Why the counts differ per turn: 7 of the 90 turns are rule-flagged emergencies and make no LLM call and no search; 21 more are escalated by the router
after the routing call alone (no search); 3 Out of Scope turns are searched but nothing matches (the no-match gate), so there is no answer call; the other 59 make both
LLM calls (44 Answered, 15 Out of Scope). Query length is small
everywhere (routing input is the fixed system prompt plus the message), so the spread is in the answer call (839 to 1,746 prompt tokens in the
direct measurement below, depending on the 1 to 4 Sections retrieved).

| turns | count | mean latency |
| --- | ---: | ---: |
| Answered (routing + answer) | 44 | 4.70 s |
| Out of Scope, both LLM calls | 15 | 3.54 s |
| Out of Scope, routing only (no Section matched) | 3 | 1.03 s |
| Escalated by the router (routing only) | 21 | 0.98 s |
| Escalated by a rule (no LLM call) | 7 | 0.03 s |

(derived from the per-request rows of `issue13-clean`; the 30 s limit is never near.) End to end over all 90 turns: **p50 3.68 s, p95 6.04 s, max 7.23 s,
mean 3.15 s** (client side; the server-side p50 and p95 were 3,658 and 6,077 ms in the analytics output). Earlier runs on the same hardware: p50 3.52 to 3.80 s,
p95 5.87 to 6.20 s (`issue7e`, `issue9a`, `issue11a`).

### Speed of the model itself

Read from Ollama's own durations, calling the real routing and answer prompts straight against Ollama from inside the `api` container
(script not committed; 5 questions, 2 timed passes, so n = 10 per row; five real KB questions with the BM25 top 4 Sections, no embedding):

| call | prompt tokens | prefill (prompt evaluation) | completion tokens | decode | whole call |
| --- | ---: | --- | ---: | --- | ---: |
| routing | 2,709 (2,707 to 2,713) | 0.30 s, about 9,200 tokens/s | 35 | 0.49 s, **73 tokens/s** | 1.03 s |
| answer | 1,464 (839 to 1,746) | 0.10 s, about 14,700 tokens/s | 198 | 3.27 s, **60 tokens/s** | 3.92 s |

Decode speed is 60 to 73 tokens per second; the compose default `MEDRAG_DECODE_TOK_S=53` used to size the reply cap is deliberately below it.
The rest of a call's time (routing: 1.03 s of which 0.30 s prefill and 0.49 s decode) is request handling, schema-constrained sampling set-up and model-server overhead.
Other speeds: BGE-M3 on 4 CPU threads embeds one question in about 133 ms on average in the run (50 ms in a quiet measurement, the README's Search section); the search
index of the 113 Sections took 42.7 s to build on the first start (embedder load 7.7 s), and a restart with the cached index is healthy in about 8 s.

## Production metric or development only

| number | where it comes from | production or development | why |
| --- | --- | --- | --- |
| prompt and completion tokens per call kind, per turn | `call_log` table, `GET /admin/analytics` (`tokens_by_call_type`, `tokens_per_turn`) | **production** (already feeds the endpoint) | exact counts from the model server, logged for every call on the request path at no extra cost; capacity and spend planning need them per kind and over time. Safe: no message text, only counts |
| mean latency per call kind | same (`by_kind[].mean_latency_ms`) | **production** | tells which call slows the system down (here the answer call), and a slow router is visible at once |
| end-to-end latency p50 / p95, volume, unanswered | same (`latency_ms`, `summary`, `volume`) | **production** | an operations and quality signal: how many users wait over a few seconds and how many questions get no answer |
| `X-Medrag-Usage` header (tokens and LLM calls of one request) | response header, off unless `MEDRAG_USAGE_HEADER=1` | **development** | it is what lets the eval runner total tokens per run and cross-check the analytics, but a header on every response is an internal detail a client should not rely on, and the analytics endpoint already holds the same data durably |
| per-status latency table above | derived offline from the eval run's per-request rows | **development** | the analytics endpoint groups latency by call kind and overall, not by status; needs the run's labelled cases, or a `status` grouping added to the endpoint |
| decode and prefill tokens/s | Ollama's `eval_duration` / `prompt_eval_duration`, read by the ad-hoc script | **development** | the API's `LLMResult` carries these durations but they are not stored in `call_log`, so the panel cannot show them today; storing them would make decode speed a cheap production metric (a drop means the GPU is throttled or the model fell back to the CPU) |
| index build time, model load time, container start | server log lines (`embedder loaded in 7.7s, index built in 42.7s`) | **development** (a startup check) | happens once per start, so it is for tuning and for a first-run check, not for a time series |
| share of the call's cost that is the prompt (65.7% routing prompt, see below) | computed from the totals above | **development** | a design lens for choosing what to cut, recomputed whenever the prompt changes |

## The most expensive call

By tokens the **routing call** is the most expensive: its prompt is 225,159 of the 342,678 tokens of the run (**65.7%**; answer prompt 30.1%, answer completion 2.8%,
routing completion 0.9%, embedding 0.6%). Each routing call sends about 2,670 tokens of fixed system prompt (the rules and 35 few-shot examples; the
message itself is about 35 tokens), identical on every call, and gets back about 37 tokens of JSON.

By time the **answer call** is the most expensive: 3.19 s mean, against 1.03 s for routing, 133 ms for embedding. It spends about 2.7 s of that
decoding 160 to 200 tokens at 60 tokens/s; its prefill is about 0.1 s. Across the run the answer call takes about 67% of all model time (59 x 3.19 s = 188 s of 282 s; routing 86 s, embedding 8 s).

So the two diagnoses differ, and so does the cure: the router costs tokens, the answer call costs seconds.

## What I would do about it

The effects are **estimates** from the measurements above, not measured; any change to the router prompt or to when it runs must be re-run through the 90-turn eval first, because
the few-shot examples are what fixed the Emergency, Dose and Franco-Arabic misses (see MODEL_DECISION.md).

1. **Fewer, shorter router examples.** Halving the system prompt (about 2,670 to 1,350 tokens) would cut about 1,300 prompt tokens per routed call,
   about 110,000 of the 342,678 tokens of this run (**about -32% tokens overall**), and about 0.15 s per routed call at the measured 9,200 tokens/s prefill
   (about 5% of a typical 3 s turn). Cost: router accuracy, so it needs the eval.
2. **Skip the router where the rules already decide.** It is already skipped for rule-flagged emergencies (7 of 90 turns). A safe, narrower extension is a plain
   single-turn lookup in Latin script that names a KB drug and carries no person words (the router adds nothing there). Saving per skipped turn: 2,713 prompt tokens and about
   1.0 s. With a share *s* of turns skipped, the overall token saving is about *s* x 66% and the time saving about *s* x 1 s per turn. The router also rewrites Franco-Arabic and follow-ups and is the only detector
   of Diagnosis and Treatment Decision, so only a very narrow rule is safe; the share *s* is not measured.
3. **Reuse the fixed prefix (prompt caching).** About 98% of the routing prompt never changes. In this setup it is recomputed each call: four consecutive routing calls in the
   measurement above reported 2,703 to 2,708 prompt tokens each (a reused prefix would show up as a lower count) and about 0.3 s of prefill each. Ollama's
   `keep_alive` (24 h, already set) only keeps the weights loaded; it does not keep the prompt cache. Getting a working prefix cache (a different model server or runtime flag, or one
   server instance per call type so the routing prefix is not evicted by the answer prompt) would not change the token counts but could save up to about 0.25 s of the 1.03 s of a routing call
   (**about -25% routing time**, at most). Untested: I did not try other runtimes.
4. **Smaller output in the answer call.** Decode time is the bulk of the answer. Citation snippets are about 24% of the characters the model writes in an answered reply (2,694 of 11,060 characters in
   the 44 answered replies of this run); the server already replaces any snippet that is not a real line of the Section, so it could pick the snippet itself from the cited `section_id`. Estimated effect:
   about 15 to 25% fewer answer completion tokens, so about 0.4 to 0.7 s per answered turn. Needs the eval, since a snippet the model chooses is the evidence the answer rests on.
5. **Smaller context window.** `num_ctx` is 8,192 but the longest prompt measured is about 2,700 tokens (the routing call; a long Session history adds a little), so 4,096 would do. This does not change tokens and I do not expect a visible
   speed change; it would shrink the KV cache and leave more of the 8 GB VRAM free (not measured).

Items 1 and 2 attack the largest token consumer; items 3 and 4 attack the slowest steps. If only one change were made, it would be item 1, since the router prompt is two thirds of
all tokens and the change is a prompt edit with a clear test (the eval).
