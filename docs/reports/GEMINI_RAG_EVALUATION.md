# Gemini 3.8 Flash — Generative RAG Evaluation (v0.2.0)

**Status: the live generative evaluation was NOT executed. No generative metrics are reported, because none were measured.**

## What happened (verified)

| Step | Result |
|---|---|
| Key access check (`GET .../v1beta/openai/models`, no generation) | HTTP 200; `models/gemini-3.8-flash` is listed |
| Smoke call #1 (`--smoke --allow-live`) | HTTP 503 Service Unavailable |
| Smoke call #2 (after 45 s) | HTTP 503 |
| Smoke call #3 (after 90 s) | HTTP 503 |
| Evaluation calls (up to 12 authorized) | **0 made** — the smoke call never succeeded, so the evaluation budget was left untouched |

- API calls that generated content: 0 successful, 3 failed with 503 (server side, not auth/quota). Failed calls return no usage data; no tokens are known and no cost can be attributed. The 503 responses are not expected to be billable, but this was not verifiable.
- Billing tier: **could not be determined** (the API exposes no billing information). Published pricing for 3.8 Flash Standard: Free Tier free of charge; Paid $0.75 in / $3.75 out per 1M tokens (thinking tokens billed as output). The user explicitly authorized the capped live run (1 smoke + ≤12 calls, estimated worst case well under $0.10). Billing settings were not touched.
- Untested live: whether the endpoint accepts `reasoning_effort: low` and `max_tokens` (the request never got past the 503). The offline tests only verify the request shape.

## Measured without the provider (offline, reproducible)

`python -m repolens.evaluation --embedder hashing` over 12 questions (10 answerable, 2 unanswerable): hit@3 = 1.00, hit@5 = 1.00, citation validity = 1.00, retrieval latency mean 4.67 ms (`eval/results.json`). This is a lexical baseline on a tiny fixture. Retrieval with the real embedding model was not re-measured for the 12-question set (the process was killed by the memory-limited Codespace).

## Pipeline verified by tests (no network, no cost)

Request shape and Bearer auth; usage/cost parsing (no invented cost); secret-free logs; prompt-injection delimiter handling; fallback to semantic retrieval on provider failure (503); metric scorers; call cap, pacing, halt on 401/403/404/429; replay refusal on stale prompts.

## Metrics design

Deterministic, no LLM judge: fact accuracy (human-written regex fact groups), lexical claim support (identifiers, paths and numbers in each answer sentence must appear in the cited excerpts), citation validity (cited range must match the file at the commit), abstention rate on unanswerable questions, retrieval/generation latency, tokens, estimated cost. Limits: claim support is lexical and can miss subtle misreadings; 12 questions over a small fixture give no statistical power.

## Reproduce the live run later

```bash
export REPOLENS_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export REPOLENS_LLM_MODEL=gemini-3.8-flash REPOLENS_LLM_REASONING_EFFORT=low
python -m repolens.rag_evaluation --smoke --allow-live
python -m repolens.rag_evaluation --mode live --embedder model --allow-live --max-calls 12 --output eval/results-gemini-live.json
python -m repolens.rag_evaluation --mode replay --embedder model   # offline replay of the recording
```
