# Gemini Generative RAG Evaluation (v0.2.0)

## Summary

| Model | Result |
|---|---|
| `gemini-3.8-flash` (default target) | **Not validated**: 3 smoke calls returned HTTP 503 (provider side). Remains configurable. |
| `gemini-3.5-flash-lite` | **Validated live** on the same OpenAI-compatible endpoint, without `reasoning_effort`. |

## API calls and billing (verified)

- Key present (53 chars, never printed); `GET .../openai/models` -> 200, both models listed.
- `gemini-3.8-flash`: 3 calls, 3x HTTP 503, no usage data.
- `gemini-3.5-flash-lite` smoke: 1 successful call (137 prompt / 12 completion tokens, 813 ms, text `TAX_RATE is 0.08 [1].`). One earlier attempt never reached the API (local env mistake).
- `gemini-3.5-flash-lite` evaluation: 12 questions, **7 real calls, all HTTP 200** (4 questions were rejected by the similarity threshold before any LLM call; 1 call returned `INSUFFICIENT_EVIDENCE`). Budget authorized: 12.
- Total successful generation calls: 8. Evaluation tokens: 8,878 prompt + 623 completion = 9,501.
- **Billing tier could not be confirmed** (the API exposes no billing data); the user explicitly authorized the capped run. **Cost: not estimated** - no list price for this model was found and none is invented (`estimated_cost_usd` is null). On the Free Tier the charge is $0.
- `max_tokens` was sent and accepted. `reasoning_effort` was deliberately omitted for this model and remains untested live.

## Live results: `gemini-3.5-flash-lite`, hashing embedder

Embedder: **`HashingEmbedder` (lexical, 8192 buckets) - NOT semantic embeddings.** These numbers must not be read as results with real semantic embeddings. Recording: `eval/recordings/gemini-3.5-flash-lite.json`.

| Metric | Value |
|---|---|
| Questions (answerable / unanswerable) | 12 (10 / 2) |
| Retrieval hit@3 / hit@5 (multi-file needs all files) | 1.00 / 1.00 |
| Fact coverage (lexical, not semantic accuracy), all answerable (n=10) | 0.50 (5/10) |
| Fact coverage among answered (n=5) | 1.00 |
| False abstention rate | 0.40 (4/10, see below) |
| Correct abstention on unanswerable (n=2) | 1.00, 0 wrong answers |
| Structural citation validity (cited ranges match file at commit) | 1.00 (8 cited sources) |
| Cited source is an expected file | 1.00 |
| Lexical claim support (supported / (supported + unsupported)) | 0.91 (10 supported, 1 unsupported, 2 unverifiable; 3 uncited sentences) |
| Retrieval latency mean / p95 | ~4-6 ms / ~6-8 ms (varies per run) |
| Generation latency mean / median / p95 | 4,735 / 1,143 / 23,579 ms |
| Tokens (prompt / completion) | 8,878 / 623 |

### Reading the numbers honestly

- **Three distinct levels**: citation *validity* is structural (the range exists in the file); *claim support* is a lexical check of identifiers/paths/numbers against the cited excerpts; *semantic fidelity* is **not measured** (no judge, no full human review).
- **Low coverage is mostly a retrieval-gate artifact**: q3, q4, q6, q10 scored below `REPOLENS_MIN_SCORE` (0.25) with the hashing embedder, so they became `insufficient_evidence` without calling the LLM. This reflects the lexical embedder, not Gemini. The threshold was not tuned. A run with semantic embeddings was not done (Codespace memory).
- q7 was a real model failure: it answered without any citation and was discarded (`no_citations`, 23.6 s latency).
- Gemini groups citations as `[1, 2]`. The service parser originally accepted only `[n]` and discarded q8 and q9; this was fixed (coverage 0.30 -> 0.50). `claim_support()` still used a `[n]`-only regex, which inflated `uncited` (6) and understated support (0.89). It now reuses the service parser `cited_indexes` (single, grouped, out-of-range and missing citations are regression-tested). Both rescorings used the **existing recording with no new API calls**; earlier figures are superseded.
- The 3 remaining `uncited` sentences are all in q2 (formula lines without a citation marker), a genuine lack of citation per sentence.
- n=10 answerable questions over a small fixture: no statistical power.

## Reproduce offline (no API, no credentials)

```bash
python -m repolens.rag_evaluation --mode replay --embedder hashing \
  --recording eval/recordings/gemini-3.5-flash-lite.json
```

Replay serves the recorded answers and refuses (`replay_miss`) if the prompt changed. Generation latency and tokens are the recorded ones; retrieval latency is re-measured. CI runs this step.

## Retrieval-only offline baseline

`python -m repolens.evaluation --embedder hashing`: 12 questions, hit@3 = hit@5 = 1.00, citation validity 1.00 (`eval/results.json`).

## Pipeline verified by tests (no network, no cost)

Request shape and Bearer auth; usage/cost parsing (no invented cost); secret-free logs; prompt-injection delimiter handling; fallback to semantic retrieval on provider failure (503); metric scorers; call cap, pacing, halt on 401/403/404/429; replay refusal on stale prompts.

## Metrics design

Deterministic, no LLM judge: fact coverage (JSON keys `fact_coverage`; presence of human-written regex fact groups; not polarity-aware, so a contradicting answer can match), lexical claim support (identifiers, paths and numbers in each answer sentence must appear in the cited excerpts), citation validity (cited range must match the file at the commit), abstention rate on unanswerable questions, retrieval/generation latency, tokens, estimated cost. Multi-file questions count as a retrieval hit only when all expected files are in the top-k. Ranking is stable only within the candidate window fetched from ChromaDB (max(k*4, 20)); ties outside it may return different results. No exhaustive retrieval was implemented. Limits: claim support is lexical and can miss subtle misreadings; 12 questions over a small fixture give no statistical power.

## Reproduce the live run

```bash
export REPOLENS_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export REPOLENS_LLM_MODEL=gemini-3.5-flash-lite   # or gemini-3.8-flash when available
python -m repolens.rag_evaluation --smoke --allow-live
python -m repolens.rag_evaluation --mode live --embedder hashing --allow-live --max-calls 12 \
  --recording eval/recordings/gemini-3.5-flash-lite.json
```
