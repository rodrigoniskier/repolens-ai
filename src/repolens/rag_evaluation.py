"""Generative RAG evaluation against hand-written ground truth.

Two modes share one scoring path:
  live    calls the configured LLM (needs --allow-live, REPOLENS_LLM_* env), hard-capped and
          paced, and records every response in a recording file.
  replay  re-scores a recording offline: no network, no credentials, no cost. Retrieval is
          re-run, and a recorded answer is reused only if the prompt hash is identical.

Metrics are deterministic string/structure checks (no LLM-as-judge); see README for limits.
Run: python -m repolens.rag_evaluation --mode replay --embedder model
"""

import argparse
import hashlib
import json
import re
import statistics
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from repolens.citations import verify_citation
from repolens.config import Settings
from repolens.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from repolens.evaluation import (
    DEFAULT_EVAL_DIR,
    FIXTURE_REPO,
    _hit,
    fixture_service,
    load_questions,
)
from repolens.llm import (
    SYSTEM_PROMPT,
    Completion,
    LLMClient,
    LLMError,
    OpenAICompatibleClient,
    build_user_prompt,
)
from repolens.models import Chunk, Hit, LLMUsage
from repolens.service import QueryResult

DEFAULT_RECORDING = DEFAULT_EVAL_DIR / "recordings" / "gemini-3.8-flash.json"
ABORT_KINDS = {"http_401", "http_403", "http_404", "http_429"}

_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ANCHORS = re.compile(
    r"`([^`]+)`"  # backticked code
    r"|\b([A-Za-z]+_[A-Za-z0-9_]+)\b"  # snake_case / SCREAMING_SNAKE
    r"|\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b"  # CamelCase
    r"|\b([\w./-]+\.(?:py|md))\b"  # file paths
    r"|(?<![\w.])(\d+(?:\.\d+)?)\b(?!\s*%)"  # plain numbers (percentages are not comparable)
)


def prompt_key(question: str, hits: list[Hit], model: str, effort: str) -> str:
    """Identity of one generation request: same key => same prompt => replayable answer."""
    # The fixture's commit SHA changes on every checkout; content identity must not depend on it.
    stable = [replace(h, chunk=replace(h.chunk, commit="")) for h in hits]
    material = "\x1f".join([model, effort, SYSTEM_PROMPT, build_user_prompt(question, stable)])
    return hashlib.sha256(material.encode()).hexdigest()


# ----------------------------------------------------------------------------- scoring


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def fact_accuracy(answer: str, groups: list[list[str]], forbidden: list[str] | None = None) -> bool:
    """True when every fact group has a matching alternative and no forbidden pattern matches."""
    if any(re.search(p, answer, re.IGNORECASE) for p in forbidden or []):
        return False
    return all(any(re.search(p, answer, re.IGNORECASE) for p in group) for group in groups)


def _anchors(sentence: str) -> set[str]:
    cleaned = _CITATION.sub("", sentence)
    return {_norm(g).strip() for m in _ANCHORS.finditer(cleaned) for g in m.groups() if g}


def _corpus(hits: list[Hit], indexes: set[int]) -> str:
    parts = []
    for i in sorted(indexes):
        c = hits[i - 1].chunk
        parts.append(f"{c.text}\n{c.path}\n{c.start_line} {c.end_line}")
    return _norm("\n".join(parts))


def _present(anchor: str, corpus: str) -> bool:
    if anchor.replace(".", "", 1).isdigit():
        return re.search(rf"(?<![\w.]){re.escape(anchor)}(?![\w.])", corpus) is not None
    return anchor in corpus


def claim_support(answer: str, hits: list[Hit]) -> dict[str, int]:
    """Check each cited sentence against the excerpts it cites.

    A sentence is `supported` when every verifiable token it contains (code identifiers, file
    paths, numbers) occurs in the cited excerpts; `unsupported` when any does not;
    `unverifiable` when it has no such token; `uncited` when it cites nothing. This is a
    lexical check: it catches invented specifics, not subtle misreadings.
    """
    counts = {"supported": 0, "unsupported": 0, "unverifiable": 0, "uncited": 0}
    for sentence in filter(None, (s.strip() for s in _SENTENCE.split(answer))):
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", sentence) if 1 <= int(n) <= len(hits)}
        if not cited:
            if _CITATION.sub("", sentence).strip(" -*•#>"):
                counts["uncited"] += 1
            continue
        anchors = _anchors(sentence)
        if not anchors:
            counts["unverifiable"] += 1
            continue
        corpus = _corpus(hits, cited)
        counts["supported" if all(_present(a, corpus) for a in anchors) else "unsupported"] += 1
    return counts


# ------------------------------------------------------------------------- LLM wrappers


class LiveRecorder:
    """Calls a real LLM under a hard call cap and pacing, and records each outcome."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        model: str,
        effort: str,
        max_calls: int,
        min_interval: float,
        sleep=time.sleep,
    ) -> None:
        self._inner, self._model, self._effort = inner, model, effort
        self._max_calls, self._min_interval, self._sleep = max_calls, min_interval, sleep
        self._last = 0.0
        self._halt: str | None = None
        self.calls = 0
        self.entries: dict[str, dict] = {}

    def complete(self, question: str, hits: list[Hit]) -> Completion:
        key = prompt_key(question, hits, self._model, self._effort)
        if self._halt:
            self.entries[key] = {"error": "aborted"}
            raise LLMError(f"LLM call skipped after {self._halt}", kind="aborted")
        if self.calls >= self._max_calls:
            self.entries[key] = {"error": "budget"}
            raise LLMError("LLM call budget exhausted", kind="budget")
        wait = self._min_interval - (time.monotonic() - self._last)
        if self.calls and wait > 0:
            self._sleep(wait)
        self.calls += 1
        self._last = time.monotonic()
        try:
            result = self._inner.complete(question, hits)
        except LLMError as exc:
            self.entries[key] = {"error": exc.kind}
            if exc.kind in ABORT_KINDS:
                self._halt = exc.kind
            raise
        completion = result if isinstance(result, Completion) else Completion(result)
        self.entries[key] = {
            "text": completion.text,
            "usage": asdict(completion.usage) if completion.usage else None,
        }
        return completion


class ReplayLLM:
    """Serves recorded answers; refuses (LLMError) when the prompt no longer matches."""

    def __init__(self, recording: dict, *, effort: str | None = None) -> None:
        self._entries = recording.get("entries", {})
        self._model = recording["model"]
        self._effort = recording.get("reasoning_effort", "") if effort is None else effort
        self.misses = 0

    def complete(self, question: str, hits: list[Hit]) -> Completion:
        entry = self._entries.get(prompt_key(question, hits, self._model, self._effort))
        if entry is None:
            self.misses += 1
            raise LLMError("no recorded response for this prompt", kind="replay_miss")
        if "error" in entry:
            raise LLMError(f"recorded failure: {entry['error']}", kind=entry["error"])
        usage = LLMUsage(**entry["usage"]) if entry.get("usage") else None
        return Completion(entry["text"], usage)


# ------------------------------------------------------------------------------ report


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "mean": round(statistics.fmean(ordered), 2),
        "median": round(statistics.median(ordered), 2),
        "p95": round(p95, 2),
        "max": round(ordered[-1], 2),
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _score_question(q: dict, res: QueryResult, root: Path, sha: str) -> dict:
    answerable = q.get("answerable", True)
    expected = set(q["expected_paths"])
    paths = [h.chunk.path for h in res.hits]
    row: dict = {
        "id": q["id"],
        "kind": q.get("kind", "single"),
        "answerable": answerable,
        "status": res.status,
        "llm_error": res.llm_error,
        "hit@3": _hit(q, expected, paths[:3]),
        "hit@5": _hit(q, expected, paths[:5]),
        "retrieval_ms": res.retrieval_ms,
        "generation_ms": res.llm_usage.latency_ms if res.llm_usage else None,
        "prompt_tokens": res.llm_usage.prompt_tokens if res.llm_usage else 0,
        "completion_tokens": res.llm_usage.completion_tokens if res.llm_usage else 0,
        "estimated_cost_usd": res.llm_usage.estimated_cost_usd if res.llm_usage else None,
        "top_paths": paths[:5],
        "answer": res.answer,
    }
    answered = res.status == "answered" and res.answer is not None
    row["accurate"] = (
        fact_accuracy(res.answer or "", q.get("required_facts", []), q.get("forbidden"))
        if answerable and answered
        else False
        if answerable
        else None
    )
    cited_hits = [res.hits[i - 1] for i in res.cited]
    row["valid_citations"] = sum(verify_citation(root, h.chunk, sha) for h in cited_hits)
    row["cited_sources"] = len(cited_hits)
    row["cites_expected_path"] = (
        any(h.chunk.path in expected for h in cited_hits) if answered and expected else None
    )
    row["claims"] = claim_support(res.answer, res.hits) if answered and res.answer else None
    return row


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["status"] == "answered"]
    claims = {k: sum(r["claims"][k] for r in answered if r["claims"]) for k in _CLAIM_KEYS}
    cited_total = sum(r["cited_sources"] for r in rows)
    gen = [r["generation_ms"] for r in rows if r["generation_ms"] is not None]
    costs = [r["estimated_cost_usd"] for r in rows if r["estimated_cost_usd"] is not None]
    return {
        "questions": len(rows),
        "answerable": len(ans),
        "unanswerable": len(unans),
        "retrieval": {
            "hit@3": _rate(sum(bool(r["hit@3"]) for r in ans), len(ans)),
            "hit@5": _rate(sum(bool(r["hit@5"]) for r in ans), len(ans)),
        },
        "answers": {
            "accuracy": _rate(sum(bool(r["accurate"]) for r in ans), len(ans)),
            "accuracy_among_answered": _rate(
                sum(bool(r["accurate"]) for r in ans if r["status"] == "answered"),
                sum(r["status"] == "answered" for r in ans),
            ),
            "false_abstention_rate": _rate(
                sum(r["status"] == "insufficient_evidence" for r in ans), len(ans)
            ),
            "provider_fallbacks": sum(bool(r["llm_error"]) for r in rows),
        },
        "abstention": {
            "correct_abstention_rate": _rate(
                sum(r["status"] == "insufficient_evidence" for r in unans), len(unans)
            ),
            "wrong_answers_to_unanswerable": sum(r["status"] == "answered" for r in unans),
        },
        "citations": {
            "validity": _rate(sum(r["valid_citations"] for r in rows), cited_total),
            "cited_sources": cited_total,
            "cites_expected_path_rate": _rate(
                sum(bool(r["cites_expected_path"]) for r in answered if r["answerable"]),
                sum(r["cites_expected_path"] is not None for r in answered),
            ),
        },
        "fidelity": {
            **claims,
            "claim_support_rate": _rate(
                claims["supported"], claims["supported"] + claims["unsupported"]
            ),
        },
        "latency_ms": {
            "retrieval": _stats([r["retrieval_ms"] for r in rows]),
            "generation": _stats(gen),
        },
        "tokens": {
            "prompt": sum(r["prompt_tokens"] for r in rows),
            "completion": sum(r["completion_tokens"] for r in rows),
            "total": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
        },
        "estimated_cost_usd": round(sum(costs), 6) if costs else None,
    }


_CLAIM_KEYS = ("supported", "unsupported", "unverifiable", "uncited")


def run_rag_evaluation(
    embedder: Embedder, llm: LLMClient, eval_dir: Path = DEFAULT_EVAL_DIR
) -> dict:
    questions = load_questions(eval_dir)
    with fixture_service(embedder, eval_dir) as (service, root, sha):
        service.llm = llm
        rows = [
            _score_question(q, service.query(q["question"], top_k=5, repo=FIXTURE_REPO), root, sha)
            for q in questions
        ]
    return {"embedder": embedder.name, "summary": summarize(rows), "per_question": rows}


# ---------------------------------------------------------------------------------- CLI


def _load_recording(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _smoke(settings: Settings) -> None:
    """One minimal real call to prove access, auth and usage reporting."""
    client = OpenAICompatibleClient(settings)
    probe = Hit(Chunk(FIXTURE_REPO, "0" * 40, "x.py", 1, 1, "python", "TAX_RATE = 0.08"), 1.0)
    try:
        result = client.complete("What is TAX_RATE?", [probe])
    except LLMError as exc:
        print(json.dumps({"error": exc.kind}))
        raise SystemExit(2) from None
    print(json.dumps({"text": result.text, "usage": asdict(result.usage)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["live", "replay"], default="replay")
    parser.add_argument("--embedder", choices=["hashing", "model"], default="model")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--output", type=Path, help="Write the JSON report here.")
    parser.add_argument("--max-calls", type=int, default=12, help="Hard cap on live LLM calls.")
    parser.add_argument("--min-interval", type=float, default=6.0, help="Seconds between calls.")
    parser.add_argument("--smoke", action="store_true", help="Make a single live call and exit.")
    parser.add_argument(
        "--allow-live", action="store_true", help="Required to call the provider (may bill)."
    )
    args = parser.parse_args()

    settings = Settings()
    if (args.mode == "live" or args.smoke) and not args.allow_live:
        parser.error("live calls require --allow-live")
    if args.smoke:
        if not settings.llm_enabled:
            parser.error("set REPOLENS_LLM_MODEL (and REPOLENS_LLM_API_KEY) first")
        _smoke(settings)
        return

    embedder: Embedder = (
        HashingEmbedder()
        if args.embedder == "hashing"
        else SentenceTransformerEmbedder(settings.embedding_model)
    )
    meta: dict
    if args.mode == "live":
        if not settings.llm_enabled:
            parser.error("set REPOLENS_LLM_MODEL (and REPOLENS_LLM_API_KEY) first")
        llm: LiveRecorder | ReplayLLM = LiveRecorder(
            OpenAICompatibleClient(settings),
            model=settings.llm_model,
            effort=settings.llm_reasoning_effort,
            max_calls=args.max_calls,
            min_interval=args.min_interval,
        )
        report = run_rag_evaluation(embedder, llm, args.eval_dir)
        meta = {
            "mode": "live",
            "model": settings.llm_model,
            "reasoning_effort": settings.llm_reasoning_effort,
            "llm_calls": llm.calls,
        }
        recording = {
            "model": settings.llm_model,
            "embedder": embedder.name,
            "reasoning_effort": settings.llm_reasoning_effort,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "entries": llm.entries,
        }
        args.recording.parent.mkdir(parents=True, exist_ok=True)
        args.recording.write_text(json.dumps(recording, indent=2) + "\n", encoding="utf-8")
    else:
        if not args.recording.exists():
            parser.error(
                f"no recording at {args.recording}; run with --mode live --allow-live first"
            )
        recording = _load_recording(args.recording)
        if recording.get("embedder") != embedder.name:
            parser.error(
                f"recording was made with embedder {recording.get('embedder')!r}, "
                f"not {embedder.name!r}"
            )
        llm = ReplayLLM(recording)
        report = run_rag_evaluation(embedder, llm, args.eval_dir)
        meta = {
            "mode": "replay",
            "model": recording["model"],
            "reasoning_effort": recording.get("reasoning_effort", ""),
            "recorded_at": recording.get("recorded_at"),
            "replay_misses": llm.misses,
        }
    report = {**meta, **report}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
