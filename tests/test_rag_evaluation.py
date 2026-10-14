"""Scoring, budget control and replay of the generative evaluation (no network, no cost)."""

import json
from dataclasses import replace

import pytest

from repolens.embeddings import HashingEmbedder
from repolens.evaluation import DEFAULT_EVAL_DIR, load_questions
from repolens.llm import INSUFFICIENT_TOKEN, Completion, LLMError
from repolens.models import Chunk, Hit, LLMUsage
from repolens.rag_evaluation import (
    LiveRecorder,
    ReplayLLM,
    claim_support,
    fact_accuracy,
    prompt_key,
    run_rag_evaluation,
)

TOKENS = Chunk(
    "acme/fixture", "a" * 40, "auth/tokens.py", 13, 18, "python",
    'def issue_token(user_id, secret, ttl_seconds: int = 3600):\n    return hmac.new(secret)',
)  # fmt: skip
HITS = [Hit(TOKENS, 0.9)]


class Scripted:
    """Deterministic stand-in provider: answers from the first excerpt, abstains on request."""

    def __init__(self, abstain_on: tuple[str, ...] = ("payment", "OAuth2")) -> None:
        self.abstain_on, self.calls = abstain_on, 0

    def complete(self, question: str, hits: list[Hit]) -> Completion:
        self.calls += 1
        if any(w in question for w in self.abstain_on):
            return Completion(INSUFFICIENT_TOKEN, LLMUsage("m", 100, 5, 105, 10.0, 0.0001))
        text = hits[0].chunk.text.replace("\n", " ")[:80]
        return Completion(f"The code says `{text}` [1].", LLMUsage("m", 100, 20, 120, 20.0, 0.0002))


def test_dataset_meets_requirements() -> None:
    qs = load_questions()
    assert len(qs) >= 12
    kinds = {q["kind"] for q in qs}
    assert {"single", "multi_file", "unanswerable"} <= kinds
    for q in qs:
        if q.get("answerable", True):
            assert q["expected_paths"] and q["required_facts"]
            for path in q["expected_paths"]:
                assert (DEFAULT_EVAL_DIR / "fixture_repo" / path).is_file()
        else:
            assert not q["expected_paths"]
    assert any(len(q["expected_paths"]) > 1 for q in qs)


def test_fact_accuracy() -> None:
    groups = [["hmac"], ["expir"]]
    assert fact_accuracy("Uses HMAC and checks expiry.", groups)
    assert not fact_accuracy("Uses HMAC only.", groups)
    assert not fact_accuracy("HMAC, expiry; also uses RSA.", groups, forbidden=["rsa"])


def test_claim_support_flags_invented_specifics_not_just_missing_citations() -> None:
    good = claim_support("The default lifetime is 3600 seconds [1].", HITS)
    assert good["supported"] == 1 and good["unsupported"] == 0
    # A citation is present, but the number is not in the cited excerpt.
    bad = claim_support("The default lifetime is 7200 seconds [1].", HITS)
    assert bad["unsupported"] == 1 and bad["supported"] == 0
    invented = claim_support("It calls `rotate_keys` [1].", HITS)
    assert invented["unsupported"] == 1
    path = claim_support("See auth/tokens.py lines 13-18 [1].", HITS)
    assert path["supported"] == 1
    mixed = claim_support("Plain prose with no specifics [1]. And an uncited claim.", HITS)
    assert mixed["unverifiable"] == 1 and mixed["uncited"] == 1
    out_of_range = claim_support("Uses `issue_token` [5].", HITS)
    assert out_of_range["uncited"] == 1 and out_of_range["supported"] == 0


def test_end_to_end_metrics_and_replay_is_identical() -> None:
    live = LiveRecorder(
        Scripted(), model="m", effort="low", max_calls=12, min_interval=0, sleep=lambda _s: None
    )
    report = run_rag_evaluation(HashingEmbedder(), live)
    s = report["summary"]
    assert s["questions"] == 12 and s["answerable"] == 10 and s["unanswerable"] == 2
    assert s["retrieval"]["hit@3"] == 1.0 and s["retrieval"]["hit@5"] == 1.0
    assert s["abstention"]["correct_abstention_rate"] == 1.0
    assert s["abstention"]["wrong_answers_to_unanswerable"] == 0
    assert s["citations"]["validity"] == 1.0
    assert s["fidelity"]["unsupported"] == 0
    assert s["tokens"]["prompt"] == 100 * live.calls
    assert live.calls <= 12

    recording = {
        "model": "m",
        "embedder": "hashing-8192",
        "reasoning_effort": "low",
        "entries": live.entries,
    }
    replay = ReplayLLM(recording)
    again = run_rag_evaluation(HashingEmbedder(), replay)
    assert replay.misses == 0
    # Retrieval latency is wall-clock and legitimately differs between runs.
    strip = lambda x: {k: v for k, v in x.items() if k != "latency_ms"}  # noqa: E731
    assert strip(again["summary"]) == strip(s)
    assert again["summary"]["latency_ms"]["generation"] == s["latency_ms"]["generation"]
    assert [r["answer"] for r in again["per_question"]] == [
        r["answer"] for r in report["per_question"]
    ]


def test_live_call_budget_is_a_hard_cap() -> None:
    provider = Scripted()
    live = LiveRecorder(
        provider, model="m", effort="", max_calls=3, min_interval=0, sleep=lambda _s: None
    )
    report = run_rag_evaluation(HashingEmbedder(), live)
    assert provider.calls == 3 == live.calls
    skipped = [r for r in report["per_question"] if r["llm_error"] == "budget"]
    assert skipped and all(r["status"] == "retrieval_only" for r in skipped)
    replay = ReplayLLM({"model": "m", "entries": live.entries, "reasoning_effort": ""})
    again = run_rag_evaluation(HashingEmbedder(), replay)
    assert replay.misses == 0
    assert [r["llm_error"] for r in again["per_question"]] == [
        r["llm_error"] for r in report["per_question"]
    ]


@pytest.mark.parametrize("kind", ["http_429", "http_401"])
def test_live_run_halts_after_quota_or_auth_failure(kind: str) -> None:
    class Failing:
        calls = 0

        def complete(self, question: str, hits: list[Hit]) -> Completion:
            Failing.calls += 1
            raise LLMError("x", kind=kind)

    live = LiveRecorder(
        Failing(), model="m", effort="", max_calls=12, min_interval=0, sleep=lambda _s: None
    )
    report = run_rag_evaluation(HashingEmbedder(), live)
    assert Failing.calls == 1  # no retries or further billable attempts
    errors = {r["llm_error"] for r in report["per_question"]}
    assert kind in errors and "aborted" in errors
    # Provider errors degrade to retrieval-only, never to fabricated answers.
    assert all(r["answer"] is None for r in report["per_question"])
    replay = ReplayLLM({"model": "m", "entries": live.entries, "reasoning_effort": ""})
    again = run_rag_evaluation(HashingEmbedder(), replay)
    assert replay.misses == 0
    assert {r["llm_error"] for r in again["per_question"]} == errors


def test_pacing_waits_between_calls() -> None:
    waits: list[float] = []
    live = LiveRecorder(
        Scripted(), model="m", effort="", max_calls=2, min_interval=5, sleep=waits.append
    )
    live.complete("q1", HITS)
    live.complete("payment?", HITS)
    assert len(waits) == 1 and 0 < waits[0] <= 5


def test_replay_refuses_stale_prompts_instead_of_inventing_answers() -> None:
    key = prompt_key("q", HITS, "m", "")
    recording = {
        "model": "m",
        "reasoning_effort": "",
        "entries": {key: {"text": "a [1]", "usage": None}},
    }
    replay = ReplayLLM(recording)
    assert replay.complete("q", HITS).text == "a [1]"
    with pytest.raises(LLMError) as info:
        replay.complete("a different question", HITS)
    assert info.value.kind == "replay_miss" and replay.misses == 1
    # Changing the excerpt text changes the key, so edited code cannot reuse old answers.
    changed = [Hit(replace(TOKENS, text="changed"), 0.9)]
    assert prompt_key("q", changed, "m", "") != key


def test_committed_recording_is_credential_free() -> None:
    for path in (DEFAULT_EVAL_DIR / "recordings").glob("*.json"):
        raw = path.read_text("utf-8")
        assert "AIza" not in raw and "Bearer" not in raw and "api_key" not in raw.lower()
        data = json.loads(raw)
        assert data["model"] and isinstance(data["entries"], dict)


def test_prompt_key_ignores_commit_sha() -> None:
    from dataclasses import replace

    other = [replace(h, chunk=replace(h.chunk, commit="f" * 40)) for h in HITS]
    assert prompt_key("q", HITS, "m", "low") == prompt_key("q", other, "m", "low")
    changed = [replace(h, chunk=replace(h.chunk, text="different")) for h in HITS]
    assert prompt_key("q", HITS, "m", "low") != prompt_key("q", changed, "m", "low")


def test_multi_file_hit_requires_all_expected_files() -> None:
    from repolens.evaluation import _hit

    q = {"kind": "multi_file"}
    expected = {"a.py", "b.py"}
    assert _hit(q, expected, ["a.py", "x.py"]) is False
    assert _hit(q, expected, ["a.py", "b.py"]) is True
    assert _hit({"kind": "single"}, {"a.py"}, ["x", "a.py"]) is True


def test_replay_without_recording_exits_cleanly(tmp_path, monkeypatch) -> None:
    import sys

    from repolens.rag_evaluation import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--mode",
            "replay",
            "--embedder",
            "hashing",
            "--recording",
            str(tmp_path / "no.json"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
