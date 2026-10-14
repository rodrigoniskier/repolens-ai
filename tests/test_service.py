from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest

from repolens.citations import permalink, verify_citation
from repolens.ingest.github import IngestError
from repolens.llm import INSUFFICIENT_TOKEN, LLMError, build_user_prompt, cited_indexes
from repolens.models import Chunk, Hit
from repolens.service import RepoLensService

URL = "https://github.com/acme/fixture"


class FakeLLM:
    def __init__(self, answer: str | None = None, error: bool = False) -> None:
        self.answer, self.error, self.prompts = answer, error, []

    def complete(self, question: str, hits: list[Hit]) -> str:
        self.prompts.append(build_user_prompt(question, hits))
        if self.error:
            raise LLMError("LLM request failed: boom")
        return self.answer or ""


def test_ingest_and_retrieval_only(service: RepoLensService, git_fixture: tuple[Path, str]) -> None:
    root, sha = git_fixture
    result = service.ingest_url(URL)
    assert result.repo == "acme/fixture" and result.commit == sha
    assert result.files_indexed >= 5 and result.chunks_indexed >= 5

    out = service.query("Which entry is evicted when the LRU cache is full?", top_k=3)
    assert out.status == "retrieval_only" and out.answer is None
    assert out.hits[0].chunk.path == "cache/lru.py"
    for hit in out.hits:
        assert hit.chunk.commit == sha
        assert verify_citation(root, hit.chunk, sha)
        assert permalink(hit.chunk).startswith(f"https://github.com/acme/fixture/blob/{sha}/")


def test_reingest_replaces_instead_of_duplicating(service: RepoLensService) -> None:
    first = service.ingest_url(URL)
    service.ingest_url(URL)
    assert service.store.count() == first.chunks_indexed


def test_insufficient_evidence_on_unrelated_question(service: RepoLensService) -> None:
    service.ingest_url(URL)
    out = service.query("zzzqqq xylophone quantum banana", top_k=3)
    assert out.status == "insufficient_evidence" and out.answer is None


def test_insufficient_evidence_on_empty_index(service: RepoLensService) -> None:
    assert service.query("anything at all").status == "insufficient_evidence"


def test_reindexing_empty_repository_leaves_search_empty(
    service: RepoLensService, tmp_path: Path
) -> None:
    service.ingest_url(URL)
    empty = tmp_path / "empty"
    empty.mkdir()
    service.ingest_directory(empty, repo="acme/fixture", commit="e" * 40)
    assert service.store.count() == 0
    assert service.query("How do I roll back a release?").hits == []


def test_llm_answer_requires_valid_citation(service: RepoLensService) -> None:
    service.ingest_url(URL)
    q = "How do I roll back a release?"
    service.llm = FakeLLM("Run the rollback script [1].")
    out = service.query(q)
    assert out.status == "answered" and out.cited == [1]

    service.llm = FakeLLM("Made-up answer with no citation.")
    assert service.query(q).status == "retrieval_only"

    service.llm = FakeLLM("See [99].")
    assert service.query(q).status == "retrieval_only"
    service.llm = FakeLLM("Claim [99]; valid source [1].")
    out = service.query(q)
    assert out.status == "retrieval_only" and out.answer is None and out.cited == []


def test_llm_insufficient_and_failure_paths(service: RepoLensService) -> None:
    service.ingest_url(URL)
    q = "How do I roll back a release?"
    service.llm = FakeLLM(INSUFFICIENT_TOKEN)
    assert service.query(q).status == "insufficient_evidence"
    service.llm = FakeLLM(error=True)
    out = service.query(q)
    assert out.status == "retrieval_only" and out.answer is None
    service.llm = FakeLLM("ok [1]")
    disabled = service.query(q, generate=False)
    assert disabled.status == "retrieval_only"
    assert "disabled" in disabled.message.lower()


def test_unconfigured_llm_message_is_accurate(service: RepoLensService) -> None:
    service.ingest_url(URL)
    out = service.query("How do I roll back a release?")
    assert out.status == "retrieval_only"
    assert "no llm configured" in out.message.lower()


def test_insufficient_sentinel_must_be_the_complete_answer(service: RepoLensService) -> None:
    service.ingest_url(URL)
    service.llm = FakeLLM(f'The model may return "{INSUFFICIENT_TOKEN}" [1].')
    out = service.query("How do I roll back a release?")
    assert out.status == "answered" and out.cited == [1]
    service.llm = FakeLLM(f"  {INSUFFICIENT_TOKEN} \n")
    assert service.query("How do I roll back a release?").status == "insufficient_evidence"


def test_reingest_failure_preserves_active_generation(
    service: RepoLensService, git_fixture: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = git_fixture
    first = service.ingest_url(URL)
    original_add = service.store._col.add

    def add_then_fail(*args, **kwargs):
        original_add(*args, **kwargs)
        raise RuntimeError("simulated Chroma write failure")

    monkeypatch.setattr(service.store._col, "add", add_then_fail)
    with pytest.raises(RuntimeError, match="simulated Chroma write failure"):
        service.ingest_directory(root, repo="acme/fixture", commit="b" * 40)

    assert service.store.count() == first.chunks_indexed
    assert service.store.repos()[0]["commit"] == first.commit
    assert service.query("How do I roll back a release?").hits[0].chunk.commit == first.commit


def test_reingest_embedding_failure_preserves_active_generation(
    service: RepoLensService, git_fixture: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = git_fixture
    first = service.ingest_url(URL)
    original_embed = service.store._embedder.embed

    def fail_embed(texts: list[str]) -> list[list[float]]:
        if len(texts) > 1:
            raise RuntimeError("simulated embedding failure")
        return original_embed(texts)

    monkeypatch.setattr(service.store._embedder, "embed", fail_embed)
    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        service.ingest_directory(root, repo="acme/fixture", commit="d" * 40)

    assert service.store.count() == first.chunks_indexed
    assert service.store.repos()[0]["commit"] == first.commit
    assert service.query("How do I roll back a release?").hits[0].chunk.commit == first.commit


def test_queries_keep_previous_generation_during_reindex(
    service: RepoLensService, git_fixture: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = git_fixture
    first = service.ingest_url(URL)
    original_add = service.store._col.add
    staged = Event()
    continue_ingest = Event()

    def add_then_pause(*args, **kwargs):
        original_add(*args, **kwargs)
        staged.set()
        if not continue_ingest.wait(timeout=5):
            raise TimeoutError("test did not release staged generation")

    monkeypatch.setattr(service.store._col, "add", add_then_pause)
    result: list[object] = []

    def reingest() -> None:
        result.append(service.ingest_directory(root, repo="acme/fixture", commit="c" * 40))

    thread = Thread(target=reingest)
    thread.start()
    try:
        assert staged.wait(timeout=5)
        during = service.query("How do I roll back a release?")
        assert during.hits[0].chunk.commit == first.commit
    finally:
        continue_ingest.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(result) == 1
    after = service.query("How do I roll back a release?")
    assert after.hits[0].chunk.commit == "c" * 40


def test_prompt_marks_excerpts_as_untrusted_data(service: RepoLensService) -> None:
    service.ingest_url(URL)
    llm = FakeLLM("x [1]")
    service.llm = llm
    service.query("How do I roll back a release?")
    assert '<excerpt n="1"' in llm.prompts[0]


def test_invalid_url_is_rejected(service: RepoLensService) -> None:
    with pytest.raises(IngestError):
        service.ingest_url("https://example.com/a/b")


def test_cited_indexes() -> None:
    assert cited_indexes("a [1] b [3] c [7] [0]", 3) == {1, 3}


def test_verify_citation_detects_tampering(
    service: RepoLensService, git_fixture: tuple[Path, str]
) -> None:
    root, sha = git_fixture
    service.ingest_url(URL)
    chunk = service.query("LRU cache evicted").hits[0].chunk
    assert verify_citation(root, chunk, sha)
    assert not verify_citation(root, chunk, "0" * 40)
    (root / chunk.path).write_text("changed\n")
    assert not verify_citation(root, chunk, sha)
    assert not verify_citation(root, replace(chunk, path="../../etc/passwd"), sha)


def test_permalink_encodes_url_reserved_path_characters() -> None:
    chunk = replace(_sample_chunk(), path="docs/a#b% c?.md")
    assert permalink(chunk) == (
        f"https://github.com/acme/fixture/blob/{chunk.commit}/docs/a%23b%25%20c%3F.md"
        f"#L{chunk.start_line}-L{chunk.end_line}"
    )


def _sample_chunk() -> Chunk:
    return Chunk(
        repo="acme/fixture",
        commit="a" * 40,
        path="docs/a.md",
        start_line=1,
        end_line=1,
        language="markdown",
        text="content",
    )


def test_similarity_score_keeps_full_precision(
    service: RepoLensService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service.ingest_url(URL)
    first = service.query("How do I roll back a release?").hits[0].chunk

    def query(**_kwargs):
        return {
            "documents": [[first.text]],
            "metadatas": [
                [
                    {
                        "repo": first.repo,
                        "commit": first.commit,
                        "path": first.path,
                        "start_line": first.start_line,
                        "end_line": first.end_line,
                        "language": first.language,
                    }
                ]
            ],
            "distances": [[0.123456789]],
        }

    monkeypatch.setattr(service.store._col, "query", query)
    hits = service.store.search("test", 1)
    assert hits[0].score == pytest.approx(1.0 - 0.123456789, abs=1e-12)
