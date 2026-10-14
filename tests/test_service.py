from dataclasses import replace
from pathlib import Path

import pytest

from repolens.citations import permalink, verify_citation
from repolens.ingest.github import IngestError
from repolens.llm import INSUFFICIENT_TOKEN, LLMError, build_user_prompt, cited_indexes
from repolens.models import Hit
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


def test_llm_insufficient_and_failure_paths(service: RepoLensService) -> None:
    service.ingest_url(URL)
    q = "How do I roll back a release?"
    service.llm = FakeLLM(INSUFFICIENT_TOKEN)
    assert service.query(q).status == "insufficient_evidence"
    service.llm = FakeLLM(error=True)
    out = service.query(q)
    assert out.status == "retrieval_only" and out.answer is None
    service.llm = FakeLLM("ok [1]")
    assert service.query(q, generate=False).status == "retrieval_only"


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
