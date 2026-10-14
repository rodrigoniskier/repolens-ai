"""Orchestrates ingestion and querying."""

import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from repolens.chunking import chunk_file
from repolens.config import Settings
from repolens.ingest.github import RepoRef, clone_repository, parse_repo_url
from repolens.ingest.scanner import ScanReport, scan_directory
from repolens.llm import (
    INSUFFICIENT_TOKEN,
    Completion,
    LLMClient,
    LLMError,
    cited_indexes,
    has_invalid_citations,
)
from repolens.models import Hit, LLMUsage
from repolens.store import VectorStore

AnswerStatus = Literal["answered", "retrieval_only", "insufficient_evidence"]
Cloner = Callable[[RepoRef], AbstractContextManager[tuple[Path, str]]]


@dataclass(slots=True)
class IngestResult:
    repo: str
    commit: str
    files_indexed: int
    chunks_indexed: int
    total_bytes: int
    truncated: bool
    skipped: dict[str, int]


@dataclass(slots=True)
class QueryResult:
    status: AnswerStatus
    answer: str | None
    message: str
    hits: list[Hit]
    cited: list[int]
    retrieval_ms: float = 0.0
    llm_usage: LLMUsage | None = None
    llm_error: str | None = None


class RepoLensService:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        llm: LLMClient | None = None,
        cloner: Cloner | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self._cloner = cloner or self._default_cloner
        self._ingest_lock = threading.Lock()

    def _default_cloner(self, ref: RepoRef) -> AbstractContextManager[tuple[Path, str]]:
        return clone_repository(
            ref,
            timeout_seconds=self.settings.clone_timeout_seconds,
            max_bytes=self.settings.max_clone_bytes,
        )

    def ingest_url(self, url: str) -> IngestResult:
        ref = parse_repo_url(url)
        with self._cloner(ref) as (path, sha):
            return self.ingest_directory(path, repo=ref.full_name, commit=sha)

    def ingest_directory(self, root: Path, *, repo: str, commit: str) -> IngestResult:
        s = self.settings
        report: ScanReport = scan_directory(
            root,
            max_files=s.max_files,
            max_file_bytes=s.max_file_bytes,
            max_total_bytes=s.max_total_bytes,
        )
        chunks = [
            c
            for f in report.files
            for c in chunk_file(
                f, repo=repo, commit=commit, max_lines=s.chunk_lines, overlap=s.chunk_overlap
            )
        ]
        with self._ingest_lock:
            self.store.replace_repo(repo, chunks)
        return IngestResult(
            repo=repo,
            commit=commit,
            files_indexed=len(report.files),
            chunks_indexed=len(chunks),
            total_bytes=report.total_bytes,
            truncated=report.truncated,
            skipped=dict(report.skipped),
        )

    def query(
        self, question: str, *, top_k: int = 5, repo: str | None = None, generate: bool = True
    ) -> QueryResult:
        start = time.perf_counter()
        hits = self.store.search(question, top_k, repo)
        retrieval_ms = round((time.perf_counter() - start) * 1000, 2)

        def result(status: AnswerStatus, answer: str | None, message: str, **extra) -> QueryResult:
            cited = extra.pop("cited", [])
            return QueryResult(
                status, answer, message, hits, cited, retrieval_ms=retrieval_ms, **extra
            )

        if not hits or hits[0].score < self.settings.min_score:
            return result(
                "insufficient_evidence",
                None,
                "Insufficient evidence: no indexed excerpt is similar enough to the question.",
            )
        if not generate:
            return result(
                "retrieval_only",
                None,
                "Answer generation disabled: showing the most relevant excerpts only.",
            )
        if self.llm is None:
            return result(
                "retrieval_only",
                None,
                "No LLM configured: showing the most relevant excerpts only.",
            )
        try:
            completion = self.llm.complete(question, hits)
        except LLMError as exc:
            return result(
                "retrieval_only",
                None,
                f"{exc}. Falling back to semantic retrieval: showing excerpts only.",
                llm_error=exc.kind,
            )
        if isinstance(completion, str):
            completion = Completion(completion)
        answer, usage = completion.text, completion.usage
        if answer.strip() == INSUFFICIENT_TOKEN:
            return result(
                "insufficient_evidence",
                None,
                "The model judged the retrieved excerpts insufficient to answer.",
                llm_usage=usage,
            )
        if has_invalid_citations(answer, len(hits)):
            return result(
                "retrieval_only",
                None,
                "The generated answer had out-of-range citations and was discarded.",
                llm_usage=usage,
                llm_error="invalid_citations",
            )
        cited = sorted(cited_indexes(answer, len(hits)))
        if not cited:
            return result(
                "retrieval_only",
                None,
                "The generated answer had no valid citations and was discarded.",
                llm_usage=usage,
                llm_error="no_citations",
            )
        return result(
            "answered",
            answer,
            "Answer generated from the cited excerpts.",
            cited=cited,
            llm_usage=usage,
        )
