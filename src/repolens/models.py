from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous line range of a source file, pinned to a commit."""

    repo: str
    commit: str
    path: str
    start_line: int
    end_line: int
    language: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.repo}@{self.commit[:12]}:{self.path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Observability data for one LLM call. Cost is an estimate at paid-tier list prices."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float | None
