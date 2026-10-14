"""FastAPI application."""

from contextlib import asynccontextmanager
from dataclasses import asdict
from importlib import resources
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from repolens import __version__
from repolens.citations import permalink
from repolens.config import Settings
from repolens.embeddings import SentenceTransformerEmbedder
from repolens.ingest.github import IngestError
from repolens.llm import OpenAICompatibleClient
from repolens.models import Hit
from repolens.service import RepoLensService
from repolens.store import VectorStore


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    indexed_chunks: int
    embedding_model: str
    llm_configured: bool


class IngestRequest(BaseModel):
    repo_url: str = Field(
        max_length=200,
        description="Public repository URL: https://github.com/<owner>/<repo>",
        examples=["https://github.com/pallets/click"],
    )


class IngestResponse(BaseModel):
    repo: str
    commit: str
    files_indexed: int
    chunks_indexed: int
    total_bytes: int
    truncated: bool = Field(description="True if file/volume limits stopped indexing early.")
    skipped: dict[str, int] = Field(description="Skipped file counts by reason.")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=10)
    repo: str | None = Field(None, max_length=200, description="Restrict to 'owner/name'.")
    generate: bool = Field(True, description="Use the configured LLM, if any.")


class Source(BaseModel):
    n: int
    repo: str
    path: str
    start_line: int
    end_line: int
    commit: str
    score: float
    url: str
    text: str
    cited: bool = False


class QueryResponse(BaseModel):
    status: Literal["answered", "retrieval_only", "insufficient_evidence"]
    answer: str | None = Field(None, description="Only set when an LLM produced a cited answer.")
    message: str
    sources: list[Source]


class RepoInfo(BaseModel):
    repo: str
    commit: str
    chunks: int


def _source(n: int, hit: Hit, cited: bool) -> Source:
    c = hit.chunk
    return Source(
        n=n,
        repo=c.repo,
        path=c.path,
        start_line=c.start_line,
        end_line=c.end_line,
        commit=c.commit,
        score=hit.score,
        url=permalink(c),
        text=c.text,
        cited=cited,
    )


def build_service(settings: Settings) -> RepoLensService:
    store = VectorStore(
        settings.data_dir / "chroma", SentenceTransformerEmbedder(settings.embedding_model)
    )
    llm = OpenAICompatibleClient(settings) if settings.llm_enabled else None
    return RepoLensService(settings, store, llm)


def create_app(service: RepoLensService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202
        app.state.service = service or build_service(Settings())
        yield

    app = FastAPI(
        title="RepoLens AI",
        version=__version__,
        description="Index a public GitHub repo and ask questions with verifiable citations.",
        lifespan=lifespan,
    )

    def svc(request: Request) -> RepoLensService:
        return request.app.state.service

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return resources.files("repolens").joinpath("static/index.html").read_text("utf-8")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        s = svc(request)
        return HealthResponse(
            version=__version__,
            indexed_chunks=s.store.count(),
            embedding_model=s.store.embedder_name,
            llm_configured=s.llm is not None,
        )

    @app.get("/repos", response_model=list[RepoInfo], tags=["index"])
    def repos(request: Request) -> list[RepoInfo]:
        return [
            RepoInfo(repo=str(r["repo"]), commit=str(r["commit"]), chunks=int(r["chunks"]))
            for r in svc(request).store.repos()
        ]

    @app.post("/ingest", response_model=IngestResponse, tags=["index"])
    def ingest(body: IngestRequest, request: Request) -> IngestResponse:
        try:
            result = svc(request).ingest_url(body.repo_url)
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse(**asdict(result))

    @app.post("/query", response_model=QueryResponse, tags=["query"])
    def query(body: QueryRequest, request: Request) -> QueryResponse:
        result = svc(request).query(
            body.question, top_k=body.top_k, repo=body.repo, generate=body.generate
        )
        return QueryResponse(
            status=result.status,
            answer=result.answer,
            message=result.message,
            sources=[_source(i, h, i in result.cited) for i, h in enumerate(result.hits, 1)],
        )

    return app


def app_factory() -> FastAPI:
    return create_app()
