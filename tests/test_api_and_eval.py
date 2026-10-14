import json

import httpx
import pytest
from fastapi.testclient import TestClient

from repolens.api import create_app
from repolens.config import Settings
from repolens.embeddings import HashingEmbedder
from repolens.evaluation import DEFAULT_EVAL_DIR, run_evaluation
from repolens.llm import LLMError, OpenAICompatibleClient
from repolens.service import RepoLensService

URL = "https://github.com/acme/fixture"


@pytest.fixture
def client(service: RepoLensService):
    with TestClient(create_app(service)) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["indexed_chunks"] == 0
    assert body["llm_configured"] is False


def test_ingest_then_query_flow(client: TestClient) -> None:
    r = client.post("/ingest", json={"repo_url": URL})
    assert r.status_code == 200, r.text
    ing = r.json()
    assert ing["chunks_indexed"] > 0 and len(ing["commit"]) == 40

    assert client.get("/repos").json()[0]["repo"] == "acme/fixture"

    r = client.post("/query", json={"question": "How do I roll back a release?", "top_k": 3})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "retrieval_only"
    assert body["answer"] is None
    top = body["sources"][0]
    assert top["path"] == "docs/deploy.md"
    assert top["commit"] == ing["commit"]
    assert top["url"].endswith(f"#L{top['start_line']}-L{top['end_line']}")


def test_query_without_evidence(client: TestClient) -> None:
    body = client.post("/query", json={"question": "zzzqqq xylophone banana"}).json()
    assert body["status"] == "insufficient_evidence"


def test_validation_and_errors(client: TestClient) -> None:
    assert client.post("/ingest", json={"repo_url": "https://example.com/a/b"}).status_code == 422
    assert client.post("/query", json={"question": "x"}).status_code == 422
    assert client.post("/query", json={"question": "valid?", "top_k": 99}).status_code == 422


def test_openapi_and_ui(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ingest", "/query"} <= set(paths)
    page = client.get("/")
    assert page.status_code == 200 and "RepoLens AI" in page.text
    assert '<label for="repo-url" class="visually-hidden">' in page.text
    assert '<label for="question" class="visually-hidden">' in page.text
    assert '<label for="repo-filter" class="visually-hidden">' in page.text
    assert 'for="repo-url" hidden' not in page.text


def test_openai_compatible_client(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_model": "m", "llm_api_key": "k"})
    seen: dict = {}

    def fake_post(url: str, json: dict, headers: dict, timeout: float) -> httpx.Response:
        seen.update(url=url, headers=headers, payload=json)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": " hi [1] "}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    assert OpenAICompatibleClient(settings).complete("q", []) == "hi [1]"
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"]["Authorization"] == "Bearer k"
    assert seen["payload"]["temperature"] == 0

    def boom(*_a, **_k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(LLMError):
        OpenAICompatibleClient(settings).complete("q", [])


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": None},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_openai_client_rejects_incomplete_json(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, response: dict
) -> None:
    settings = settings.model_copy(update={"llm_model": "m"})
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *_args, **_kwargs: httpx.Response(
            200,
            json=response,
            request=httpx.Request("POST", "https://example.invalid"),
        ),
    )
    with pytest.raises(LLMError, match="LLM request failed"):
        OpenAICompatibleClient(settings).complete("q", [])


def test_llm_disabled_without_model(settings: Settings) -> None:
    assert not settings.llm_enabled
    assert settings.model_copy(update={"llm_model": "x"}).llm_enabled


def test_settings_validate_chunk_overlap(settings: Settings) -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        Settings(chunk_lines=5, chunk_overlap=5, _env_file=None)
    valid = Settings(chunk_lines=5, chunk_overlap=4, _env_file=None)
    assert valid.chunk_overlap < valid.chunk_lines


def test_evaluation_report_is_deterministic_and_real() -> None:
    dataset = json.loads((DEFAULT_EVAL_DIR / "dataset.json").read_text("utf-8"))
    report = run_evaluation(HashingEmbedder())
    assert report["questions"] == len(dataset["questions"])
    assert 0.0 <= report["hit@3"] <= 1.0
    assert report["citation_validity"] == 1.0
    again = run_evaluation(HashingEmbedder())
    assert [r["hit"] for r in report["per_question"]] == [r["hit"] for r in again["per_question"]]
    assert report["hit@3"] == again["hit@3"]
