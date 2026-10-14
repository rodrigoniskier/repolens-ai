from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from repolens.embeddings import HashingEmbedder, collection_name_for
from repolens.models import Chunk
from repolens.store import VectorStore


def test_legacy_index_migrates_and_active_generations_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    embedder = HashingEmbedder()
    client = chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=collection_name_for(embedder),
        metadata={"hnsw:space": "cosine", "embedder": embedder.name},
    )
    chunks = [
        Chunk(
            repo="acme/one",
            commit="a" * 40,
            path="deploy.md",
            start_line=1,
            end_line=1,
            language="markdown",
            text="rollback release deployment",
        ),
        Chunk(
            repo="acme/two",
            commit="b" * 40,
            path="cache.py",
            start_line=1,
            end_line=1,
            language="python",
            text="cache eviction",
        ),
    ]
    collection.add(
        ids=[chunk.id for chunk in chunks],
        embeddings=embedder.embed([VectorStore._doc(chunk) for chunk in chunks]),
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "repo": chunk.repo,
                "commit": chunk.commit,
                "path": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language,
            }
            for chunk in chunks
        ],
    )

    store = VectorStore(path, embedder)
    assert store.count() == 2
    store.replace_repo("acme/one", [replace(chunks[0], commit="c" * 40)])

    reopened = VectorStore(path, HashingEmbedder())
    assert reopened.count() == 2
    assert {entry["repo"] for entry in reopened.repos()} == {"acme/one", "acme/two"}
    assert reopened.search("rollback release", 2, repo="acme/one")[0].chunk.commit == "c" * 40
    assert reopened.search("cache eviction", 2, repo="acme/two")[0].chunk.commit == "b" * 40


def test_first_legacy_reindex_hides_staged_generation_from_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "chroma"
    embedder = HashingEmbedder()
    client = chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=collection_name_for(embedder),
        metadata={
            "hnsw:space": "cosine",
            "embedder": embedder.name,
            "repolens_active_generations": '{"acme/legacy":null}',
        },
    )
    legacy = Chunk(
        repo="acme/legacy",
        commit="a" * 40,
        path="README.md",
        start_line=1,
        end_line=1,
        language="markdown",
        text="legacy indexed content",
    )
    replacement = replace(legacy, commit="b" * 40, text="replacement staged content")
    orphaned_staging = replace(legacy, commit="c" * 40, text="orphaned staging content")
    collection.add(
        ids=[legacy.id, f"abandoned:{orphaned_staging.id}"],
        embeddings=embedder.embed([VectorStore._doc(legacy), VectorStore._doc(orphaned_staging)]),
        documents=[legacy.text, orphaned_staging.text],
        metadatas=[
            {
                "repo": legacy.repo,
                "commit": legacy.commit,
                "path": legacy.path,
                "start_line": legacy.start_line,
                "end_line": legacy.end_line,
                "language": legacy.language,
            },
            {
                "repo": orphaned_staging.repo,
                "commit": orphaned_staging.commit,
                "path": orphaned_staging.path,
                "start_line": orphaned_staging.start_line,
                "end_line": orphaned_staging.end_line,
                "language": orphaned_staging.language,
                "generation": "abandoned-generation",
            },
        ],
    )

    store = VectorStore(path, embedder)
    active_generation = store._active_generations["acme/legacy"]
    assert active_generation is not None
    assert store.search("orphaned staging content", 10)[0].chunk.commit == legacy.commit
    store = VectorStore(path, embedder)
    assert store._active_generations["acme/legacy"] == active_generation
    assert store.search("orphaned staging content", 10)[0].chunk.commit == legacy.commit
    original_add = store._col.add
    staged = Event()
    continue_ingest = Event()

    def add_then_pause(*args, **kwargs) -> None:
        original_add(*args, **kwargs)
        staged.set()
        if not continue_ingest.wait(timeout=5):
            raise TimeoutError("test did not release staged generation")

    monkeypatch.setattr(store._col, "add", add_then_pause)
    errors: list[BaseException] = []

    def reindex() -> None:
        try:
            store.replace_repo("acme/legacy", [replacement])
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=reindex)
    thread.start()
    try:
        assert staged.wait(timeout=5)
        hits = store.search("replacement staged content", 10)
        assert hits
        assert {hit.chunk.commit for hit in hits} == {legacy.commit}
        assert all(hit.chunk.text == legacy.text for hit in hits)
    finally:
        continue_ingest.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
