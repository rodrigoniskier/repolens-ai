from dataclasses import replace
from pathlib import Path

import chromadb
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
