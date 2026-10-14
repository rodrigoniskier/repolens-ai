"""Persistent ChromaDB vector index (cosine similarity, embeddings supplied by us)."""

import json
import threading
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from repolens.embeddings import Embedder, collection_name_for
from repolens.models import Chunk, Hit

_ACTIVE_GENERATIONS = "repolens_active_generations"


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = client.get_or_create_collection(
            name=collection_name_for(embedder),
            metadata={"hnsw:space": "cosine", "embedder": embedder.name},
        )
        self._lock = threading.RLock()
        self._active_generations = self._load_active_generations()

    @staticmethod
    def _doc(chunk: Chunk) -> str:
        return f"{chunk.path}\n{chunk.text}"

    def _load_active_generations(self) -> dict[str, str]:
        metadata = self._col.metadata or {}
        encoded = metadata.get(_ACTIVE_GENERATIONS)
        if encoded is not None:
            active = json.loads(str(encoded))
            if not isinstance(active, dict) or any(
                not isinstance(repo, str)
                or (generation is not None and not isinstance(generation, str))
                for repo, generation in active.items()
            ):
                raise ValueError("Invalid active-generation metadata in the vector index.")
        else:
            records = self._col.get(include=["metadatas"])
            active = {str(meta["repo"]): None for meta in records["metadatas"] or []}
        active = self._migrate_legacy_generations(active)
        self._save_active_generations(active)
        return active

    def _migrate_legacy_generations(self, active: dict[str, str | None]) -> dict[str, str]:
        migrated: dict[str, str] = {}
        for repo, generation in active.items():
            if generation is not None:
                migrated[repo] = generation
                continue
            records = self._col.get(where={"repo": {"$eq": repo}}, include=["metadatas"])
            legacy_records = [
                (record_id, metadata)
                for record_id, metadata in zip(
                    records["ids"], records["metadatas"] or [], strict=True
                )
                if "generation" not in metadata
            ]
            if not legacy_records:
                continue
            legacy_generation = uuid.uuid4().hex
            self._col.update(
                ids=[record_id for record_id, _metadata in legacy_records],
                metadatas=[
                    {**metadata, "generation": legacy_generation}
                    for _record_id, metadata in legacy_records
                ],
            )
            migrated[repo] = legacy_generation
        return migrated

    def _save_active_generations(self, active: dict[str, str]) -> None:
        metadata = dict(self._col.metadata or {})
        metadata.pop("hnsw:space", None)
        metadata[_ACTIVE_GENERATIONS] = json.dumps(active, sort_keys=True, separators=(",", ":"))
        self._col.modify(metadata=metadata)

    @staticmethod
    def _generation_filter(repo: str, generation: str) -> dict[str, Any]:
        return {
            "$and": [
                {"repo": {"$eq": repo}},
                {"generation": {"$eq": generation}},
            ]
        }

    def _active_filter(self, repos: dict[str, str]) -> dict[str, Any]:
        conditions = [
            self._generation_filter(repo, generation) for repo, generation in repos.items()
        ]
        if len(conditions) == 1:
            return conditions[0]
        return {"$or": conditions}

    def replace_repo(self, repo: str, chunks: list[Chunk]) -> None:
        """Stage a complete generation, then atomically make it visible to readers."""
        generation = uuid.uuid4().hex
        old_records = self._col.get(where={"repo": {"$eq": repo}}, include=["metadatas"])
        old_ids = old_records["ids"]

        try:
            batch = 64
            for i in range(0, len(chunks), batch):
                part = chunks[i : i + batch]
                self._col.add(
                    ids=[f"{generation}:{chunk.id}" for chunk in part],
                    embeddings=self._embedder.embed([self._doc(chunk) for chunk in part]),
                    documents=[chunk.text for chunk in part],
                    metadatas=[
                        {
                            "repo": chunk.repo,
                            "commit": chunk.commit,
                            "path": chunk.path,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "language": chunk.language,
                            "generation": generation,
                        }
                        for chunk in part
                    ],
                )
        except Exception as exc:
            try:
                self._col.delete(
                    where={
                        "$and": [
                            {"repo": {"$eq": repo}},
                            {"generation": {"$eq": generation}},
                        ]
                    }
                )
            except Exception as cleanup_error:
                exc.add_note(
                    f"Could not remove incomplete staged index generation: {cleanup_error!r}"
                )
            raise

        with self._lock:
            active = dict(self._active_generations)
            active[repo] = generation
            self._save_active_generations(active)
            self._active_generations = active
            if old_ids:
                self._col.delete(ids=old_ids)

    def search(self, query: str, k: int, repo: str | None = None) -> list[Hit]:
        with self._lock:
            if not self._active_generations or (
                repo is not None and repo not in self._active_generations
            ):
                return []
            collection_count = self._col.count()
            if collection_count == 0:
                return []
            active = (
                {repo: self._active_generations[repo]}
                if repo is not None
                else dict(self._active_generations)
            )
            res = self._col.query(
                query_embeddings=self._embedder.embed([query]),
                n_results=min(k, collection_count),
                where=self._active_filter(active),
                include=["documents", "metadatas", "distances"],
            )
            hits = []
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            ):
                chunk = Chunk(
                    repo=str(meta["repo"]),
                    commit=str(meta["commit"]),
                    path=str(meta["path"]),
                    start_line=int(meta["start_line"]),
                    end_line=int(meta["end_line"]),
                    language=str(meta["language"]),
                    text=doc,
                )
                hits.append(Hit(chunk=chunk, score=1.0 - float(dist)))
            return hits

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    def count(self) -> int:
        with self._lock:
            if not self._active_generations:
                return 0
            records = self._col.get(
                where=self._active_filter(self._active_generations), include=["metadatas"]
            )
            return len(records["ids"])

    def repos(self) -> list[dict[str, str | int]]:
        with self._lock:
            if not self._active_generations:
                return []
            data = self._col.get(
                where=self._active_filter(self._active_generations), include=["metadatas"]
            )
            summary: dict[str, dict[str, str | int]] = {}
            for meta in data["metadatas"] or []:
                entry = summary.setdefault(
                    str(meta["repo"]),
                    {"repo": str(meta["repo"]), "commit": str(meta["commit"]), "chunks": 0},
                )
                entry["chunks"] = int(entry["chunks"]) + 1
            return sorted(summary.values(), key=lambda entry: str(entry["repo"]))
