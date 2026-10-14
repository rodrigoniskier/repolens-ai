"""Persistent ChromaDB vector index (cosine similarity, embeddings supplied by us)."""

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from repolens.embeddings import Embedder, collection_name_for
from repolens.models import Chunk, Hit


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

    @staticmethod
    def _doc(chunk: Chunk) -> str:
        return f"{chunk.path}\n{chunk.text}"

    def replace_repo(self, repo: str, chunks: list[Chunk]) -> None:
        """Atomically-enough swap: drop previous entries for the repo, then add the new ones."""
        self._col.delete(where={"repo": repo})
        batch = 64
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self._col.add(
                ids=[c.id for c in part],
                embeddings=self._embedder.embed([self._doc(c) for c in part]),  # type: ignore[arg-type]
                documents=[c.text for c in part],
                metadatas=[
                    {
                        "repo": c.repo,
                        "commit": c.commit,
                        "path": c.path,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "language": c.language,
                    }
                    for c in part
                ],
            )

    def search(self, query: str, k: int, repo: str | None = None) -> list[Hit]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=self._embedder.embed([query]),  # type: ignore[arg-type]
            n_results=min(k, self._col.count()),
            where={"repo": repo} if repo else None,
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
            hits.append(Hit(chunk=chunk, score=round(1.0 - float(dist), 4)))
        return hits

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    def count(self) -> int:
        return self._col.count()

    def repos(self) -> list[dict[str, str | int]]:
        data = self._col.get(include=["metadatas"])
        summary: dict[str, dict[str, str | int]] = {}
        for meta in data["metadatas"] or []:
            entry = summary.setdefault(
                str(meta["repo"]),
                {"repo": str(meta["repo"]), "commit": str(meta["commit"]), "chunks": 0},
            )
            entry["chunks"] = int(entry["chunks"]) + 1
        return sorted(summary.values(), key=lambda e: str(e["repo"]))
