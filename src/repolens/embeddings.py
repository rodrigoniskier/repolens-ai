"""Local embedding backends. No network calls to paid APIs."""

import hashlib
import math
import re
import threading
import zlib
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """sentence-transformers model, loaded lazily (first use may download the model)."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # noqa: ANN202
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_STOP = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the this to what "
    "where which with".split()
)


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


class HashingEmbedder:
    """Deterministic lexical embedder (hashed bag of words). For tests and offline evals only."""

    def __init__(self, dim: int = 8192) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for word in _TOKEN.findall(text):
            out.extend(p.lower() for p in _CAMEL.findall(word))
        return [_stem(t) for t in out if t not in _STOP and len(t) > 1]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self._tokens(text):
                h = zlib.crc32(tok.encode())
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


def collection_name_for(embedder: Embedder) -> str:
    """Different embedders produce incompatible vectors, so each gets its own collection."""
    digest = hashlib.sha1(embedder.name.encode(), usedforsecurity=False).hexdigest()[:12]
    return f"repolens-{digest}"
