"""Loads the real sentence-transformers model; skipped by default (needs network/cache)."""

import pytest

from repolens.embeddings import SentenceTransformerEmbedder
from repolens.evaluation import run_evaluation

pytestmark = pytest.mark.slow


def test_real_model_meets_baseline() -> None:
    report = run_evaluation(SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"))
    assert report["hit@3"] >= 0.75
    assert report["citation_validity"] == 1.0
