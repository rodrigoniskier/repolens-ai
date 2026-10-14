"""Deterministic retrieval evaluation: hit@3, hit@5, latency and citation validity.

Ground truth is the hand-written eval/dataset.json; no generated answers are used.
Run: python -m repolens.evaluation [--embedder hashing|model]
"""

import argparse
import json
import shutil
import statistics
import subprocess  # noqa: S404
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from repolens.citations import verify_citation
from repolens.config import Settings
from repolens.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from repolens.ingest.github import head_commit
from repolens.service import RepoLensService
from repolens.store import VectorStore

DEFAULT_EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
FIXTURE_REPO = "acme/fixture"


def _commit_fixture(src: Path, dest: Path) -> str:
    """Copy the fixture into a throwaway git repo so the commit SHA is real."""
    shutil.copytree(src, dest)
    identity = ["-c", "user.name=eval", "-c", "user.email=eval@example.invalid"]
    for args in (["init", "-q"], ["add", "-A"], [*identity, "commit", "-q", "-m", "fixture"]):
        subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True)  # noqa: S603, S607
    sha = head_commit(dest)
    if sha is None:
        raise RuntimeError("could not create fixture commit")
    return sha


@contextmanager
def fixture_service(
    embedder: Embedder, eval_dir: Path = DEFAULT_EVAL_DIR, **settings_overrides
) -> Iterator[tuple[RepoLensService, Path, str]]:
    """Index the fixture repo into a throwaway store; yields (service, checkout root, commit)."""
    with tempfile.TemporaryDirectory(prefix="repolens-eval-") as tmp:
        tmp_path = Path(tmp)
        root = tmp_path / "fixture"
        sha = _commit_fixture(eval_dir / "fixture_repo", root)
        settings = Settings(data_dir=tmp_path / "data", _env_file=None, **settings_overrides)  # type: ignore[call-arg]
        service = RepoLensService(settings, VectorStore(tmp_path / "chroma", embedder))
        service.ingest_directory(root, repo=FIXTURE_REPO, commit=sha)
        yield service, root, sha


def load_questions(eval_dir: Path = DEFAULT_EVAL_DIR) -> list[dict]:
    return json.loads((eval_dir / "dataset.json").read_text("utf-8"))["questions"]


def run_evaluation(embedder: Embedder, eval_dir: Path = DEFAULT_EVAL_DIR, k: int = 3) -> dict:
    """Retrieval metrics. hit@k only counts answerable questions (those with expected paths)."""
    questions = load_questions(eval_dir)
    depth = max(k, 5)
    with fixture_service(embedder, eval_dir) as (service, root, sha):
        rows = []
        for q in questions:
            start = time.perf_counter()
            hits = service.store.search(q["question"], depth, FIXTURE_REPO)
            latency_ms = (time.perf_counter() - start) * 1000
            expected = set(q["expected_paths"])
            paths = [h.chunk.path for h in hits]
            rows.append(
                {
                    "id": q["id"],
                    "hit": any(p in expected for p in paths[:k]) if expected else None,
                    "hit@5": any(p in expected for p in paths[:5]) if expected else None,
                    "latency_ms": round(latency_ms, 2),
                    "citations": len(hits),
                    "valid_citations": sum(verify_citation(root, h.chunk, sha) for h in hits),
                    "top_paths": paths[:k],
                }
            )

    scored = [r for r in rows if r["hit"] is not None]
    total_citations = sum(r["citations"] for r in rows)
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "embedder": embedder.name,
        "questions": len(rows),
        "answerable_questions": len(scored),
        f"hit@{k}": round(sum(r["hit"] for r in scored) / len(scored), 4),
        "hit@5": round(sum(r["hit@5"] for r in scored) / len(scored), 4),
        "citation_validity": round(sum(r["valid_citations"] for r in rows) / total_citations, 4)
        if total_citations
        else 0.0,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "max": latencies[-1],
        },
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=["hashing", "model"], default="model")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args()

    embedder: Embedder = (
        HashingEmbedder()
        if args.embedder == "hashing"
        else SentenceTransformerEmbedder(Settings(_env_file=None).embedding_model)  # type: ignore[call-arg]
    )
    report = run_evaluation(embedder, args.eval_dir)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
