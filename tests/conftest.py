import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from repolens.config import Settings
from repolens.embeddings import HashingEmbedder
from repolens.ingest.github import RepoRef, head_commit
from repolens.service import RepoLensService
from repolens.store import VectorStore

FIXTURE = Path(__file__).resolve().parents[1] / "eval" / "fixture_repo"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", min_score=0.1, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", HashingEmbedder())


@pytest.fixture
def git_fixture(tmp_path: Path) -> tuple[Path, str]:
    """A local git repo built from eval/fixture_repo (no network)."""
    root = tmp_path / "fixture"
    root.mkdir()
    for src in FIXTURE.rglob("*"):
        if src.is_file():
            dest = root / src.relative_to(FIXTURE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
    ident = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    for args in (["init", "-q"], ["add", "-A"], [*ident, "commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    sha = head_commit(root)
    assert sha
    return root, sha


@pytest.fixture
def service(settings: Settings, store: VectorStore, git_fixture: tuple[Path, str]):
    root, sha = git_fixture

    @contextmanager
    def local_cloner(_ref: RepoRef) -> Iterator[tuple[Path, str]]:
        yield root, sha

    return RepoLensService(settings, store, cloner=local_cloner)
