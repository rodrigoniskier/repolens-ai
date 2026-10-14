from pathlib import Path

import pytest

from repolens.ingest import filters
from repolens.ingest.github import (
    IngestError,
    RepoRef,
    _git_env,
    clone_repository,
    parse_repo_url,
)
from repolens.ingest.scanner import scan_directory


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a/b",
        "https://gitlab.com/a/b",
        "https://github.com/a",
        "https://github.com/a/b/c",
        "https://github.com/a/..",
        "file:///etc/passwd",
        "git@github.com:a/b.git",
        "https://github.com/a/b?x=1",
        "https://evil.com/https://github.com/a/b",
        "--upload-pack=x",
    ],
)
def test_parse_repo_url_rejects(url: str) -> None:
    with pytest.raises(IngestError):
        parse_repo_url(url)


def test_parse_repo_url_accepts() -> None:
    ref = parse_repo_url("https://github.com/pallets/click.git")
    assert ref.full_name == "pallets/click"
    assert ref.clone_url == "https://github.com/pallets/click.git"


def test_git_environment_disables_global_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "untrusted-gitconfig")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "filter.exploit.clean")
    env = _git_env()
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "GIT_CONFIG_KEY_0" not in env


def test_clone_rechecks_size_after_git_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FinishedClone:
        returncode = 0

        def __init__(self, cmd, **_kwargs) -> None:
            target = Path(cmd[-1])
            target.mkdir()
            (target / "oversized.py").write_bytes(b"over limit")

        def poll(self) -> int:
            return 0

    monkeypatch.setattr("repolens.ingest.github.subprocess.Popen", FinishedClone)
    with pytest.raises(IngestError, match="clone size"):
        with clone_repository(RepoRef("owner", "repo"), timeout_seconds=5, max_bytes=1):
            pytest.fail("oversized clone should not be yielded")


def test_filters() -> None:
    assert filters.language_for("a/b.py") == "python"
    assert filters.language_for("README.md") == "markdown"
    assert filters.language_for("x.png") is None
    assert filters.has_ignored_dir("node_modules/x/y.py")
    assert not filters.has_ignored_dir("src/y.py")
    assert filters.is_secret_name(".env") and filters.is_secret_name("deploy/.env.prod")
    assert filters.is_secret_name("server.pem")
    assert not filters.is_secret_name("environment.py")
    assert filters.contains_secret("k = 'AKIAABCDEFGHIJKLMNOP'")
    assert filters.contains_secret("-----BEGIN RSA PRIVATE KEY-----")
    assert not filters.contains_secret("print('hello')")
    assert filters.is_binary(b"abc\x00def")


def _scan(root: Path, **kw: int):
    args = {"max_files": 100, "max_file_bytes": 10_000, "max_total_bytes": 100_000}
    return scan_directory(root, **{**args, **kw})


def test_scanner_skips_unsafe_content(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "doc.md").write_text("# T\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("x=1")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("x=1")
    (tmp_path / "cfg.py").write_text("T = 'ghp_" + "a" * 36 + "'\n")
    (tmp_path / "credentials.py").write_text("A = 1")
    (tmp_path / "bin.py").write_bytes(b"\x00\x01")
    (tmp_path / "latin.py").write_bytes("é = 1".encode("latin-1"))
    (tmp_path / "big.py").write_text("x = 1\n" * 5000)
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("leak = 1\n")
    (tmp_path / "link.py").symlink_to(outside)

    report = _scan(tmp_path)

    assert sorted(f.path for f in report.files) == ["doc.md", "ok.py"]
    for reason in ("ignored_dir", "secret_content", "secret_name", "binary", "not_utf8"):
        assert report.skipped[reason] >= 1, reason
    assert report.skipped["too_large"] == 1
    assert report.skipped["symlink"] == 1


def test_scanner_enforces_limits(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x = 1\n")
    report = _scan(tmp_path, max_files=2)
    assert len(report.files) == 2 and report.truncated
    report = _scan(tmp_path, max_total_bytes=12)
    assert len(report.files) == 2 and report.truncated
