"""Safe shallow clone of public GitHub repositories (git data only, never executed)."""

import os
import re
import shutil
import subprocess  # noqa: S404
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_NAME = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}"
_URL_RE = re.compile(rf"^https://github\.com/(?P<owner>{_NAME})/(?P<name>{_NAME})/?$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class IngestError(Exception):
    """Raised for invalid input or failures while fetching a repository."""


@dataclass(frozen=True, slots=True)
class RepoRef:
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


def parse_repo_url(url: str) -> RepoRef:
    """Accept only https://github.com/<owner>/<repo>; anything else is rejected."""
    match = _URL_RE.match(url.strip())
    name = match["name"].removesuffix(".git") if match else ""
    if not match or not name or set(name) == {"."}:
        raise IngestError("Only URLs like https://github.com/<owner>/<repo> are accepted.")
    return RepoRef(match["owner"], name)


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(dirpath) / f).lstat().st_size
            except OSError:
                continue
    return total


def _git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=_git_env(),
    )


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG", "SSL_CERT_FILE"}}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def head_commit(directory: Path) -> str | None:
    """Return the HEAD SHA of a local git checkout, or None if unavailable."""
    result = _git(["rev-parse", "HEAD"], cwd=directory)
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and _SHA_RE.match(sha) else None


@contextmanager
def clone_repository(
    ref: RepoRef, *, timeout_seconds: int, max_bytes: int
) -> Iterator[tuple[Path, str]]:
    """Shallow-clone into a temp dir (hooks and symlinks disabled); yield (path, commit sha)."""
    if shutil.which("git") is None:
        raise IngestError("git is not installed on the server.")
    tmp = Path(tempfile.mkdtemp(prefix="repolens-"))
    target = tmp / "repo"
    try:
        cmd = [
            "git",
            "-c", "core.hooksPath=/dev/null",
            "-c", "core.symlinks=false",
            "-c", "protocol.allow=never",
            "-c", "protocol.https.allow=always",
            "-c", "credential.helper=",
            "clone", "--depth=1", "--no-tags", "--single-branch", "--quiet",
            "--", ref.clone_url, str(target),
        ]  # fmt: skip
        proc = subprocess.Popen(  # noqa: S603
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=_git_env()
        )
        deadline = time.monotonic() + timeout_seconds
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                raise IngestError("Clone timed out.")
            if target.exists() and _dir_size(target) > max_bytes:
                proc.kill()
                proc.wait()
                raise IngestError("Repository exceeds the allowed clone size.")
            time.sleep(0.25)
        if proc.returncode != 0:
            raise IngestError("Could not clone the repository (is it public and does it exist?).")
        sha = head_commit(target)
        if sha is None:
            raise IngestError("Could not determine the commit SHA.")
        yield target, sha
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
