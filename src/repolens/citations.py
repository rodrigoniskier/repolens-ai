"""Citation helpers: GitHub permalinks and verification against a local checkout."""

from pathlib import Path
from urllib.parse import quote

from repolens.models import Chunk


def permalink(chunk: Chunk) -> str:
    path = quote(chunk.path, safe="/")
    return (
        f"https://github.com/{chunk.repo}/blob/{chunk.commit}/{path}"
        f"#L{chunk.start_line}-L{chunk.end_line}"
    )


def verify_citation(root: Path, chunk: Chunk, expected_commit: str | None = None) -> bool:
    """True if the cited file/lines exist under root and match the quoted text exactly."""
    root = root.resolve()
    target = (root / chunk.path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return False
    if expected_commit is not None and chunk.commit != expected_commit:
        return False
    lines = target.read_text(encoding="utf-8").splitlines()
    if not 1 <= chunk.start_line <= chunk.end_line <= len(lines):
        return False
    return "\n".join(lines[chunk.start_line - 1 : chunk.end_line]) == chunk.text
