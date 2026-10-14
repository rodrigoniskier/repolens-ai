"""Structure-aware chunking that keeps exact 1-based line ranges."""

import ast
import re

from repolens.ingest.scanner import SourceFile
from repolens.models import Chunk

_HEADING = re.compile(r"^#{1,6}\s+\S")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _python_starts(text: str) -> set[int]:
    """0-based line indexes where a top-level def/class (including decorators) begins."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return set()
    starts: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            starts.add(first - 1)
    return starts


def _markdown_starts(lines: list[str]) -> set[int]:
    starts: set[int] = set()
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and _HEADING.match(line):
            starts.add(i)
    return starts


def _segments(lines: list[str], starts: set[int]) -> list[tuple[int, int]]:
    """Half-open [start, end) segments split at the given boundaries."""
    cuts = sorted({0, *starts, len(lines)})
    return [(a, b) for a, b in zip(cuts, cuts[1:], strict=False) if b > a]


def chunk_file(
    file: SourceFile, *, repo: str, commit: str, max_lines: int, overlap: int
) -> list[Chunk]:
    if overlap >= max_lines:
        raise ValueError("overlap must be smaller than max_lines")
    lines = file.text.splitlines()
    if not lines:
        return []
    starts = _python_starts(file.text) if file.language == "python" else _markdown_starts(lines)

    # Merge small adjacent segments up to max_lines; window oversized ones with overlap.
    ranges: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for a, b in _segments(lines, starts):
        if b - a > max_lines:
            if cur:
                ranges.append(cur)
                cur = None
            step = max_lines - overlap
            pos = a
            while pos < b:
                end = min(pos + max_lines, b)
                ranges.append((pos, end))
                if end == b:
                    break
                pos += step
        elif cur and (b - cur[0]) <= max_lines:
            cur = (cur[0], b)
        else:
            if cur:
                ranges.append(cur)
            cur = (a, b)
    if cur:
        ranges.append(cur)

    chunks = []
    for a, b in ranges:
        text = "\n".join(lines[a:b])
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                repo=repo,
                commit=commit,
                path=file.path,
                start_line=a + 1,
                end_line=b,
                language=file.language,
                text=text,
            )
        )
    return chunks
