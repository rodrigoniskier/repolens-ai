"""Walks a checked-out directory and yields indexable text files within strict limits."""

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from repolens.ingest import filters


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str  # POSIX path relative to the repository root
    language: str
    text: str


@dataclass(slots=True)
class ScanReport:
    files: list[SourceFile] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    total_bytes: int = 0
    truncated: bool = False


def scan_directory(
    root: Path, *, max_files: int, max_file_bytes: int, max_total_bytes: int
) -> ScanReport:
    """Collect .py/.md files. Symlinks are never followed and nothing is executed."""
    root = root.resolve()
    report = ScanReport()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept = []
        for d in sorted(dirnames):
            if d in filters.IGNORED_DIRS:
                report.skipped["ignored_dir"] += 1
            elif (Path(dirpath) / d).is_symlink():
                report.skipped["symlink"] += 1
            else:
                kept.append(d)
        dirnames[:] = kept

        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            language = filters.language_for(rel)
            if language is None:
                report.skipped["unsupported_type"] += 1
                continue
            if full.is_symlink() or not full.is_file():
                report.skipped["symlink"] += 1
                continue
            if filters.is_secret_name(rel):
                report.skipped["secret_name"] += 1
                continue
            size = full.stat().st_size
            if size > max_file_bytes:
                report.skipped["too_large"] += 1
                continue
            if len(report.files) >= max_files or report.total_bytes + size > max_total_bytes:
                report.truncated = True
                report.skipped["limit_reached"] += 1
                continue

            data = full.read_bytes()
            if filters.is_binary(data):
                report.skipped["binary"] += 1
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                report.skipped["not_utf8"] += 1
                continue
            if filters.contains_secret(text):
                report.skipped["secret_content"] += 1
                continue

            report.files.append(SourceFile(path=rel, language=language, text=text))
            report.total_bytes += size

    return report
