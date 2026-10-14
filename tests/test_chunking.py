import pytest

from repolens.chunking import chunk_file
from repolens.ingest.scanner import SourceFile


def _chunks(text: str, language: str = "python", max_lines: int = 10, overlap: int = 2):
    file = SourceFile(path="m.py" if language == "python" else "d.md", language=language, text=text)
    return chunk_file(file, repo="o/r", commit="a" * 40, max_lines=max_lines, overlap=overlap)


def test_line_ranges_match_source_exactly() -> None:
    text = "\n".join(f"line {i}" for i in range(1, 58))
    lines = text.splitlines()
    chunks = _chunks(text, "markdown")
    assert chunks
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line])
        assert c.commit == "a" * 40 and c.repo == "o/r"
    assert chunks[0].start_line == 1 and chunks[-1].end_line == len(lines)


def test_python_chunks_split_on_definitions() -> None:
    src = "import os\n\n\ndef a():\n    return 1\n\n\n@dec\nclass B:\n    x = 1\n"
    chunks = _chunks(src, max_lines=5, overlap=1)
    starts = [c.start_line for c in chunks]
    assert 4 in starts and 8 in starts  # decorator line is included in the class chunk


def test_markdown_headings_ignore_fenced_code() -> None:
    md = "# A\ntext\n```\n# not a heading\n```\n## B\nmore\n"
    chunks = _chunks(md, "markdown", max_lines=6, overlap=1)
    assert [c.start_line for c in chunks] == [1, 6]


def test_invalid_python_falls_back_to_windows() -> None:
    chunks = _chunks("def broken(:\n" * 30)
    assert len(chunks) > 1


def test_empty_and_blank_files() -> None:
    assert _chunks("") == []
    assert _chunks("\n\n\n") == []


def test_overlap_must_be_smaller() -> None:
    with pytest.raises(ValueError):
        _chunks("x = 1", max_lines=5, overlap=5)


def test_chunk_id_is_unique_per_range() -> None:
    chunks = _chunks("\n".join("x = 1" for _ in range(50)))
    assert len({c.id for c in chunks}) == len(chunks)
