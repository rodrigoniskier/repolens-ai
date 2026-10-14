"""Path and content filters. Repository content is untrusted: it is only ever read as text."""

import re
from pathlib import PurePosixPath

ALLOWED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".md": "markdown",
    ".markdown": "markdown",
}

IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn", ".venv", "venv", "env", ".env.d", "node_modules", "__pycache__",
        "site-packages", "vendor", "third_party", "dist", "build", ".tox", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", ".idea", ".vscode", ".eggs",
    }
)  # fmt: skip

SECRET_NAME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\.env(\..*)?$",
        r"\.(pem|key|p12|pfx|jks|keystore|crt|cer)$",
        r"^id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$",
        r"^(credentials|secrets?)(\..*)?$",
        r"^\.(netrc|pgpass|npmrc|pypirc)$",
    )
)

SECRET_CONTENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{50,}\b",
        r"\bsk-[A-Za-z0-9_-]{32,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
        r"\bAIza[0-9A-Za-z_-]{35}\b",
    )
)


def language_for(path: str) -> str | None:
    return ALLOWED_EXTENSIONS.get(PurePosixPath(path).suffix.lower())


def has_ignored_dir(path: str) -> bool:
    return any(part in IGNORED_DIRS for part in PurePosixPath(path).parts[:-1])


def is_secret_name(path: str) -> bool:
    name = PurePosixPath(path).name
    return any(p.search(name) for p in SECRET_NAME_PATTERNS)


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_CONTENT_PATTERNS)


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]
