"""Retry helpers."""

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_with_backoff(
    operation: Callable[[], T], attempts: int = 5, base_delay: float = 0.5
) -> T:
    """Call a flaky operation again after failures, doubling the wait each time with jitter."""
    for attempt in range(attempts):
        try:
            return operation()
        except OSError:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay + random.uniform(0, delay / 10))  # noqa: S311
    raise RuntimeError("unreachable")
