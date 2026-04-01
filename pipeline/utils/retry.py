"""
Async retry decorator with exponential backoff and jitter.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class RetryExhausted(Exception):
    pass


def async_retry(
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    reraise_on_exhaust: bool = True,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    cap = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    sleep_for = random.uniform(0, cap)
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs",
                        attempt, max_attempts, func.__qualname__, exc, sleep_for,
                    )
                    await asyncio.sleep(sleep_for)

            if reraise_on_exhaust and last_exc is not None:
                raise last_exc
            raise RetryExhausted(f"{func.__qualname__} failed after {max_attempts} attempts")

        return wrapper  # type: ignore[return-value]

    return decorator
