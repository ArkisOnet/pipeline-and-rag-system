"""
Token-bucket rate limiter for async code.
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, requests_per_second: float = 1.0) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self._interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._last_release: float = 0.0

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last_release + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_release = time.monotonic()
