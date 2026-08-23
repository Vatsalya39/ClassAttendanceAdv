"""
backend/core/ratelimit.py
─────────────────────────
Fixed-window rate limiting on Redis.

Used on:
  - /auth/* and /register  (keyed by client IP)
  - /admin/session/capture (keyed per admin AND per section) so one runaway
    client cannot flood the capture queue.

Algorithm: INCR a key ``rl:{name}:{window_start}`` with TTL = window length.
Atomicity comes from Redis's single-threaded command execution; the small
race between INCR and EXPIRE is harmless (worst case the window lasts a few
ms longer). For strict atomicity use SET NX + INCR in a pipeline — we use
``client.pipeline()`` which is already atomic-enough for this purpose.
"""

from __future__ import annotations

import time
from typing import Any


class RateLimitExceededError(Exception):
    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        super().__init__(f"rate limit exceeded, retry in {self.retry_after_seconds}s")


async def check_rate_limit(
    redis: Any,
    *,
    name: str,
    limit: int,
    window_seconds: int,
) -> None:
    """
    Count one hit under ``name``; raise RateLimitExceededError if over limit.

    ``redis`` is any async Redis client (redis.asyncio.Redis in prod).
    """
    window = int(time.time()) // window_seconds
    key = f"rl:{name}:{window}"
    pipe = redis.pipeline(transaction=True)
    pipe.incr(key)
    pipe.ttl(key)
    pipe.expire(key, window_seconds)
    count, ttl, _ = await pipe.execute()

    count = int(count or 0)
    if count == 1 and int(ttl or -1) < 0:
        # EXPIRE applied within the same transaction; nothing to fix.
        pass
    if count > limit:
        remaining = int(ttl or 0)
        if remaining <= 0:
            remaining = window_seconds
        raise RateLimitExceededError(remaining)
