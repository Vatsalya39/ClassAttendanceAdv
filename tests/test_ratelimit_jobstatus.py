"""
tests/test_ratelimit_jobstatus.py
─────────────────────────────────
Phase 3 unit tests for the Redis-backed rate limiter and job-status store,
using a tiny in-memory fake (hermetic — no live Redis required).
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.core.jobstatus import JobStatus, claim, get_status, release_claim, set_status
from backend.core.ratelimit import RateLimitExceededError, check_rate_limit


class FakeRedis:
    """Just enough of redis.asyncio for the limiter + job status code."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.ttls: dict[str, int] = {}

    async def set(self, key: str, value: Any, nx: bool = False, ex: int | None = None) -> bool:
        if nx and key in self.store:
            return False
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def get(self, key: str) -> Any:
        return self.store.get(key)

    async def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._ops: list[tuple[str, tuple[Any, ...]]] = []

    def incr(self, key: str) -> FakePipeline:
        self._ops.append(("incr", (key,)))
        return self

    def ttl(self, key: str) -> FakePipeline:
        self._ops.append(("ttl", (key,)))
        return self

    def expire(self, key: str, seconds: int) -> FakePipeline:
        self._ops.append(("expire", (key, seconds)))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for op, args in self._ops:
            if op == "incr":
                key = args[0]
                current = int(self._redis.store.get(key, 0)) + 1
                self._redis.store[key] = current
                results.append(current)
            elif op == "ttl":
                results.append(self._redis.ttls.get(args[0], -1))
            elif op == "expire":
                self._redis.ttls[args[0]] = args[1]
                results.append(True)
        self._ops.clear()
        return results


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


async def test_rate_limit_allows_under_limit(fake_redis: FakeRedis) -> None:
    for _ in range(5):
        await check_rate_limit(fake_redis, name="auth", limit=5, window_seconds=60)


async def test_rate_limit_blocks_over_limit(fake_redis: FakeRedis) -> None:
    for _ in range(3):
        await check_rate_limit(fake_redis, name="capture", limit=3, window_seconds=60)
    with pytest.raises(RateLimitExceededError):
        await check_rate_limit(fake_redis, name="capture", limit=3, window_seconds=60)


async def test_rate_limit_is_namespaced(fake_redis: FakeRedis) -> None:
    await check_rate_limit(fake_redis, name="a", limit=1, window_seconds=60)
    # Different namespace unaffected.
    await check_rate_limit(fake_redis, name="b", limit=1, window_seconds=60)


async def test_job_status_lifecycle(fake_redis: FakeRedis) -> None:
    assert await get_status(fake_redis, "j1") is None
    await set_status(fake_redis, "j1", JobStatus.queued)
    assert (await get_status(fake_redis, "j1"))["status"] == "queued"
    await set_status(fake_redis, "j1", JobStatus.processing)
    assert (await get_status(fake_redis, "j1"))["status"] == "processing"

    result = {
        "matched": [{"student_id": "s", "distance": 0.2}],
        "review_needed": [],
    }
    await set_status(fake_redis, "j1", JobStatus.done, result=result)
    final = await get_status(fake_redis, "j1")
    assert final is not None
    assert final["status"] == "done"
    assert final["result"]["matched"][0]["student_id"] == "s"


async def test_job_failure_payload(fake_redis: FakeRedis) -> None:
    await set_status(fake_redis, "j2", JobStatus.failed, error="boom")
    data = await get_status(fake_redis, "j2")
    assert data is not None
    assert data["status"] == "failed"
    assert data["error"] == "boom"


async def test_claim_is_exclusive_and_releasable(fake_redis: FakeRedis) -> None:
    assert await claim(fake_redis, "job-x") is True  # first claim wins
    assert await claim(fake_redis, "job-x") is False  # duplicate delivery skipped
    await release_claim(fake_redis, "job-x")
    assert await claim(fake_redis, "job-x") is True  # retry after failure allowed
