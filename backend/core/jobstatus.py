"""
backend/core/jobstatus.py
─────────────────────────
Capture-job status store backed by Redis.

Lifecycle:  queued → processing → done | failed

The API writes "queued" at enqueue time and reads status when the admin
polls GET /admin/session/capture/{job_id}; the worker flips it to
"processing" / "done" / "failed". Entries auto-expire (TTL) so Redis stays
bounded.

The claim() helper also gives the worker idempotency: only one worker can
claim a job_id, and a re-delivered job whose result was already stored is
skipped.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def _done_marker(job_id: str) -> str:
    return f"job:{job_id}:done"


async def set_status(
    redis: Any,
    job_id: str,
    status: JobStatus,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    ttl_seconds: int = 3600,
) -> None:
    payload: dict[str, Any] = {"status": status.value}
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    await redis.set(_key(job_id), json.dumps(payload), ex=ttl_seconds)


async def get_status(redis: Any, job_id: str) -> dict[str, Any] | None:
    """Returns {'status': ..., 'result'?: ..., 'error'?: ...} or None."""
    raw = await redis.get(_key(job_id))
    if raw is None:
        return None
    decoded: dict[str, Any] = json.loads(raw)
    return decoded


async def claim(redis: Any, job_id: str, ttl_seconds: int = 3600) -> bool:
    """
    Try to claim exclusive processing rights for this job.

    Returns False when the job has already been processed to completion —
    that is the idempotency guard for redelivered messages.
    """
    return bool(await redis.set(_done_marker(job_id), "1", nx=True, ex=ttl_seconds))


async def set_job_scope(redis: Any, job_id: str, section_id: str, ttl_seconds: int = 3600) -> None:
    """
    Record which section a job belongs to so GET /admin/session/capture/{id}
    can verify the polling admin's scope (tenant isolation on job status).
    """
    await redis.set(f"job:{job_id}:section", section_id, ex=ttl_seconds)


async def get_job_scope(redis: Any, job_id: str) -> str | None:
    raw = await redis.get(f"job:{job_id}:section")
    return raw if isinstance(raw, str) else None


async def release_claim(redis: Any, job_id: str) -> None:
    """
    Drop the claim after a FAILED attempt so a redelivered message may retry.
    Successfully completed jobs keep their claim (idempotency).
    """
    await redis.delete(_done_marker(job_id))
