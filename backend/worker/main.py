"""
backend/worker/main.py
──────────────────────
Capture-job worker — a separate process from the API server.

Loop:
    queue.consume()
      ├─ claim(job_id) via Redis SET NX      (idempotency guard)
      ├─ status = processing
      ├─ storage.download(image_url)
      ├─ batch_process(image)                (ml.pipeline, warm model)
      ├─ match_faces_to_roster(section_id)   (ROSTER-SCOPED — never global)
      ├─ crop unmatched faces → storage.upload
      ├─ write_capture_results()             (attendance + review_queue, idempotent)
      ├─ status = done {matched, review_needed}
      └─ ack
    on error → status = failed, nack → dead-letter

Run locally:
    python -m backend.worker.main

In docker-compose this runs as a separate container sharing the API image.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

import cv2
import numpy as np

from backend.core.config import settings
from backend.core.jobstatus import JobStatus, claim, release_claim, set_status
from backend.core.queue import CaptureJobMessage, ReceivedMessage
from backend.core.storage import make_storage_backend
from backend.db.session import dispose_engine, get_session_factory
from backend.services.capture import match_faces_to_roster, write_capture_results
from ml.utils import load_image_from_bytes, resize_for_detection

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("worker")

_stop_event = asyncio.Event()

# Bounding-box margin (fraction of box size) around unmatched faces for crops.
_CROP_MARGIN = 0.25


def _handle_signal(sig: int, frame: object) -> None:
    log.info("Received signal %s, shutting down gracefully…", sig)
    _stop_event.set()


def _crop_to_jpeg(image: np.ndarray, bbox: tuple[int, int, int, int]) -> bytes:
    """Encode a face crop (with margin) as JPEG bytes for the review queue."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    mx = int((x2 - x1) * _CROP_MARGIN)
    my = int((y2 - y1) * _CROP_MARGIN)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    ok, buf = cv2.imencode(".jpg", image[y1:y2, x1:x2])
    if not ok:  # pragma: no cover — cv2 encode of a valid slice always works
        raise RuntimeError("failed to encode face crop")
    return buf.tobytes()


async def process_capture_job(
    message: CaptureJobMessage,
    *,
    storage: Any,
    redis: Any,
) -> dict[str, Any]:
    """
    Process one capture job end-to-end. Returns the result payload stored in
    Redis. Split out from the consume loop so tests can call it directly.
    """
    job_id = str(message.job_id)

    # Idempotency: a redelivered job that already completed is skipped.
    if not await claim(redis, job_id, ttl_seconds=settings.job_status_ttl_seconds):
        log.info("job %s already processed — skipping duplicate delivery", job_id)
        return {}

    await set_status(
        redis, job_id, JobStatus.processing, ttl_seconds=settings.job_status_ttl_seconds
    )
    t0 = asyncio.get_event_loop().time()

    try:
        image_bytes = await storage.download(message.image_url)
        image = load_image_from_bytes(image_bytes)
        image = resize_for_detection(image)

        # Import here so the worker process loads InsightFace lazily/warmly.
        from ml.pipeline import batch_process

        faces = batch_process(image)
        log.info(
            "job %s: %d face(s) detected in %.2fs",
            job_id,
            len(faces),
            asyncio.get_event_loop().time() - t0,
        )

        factory = get_session_factory()
        async with factory() as session:
            pairs = [((f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2), f.embedding) for f in faces]
            outcomes = await match_faces_to_roster(
                session,
                section_id=message.section_id,
                face_embeddings=pairs,
                threshold=settings.face_match_threshold,
            )

            # Persist crops for unmatched faces so admins can eyeball them.
            crop_urls: dict[int, str] = {}
            for idx, outcome in enumerate(outcomes):
                if not outcome.matched:
                    crop_bytes = _crop_to_jpeg(image, outcome.bbox)
                    crop_urls[idx] = await storage.upload(
                        data=crop_bytes,
                        filename=f"review_{job_id[:8]}_{idx}.jpg",
                        container="review-crops",
                        content_type="image/jpeg",
                    )

            capture_outcome = await write_capture_results(
                session,
                session_id=message.session_id,
                outcomes=outcomes,
                crop_urls=crop_urls,
            )
            from backend.db import dao

            await dao.audit_action(
                session,
                admin_id=message.admin_id,
                action="capture.processed",
                target_type="class_session",
                target_id=message.session_id,
                detail=f"job={message.job_id}",
            )
            await session.commit()

        result = {
            "matched": [
                {"student_id": str(o.student_id), "distance": o.match_distance}
                for o in capture_outcome.matched
            ],
            "review_needed": [
                {"student_id": str(o.student_id), "distance": o.match_distance}
                for o in capture_outcome.review_needed
            ],
        }
        await set_status(
            redis,
            job_id,
            JobStatus.done,
            result=result,
            ttl_seconds=settings.job_status_ttl_seconds,
        )
        log.info(
            "job %s done: matched=%d review=%d (%.2fs)",
            job_id,
            len(capture_outcome.matched),
            len(capture_outcome.review_needed),
            asyncio.get_event_loop().time() - t0,
        )
        return result

    except Exception as exc:  # noqa: BLE001 — worker must survive anything
        log.exception("job %s failed", job_id)
        await set_status(
            redis,
            job_id,
            JobStatus.failed,
            error=str(exc)[:512],
            ttl_seconds=settings.job_status_ttl_seconds,
        )
        # Allow a redelivered copy to retry (the ack/nack policy still
        # dead-letters this delivery; the claim must not block retries).
        await release_claim(redis, job_id)
        raise


async def run_worker(concurrency: int | None = None) -> None:
    """Main worker loop: consume jobs until stopped."""
    import redis.asyncio as aioredis

    from backend.core.queue import make_queue_backend

    workers_allowed = concurrency or settings.worker_concurrency
    queue = make_queue_backend(settings)
    storage = make_storage_backend(settings)
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    log.info(
        "Worker starting | environment=%s queue=%s concurrency>=%d threshold=%.2f",
        settings.environment.value,
        settings.queue_backend.value,
        workers_allowed,
        settings.face_match_threshold,
    )

    async def consume_forever() -> None:
        async for received in queue.consume():
            received_msg: ReceivedMessage = received
            if _stop_event.is_set():
                break
            try:
                await process_capture_job(received_msg.payload, storage=storage, redis=redis)
                await received_msg.ack()
            except Exception:  # noqa: BLE001
                await received_msg.nack(reason="capture-processing-failed")
            if _stop_event.is_set():
                break

    consumer_task = asyncio.create_task(consume_forever())
    stop_watch = asyncio.create_task(_stop_event.wait())
    done, pending = await asyncio.wait(
        {consumer_task, stop_watch}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    await redis.close()
    await queue.close()
    await storage.close()
    await dispose_engine()
    log.info("Worker stopped.")


def main() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_worker())
    sys.exit(0)


if __name__ == "__main__":
    main()
