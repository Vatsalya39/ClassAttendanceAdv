"""
backend/worker/main.py
─────────────────────
Capture-job worker — a separate process from the API server.

Loop:
    queue.consume()
      ├─ claim(job_id) via Redis SET NX      (idempotency guard)
      ├─ status = processing
      ├─ for each burst frame (Module 4):
      │    storage.download(frame_url)
      │    batch_process(image, native-res, tiled, quality gates, TTA)
      │    match_and_assign(section_id)       (ROSTER-SCOPED — never global)
      │    crop unmatched faces → storage.upload
      ├─ write_capture_results()             (attendance + review_queue, idempotent)
      ├─ auto-resolve reviews confirmed by later burst frames (Module 4)
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
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from backend.core.config import settings
from backend.core.jobstatus import JobStatus, claim, release_claim, set_status
from backend.core.queue import CaptureJobMessage, ReceivedMessage
from backend.core.storage import make_storage_backend
from backend.db.session import dispose_engine, get_session_factory
from backend.services.capture import FaceOutcome, match_and_assign, write_capture_results
from ml.pipeline import QualityGateConfig, TiledDetectionConfig, batch_process
from ml.utils import load_image_from_bytes

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


def _providers_from_settings() -> list[str] | None:
    """Parse the comma-separated ONNX providers env into a list."""
    raw = settings.onnx_providers.strip()
    if not raw:
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def _detect_frame_faces(image: np.ndarray) -> list[Any]:
    """
    Run the Module 1+2 pipeline on ONE frame at NATIVE resolution
    (no resize — session photos are never downscaled anymore).
    """
    quality = QualityGateConfig(
        min_size_px=settings.face_min_size_px,
        min_blur_var=settings.face_min_blur_var,
        max_pose_ratio=settings.face_max_pose_ratio,
        use_pnp_pose=settings.face_use_pnp_pose,
        max_yaw_deg=settings.face_max_yaw_deg,
        max_pitch_deg=settings.face_max_pitch_deg,
    )
    tiled = (
        TiledDetectionConfig(
            tile_min_size=settings.session_tile_min_size,
            overlap=settings.session_tile_overlap,
            nms_iou=settings.face_nms_iou,
        )
        if settings.session_tiled_detection
        else None
    )

    return batch_process(
        image,
        tta_flip=settings.face_tta_flip,
        quality_config=quality,
        tiled_config=tiled,
        model_pack=settings.insightface_model_pack,
        providers=_providers_from_settings(),
        det_size=settings.session_det_size,
        max_faces=settings.face_detection_max_faces,
    )


async def _auto_resolve_reviews(
    session: Any,
    *,
    newly_matched_students: set[Any],
    session_id: Any,
) -> int:
    """
    Module 4: when a later burst frame confidently matches a student whose
    face is sitting in the review queue for THIS class session, resolve the
    pending item automatically — the burst itself is the evidence.

    Only resolves items whose best_guess equals the confirmed student, i.e.
    the system had guessed right but lacked confidence; admin eyeballing
    stays available for everything else.
    """
    if not newly_matched_students:
        return 0

    from sqlalchemy import select

    from backend.db.models import ReviewQueue, ReviewStatus

    rows = (
        (
            await session.execute(
                select(ReviewQueue).where(
                    ReviewQueue.session_id == session_id,
                    ReviewQueue.status == ReviewStatus.pending,
                )
            )
        )
        .scalars()
        .all()
    )
    resolved = 0
    for review in rows:
        if review.best_guess_student_id in newly_matched_students:
            review.status = ReviewStatus.resolved
            review.resolved_student_id = review.best_guess_student_id
            review.resolved_at = datetime.now(UTC)
            # No admin id — the worker resolved it; audit_action below
            # records the auto-resolve count on the capture.processed entry.
            resolved += 1
    if resolved:
        await session.flush()
        log.info("auto-resolved %d review item(s) from burst frames", resolved)
    return resolved


async def process_capture_job(
    message: CaptureJobMessage,
    *,
    storage: Any,
    redis: Any,
) -> dict[str, Any]:
    """
    Process one capture job end-to-end (all burst frames). Returns the
    result payload stored in Redis. Split out from the consume loop so tests
    can call it directly.
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
        # Module 4: primary frame + any burst frames (legacy messages carry
        # only image_url, so frame_urls may be empty).
        frame_urls = [message.image_url] + [
            u for u in (message.frame_urls or []) if u != message.image_url
        ]

        # Import here so the worker process loads InsightFace lazily/warmly.
        from ml.pipeline import _configure_pipeline

        _configure_pipeline(
            model_pack=settings.insightface_model_pack,
            providers=_providers_from_settings(),
            det_size=settings.session_det_size,
        )

        factory = get_session_factory()
        async with factory() as session:
            all_outcomes: list[FaceOutcome] = []
            crop_urls: dict[int, str] = {}
            matched_students: set[Any] = set()

            if settings.burst_temporal_fusion and len(frame_urls) > 1:
                from ml.temporal import fuse_burst_detections

                burst_faces: list[list[Any]] = []
                frame_images: list[np.ndarray] = []
                for frame_idx, frame_url in enumerate(frame_urls):
                    image_bytes = await storage.download(frame_url)
                    image = load_image_from_bytes(image_bytes)
                    frame_images.append(image)
                    faces = _detect_frame_faces(image)
                    burst_faces.append(faces)
                    log.info(
                        "job %s burst frame %d/%d: %d face(s) detected in %.2fs",
                        job_id,
                        frame_idx + 1,
                        len(frame_urls),
                        len(faces),
                        asyncio.get_event_loop().time() - t0,
                    )

                fused_faces = fuse_burst_detections(
                    burst_faces, iou_threshold=settings.burst_iou_threshold
                )
                log.info(
                    "job %s: temporally fused %d burst frames into %d unique face track(s)",
                    job_id,
                    len(frame_urls),
                    len(fused_faces),
                )

                pairs = [
                    ((f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2), f.embedding) for f in fused_faces
                ]
                outcomes = await match_and_assign(
                    session,
                    section_id=message.section_id,
                    face_embeddings=pairs,
                    threshold=settings.face_match_threshold,
                    top_k=settings.face_match_top_k,
                    margin=settings.face_match_margin,
                    candidate_students=settings.face_candidate_students,
                    face_quality_flags=[f.quality_flags for f in fused_faces],
                )
                all_outcomes.extend(outcomes)
                for i, outcome in enumerate(outcomes):
                    if not outcome.matched:
                        crop_bytes = _crop_to_jpeg(frame_images[0], outcome.bbox)
                        crop_urls[i] = await storage.upload(
                            data=crop_bytes,
                            filename=f"review_{job_id[:8]}_{i}.jpg",
                            container="review-crops",
                            content_type="image/jpeg",
                        )
                matched_students.update(
                    o.student_id for o in outcomes if o.matched and o.student_id
                )
            else:
                for frame_idx, frame_url in enumerate(frame_urls):
                    image_bytes = await storage.download(frame_url)
                    # NATIVE resolution — no resize_for_detection for session
                    # photos (Module 2); tiled detection handles wide shots.
                    image = load_image_from_bytes(image_bytes)
                    faces = _detect_frame_faces(image)
                    log.info(
                        "job %s frame %d/%d: %d face(s) detected in %.2fs",
                        job_id,
                        frame_idx + 1,
                        len(frame_urls),
                        len(faces),
                        asyncio.get_event_loop().time() - t0,
                    )

                    pairs = [
                        ((f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2), f.embedding) for f in faces
                    ]
                    outcomes = await match_and_assign(
                        session,
                        section_id=message.section_id,
                        face_embeddings=pairs,
                        threshold=settings.face_match_threshold,
                        top_k=settings.face_match_top_k,
                        margin=settings.face_match_margin,
                        candidate_students=settings.face_candidate_students,
                        face_quality_flags=[f.quality_flags for f in faces],
                    )

                    # Crops must be encoded while THIS frame's image is in
                    # memory; indices are global across the accumulated faces.
                    offset = len(all_outcomes)
                    all_outcomes.extend(outcomes)
                    for i, outcome in enumerate(outcomes):
                        if not outcome.matched:
                            crop_bytes = _crop_to_jpeg(image, outcome.bbox)
                            crop_urls[offset + i] = await storage.upload(
                                data=crop_bytes,
                                filename=f"review_{job_id[:8]}_{offset + i}.jpg",
                                container="review-crops",
                                content_type="image/jpeg",
                            )

                    matched_students.update(
                        o.student_id for o in outcomes if o.matched and o.student_id
                    )

            # Module 4: burst frames may confirm students whose earlier-frame
            # faces went to review — auto-resolve those pending items.
            auto_resolved = await _auto_resolve_reviews(
                session,
                newly_matched_students=matched_students,
                session_id=message.session_id,
            )

            capture_outcome = await write_capture_results(
                session,
                session_id=message.session_id,
                outcomes=all_outcomes,
                crop_urls=crop_urls,
            )
            from backend.db import dao

            await dao.audit_action(
                session,
                admin_id=message.admin_id,
                action="capture.processed",
                target_type="class_session",
                target_id=message.session_id,
                detail=(
                    f"job={message.job_id} frames={len(frame_urls)} "
                    f"auto_resolved={auto_resolved}"
                ),
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
            "job %s done: matched=%d review=%d frames=%d (%.2fs)",
            job_id,
            len(capture_outcome.matched),
            len(capture_outcome.review_needed),
            len(frame_urls),
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
        "Worker starting | environment=%s queue=%s concurrency>=%d "
        "threshold=%.2f margin=%.2f top_k=%d pack=%s tta=%s",
        settings.environment.value,
        settings.queue_backend.value,
        workers_allowed,
        settings.face_match_threshold,
        settings.face_match_margin,
        settings.face_match_top_k,
        settings.insightface_model_pack,
        settings.face_tta_flip,
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
