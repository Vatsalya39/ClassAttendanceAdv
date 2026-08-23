"""
backend/services/capture.py
───────────────────────────
Capture-processing business logic.

This module is the bridge between Phase 2's ML pipeline and Phase 1's DAO:

    batch_process(image)  →  for each face:
        find_nearest_embeddings_for_roster(embedding, section_id)
            ├─ distance ≤ threshold → mark_attendance (idempotent)
            └─ otherwise            → enqueue_review (never force-matched)

Kept separate from backend/worker/main.py so tests can exercise the
matching/attendance logic without images, a queue, or InsightFace installed.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.dao import (
    find_nearest_embeddings_for_roster,
    get_class_session,
    mark_attendance,
)

if TYPE_CHECKING:
    from datetime import date

log = logging.getLogger(__name__)


@dataclass
class FaceOutcome:
    """Result of matching ONE detected face against one section's roster."""

    bbox: tuple[int, int, int, int]
    matched: bool
    student_id: uuid.UUID | None = None
    match_distance: float | None = None
    # Kept so unmatched faces can be re-searched institution-wide later.
    embedding: list[float] | None = None


@dataclass
class CaptureOutcome:
    """Aggregated result of processing one classroom photo."""

    faces_total: int = 0
    matched: list[FaceOutcome] = field(default_factory=list)
    review_needed: list[FaceOutcome] = field(default_factory=list)


async def match_faces_to_roster(
    session: AsyncSession,
    *,
    section_id: uuid.UUID,
    face_embeddings: list[tuple[tuple[int, int, int, int], list[float]]],
    threshold: float,
    top_k: int = 1,
) -> list[FaceOutcome]:
    """
    Match every detected face against ONLY this section's roster.

    ``face_embeddings`` is a list of (bbox, embedding) pairs as produced by
    ml.pipeline.batch_process(). The threshold comparison happens here (not
    in SQL) so the calibration value stays a single setting.
    """
    outcomes: list[FaceOutcome] = []
    for bbox, embedding in face_embeddings:
        candidates = await find_nearest_embeddings_for_roster(
            session, embedding, section_id=section_id, top_k=top_k
        )
        best = candidates[0] if candidates else None
        if best is not None and best.distance <= threshold:
            outcomes.append(
                FaceOutcome(
                    bbox=bbox,
                    matched=True,
                    student_id=best.student_id,
                    match_distance=float(best.distance),
                )
            )
        else:
            outcomes.append(
                FaceOutcome(
                    bbox=bbox,
                    matched=False,
                    student_id=None,
                    match_distance=float(best.distance) if best else None,
                    embedding=embedding,
                )
            )
    return outcomes


async def write_capture_results(
    session: AsyncSession,
    *,
    session_id: uuid.UUID,
    outcomes: list[FaceOutcome],
    crop_urls: dict[int, str] | None = None,
) -> CaptureOutcome:
    """
    Persist matching results for one capture job.

    Matched faces → attendance rows (idempotent via ON CONFLICT).
    Unmatched     → review_queue rows (with best-guess info when available).

    ``crop_urls`` maps the index of each unmatched face to its stored crop
    URL. Returns the aggregated outcome; caller owns the commit and the
    job-status update.
    """
    class_session = await get_class_session(session, session_id)
    if class_session is None:
        raise LookupError(f"class session {session_id} does not exist")

    session_date: date = class_session.session_date
    outcome = CaptureOutcome(faces_total=len(outcomes))

    for idx, face in enumerate(outcomes):
        if face.matched and face.student_id is not None:
            inserted = await mark_attendance(
                session,
                student_id=face.student_id,
                session_id=session_id,
                session_date=session_date,
                match_distance=face.match_distance,
            )
            if inserted:
                outcome.matched.append(face)
            else:
                log.debug(
                    "write_capture_results: duplicate attendance for student %s "
                    "in session %s — idempotently skipped",
                    face.student_id,
                    session_id,
                )
        else:
            # Lazy import avoids a cycle: review dao pulls models only.
            from backend.db.dao import enqueue_review

            await enqueue_review(
                session,
                session_id=session_id,
                face_crop_url=(crop_urls or {}).get(idx),
                face_embedding=face.embedding,
                best_guess_student_id=face.student_id,
                best_guess_distance=face.match_distance,
            )
            outcome.review_needed.append(face)

    await session.flush()
    return outcome
