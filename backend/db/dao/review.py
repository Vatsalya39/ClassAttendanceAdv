"""
backend/db/dao/review.py
────────────────────────
Review-queue writes: unmatched faces go here instead of being force-matched.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select
from sqlalchemy.engine import CursorResult

from backend.db.models import ClassSession, ReviewQueue, ReviewStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def enqueue_review(
    session: AsyncSession,
    *,
    session_id: uuid.UUID,
    face_crop_url: str | None = None,
    face_embedding: list[float] | None = None,
    best_guess_student_id: uuid.UUID | None = None,
    best_guess_distance: float | None = None,
) -> ReviewQueue:
    row = ReviewQueue(
        review_id=uuid.uuid4(),
        session_id=session_id,
        face_crop_url=face_crop_url,
        face_embedding=face_embedding,
        best_guess_student_id=best_guess_student_id,
        best_guess_distance=best_guess_distance,
        status=ReviewStatus.pending,
    )
    session.add(row)
    await session.flush()
    return row


async def get_review_item(session: AsyncSession, review_id: uuid.UUID) -> ReviewQueue | None:
    return await session.get(ReviewQueue, review_id)


async def list_pending_reviews(
    session: AsyncSession,
    *,
    section_ids: list[uuid.UUID],
    limit: int = 100,
    offset: int = 0,
) -> list[ReviewQueue]:
    """Pending items for the given (access-checked) sections, oldest first."""
    if not section_ids:
        return []
    stmt = (
        select(ReviewQueue)
        .join(ClassSession, ClassSession.session_id == ReviewQueue.session_id)
        .where(
            ClassSession.section_id.in_(section_ids),
            ReviewQueue.status == ReviewStatus.pending,
        )
        .order_by(ReviewQueue.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    return list((await session.execute(stmt)).scalars())


class ReviewAlreadyResolvedError(Exception):
    pass


async def resolve_review(
    session: AsyncSession,
    *,
    review_id: uuid.UUID,
    admin_id: uuid.UUID,
    resolved_student_id: uuid.UUID | None,
) -> ReviewQueue:
    """
    Resolve a pending item. ``resolved_student_id=None`` means DISCARD.

    Raises ReviewAlreadyResolvedError if the item is not pending.
    """
    review = await get_review_item(session, review_id)
    if review is None:
        raise LookupError(f"review {review_id} does not exist")
    if review.status != ReviewStatus.pending:
        raise ReviewAlreadyResolvedError(f"review {review_id} already resolved")

    review.status = ReviewStatus.resolved
    review.resolved_student_id = resolved_student_id
    review.resolved_by_admin_id = admin_id
    review.resolved_at = datetime.now(UTC)
    await session.flush()
    return review


async def delete_reviews_for_student(session: AsyncSession, student_id: uuid.UUID) -> int:
    """
    Remove review items that reference the student, whether as the unmatched
    face's best guess or as the manually resolved assignment
    (docs/data-retention.md step 4).
    """
    from sqlalchemy import delete, or_

    result = cast(
        "CursorResult[Any]",
        await session.execute(
            delete(ReviewQueue).where(
                or_(
                    ReviewQueue.best_guess_student_id == student_id,
                    ReviewQueue.resolved_student_id == student_id,
                )
            )
        ),
    )
    return int(result.rowcount)
