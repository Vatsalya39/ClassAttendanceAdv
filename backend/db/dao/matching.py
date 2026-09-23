"""
backend/db/dao/matching.py
──────────────────────────
THE two vector-search entry points.

find_nearest_embeddings_for_roster
    Joins face_embeddings to enrollments filtered by section_id BEFORE the
    vector ordering. The candidate set is that section's roster embeddings
    (~60-70 students × 3-5 images = a few hundred rows at most) — identical
    latency whether the system has 5,000 or 50,000 enrolled students.
    This is the ONLY function the capture worker may use.

find_nearest_embeddings_institution_wide
    The deliberately-slower fallback used exclusively by the admin's manual
    review-queue "search all students" action. Explicitly named so it never
    gets reused as a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

from backend.db.models import Enrollment, FaceEmbedding, Student

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class RosterMatch:
    """One candidate row returned by a vector search."""

    student_id: uuid.UUID
    embedding_id: uuid.UUID
    distance: float  # cosine distance (0 = identical direction)


async def find_nearest_embeddings_for_roster(
    session: AsyncSession,
    query_vector: list[float],
    section_id: uuid.UUID,
    top_k: int = 1,
) -> list[RosterMatch]:
    """
    Top-k nearest embeddings among students ENROLLED IN ``section_id`` only.

    The join to enrollments happens before the ORDER BY on the cosine
    distance operator, so Postgres ranks at most the section's few hundred
    candidate embeddings — never the full table (see the comment on
    FaceEmbedding's HNSW index in models.py).
    """
    distance_expr = FaceEmbedding.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            FaceEmbedding.student_id,
            FaceEmbedding.embedding_id,
            distance_expr.label("distance"),
        )
        .join(Enrollment, Enrollment.student_id == FaceEmbedding.student_id)
        .where(Enrollment.section_id == section_id)
        .order_by(distance_expr.asc())
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [
        RosterMatch(
            student_id=r.student_id, embedding_id=r.embedding_id, distance=float(r.distance)
        )
        for r in rows
    ]


async def find_nearest_embeddings_for_student(
    session: AsyncSession,
    query_vector: list[float],
    student_id: uuid.UUID,
    top_k: int = 5,
) -> list[RosterMatch]:
    """
    Top-k nearest embeddings belonging to ONE student only.

    Enrollment-hygiene check (Module 4): a new enrollment photo must be
    closer to its own student's existing references than to anyone else's —
    this returns the "own" side of that comparison. Scope is a single
    student, so it is trivially roster-safe.
    """
    distance_expr = FaceEmbedding.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            FaceEmbedding.student_id,
            FaceEmbedding.embedding_id,
            distance_expr.label("distance"),
        )
        .where(FaceEmbedding.student_id == student_id)
        .order_by(distance_expr.asc())
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [
        RosterMatch(
            student_id=r.student_id, embedding_id=r.embedding_id, distance=float(r.distance)
        )
        for r in rows
    ]


async def find_nearest_embeddings_institution_wide(
    session: AsyncSession,
    query_vector: list[float],
    institution_id: uuid.UUID,
    top_k: int = 5,
) -> list[RosterMatch]:
    """
    Institution-wide search — SLOWER by design and scoped to one tenant.

    Used ONLY by POST /admin/review-queue/{id}/resolve with the explicit
    "search all students" flag. Never call this from the automatic capture
    path; that path must use find_nearest_embeddings_for_roster.
    """
    distance_expr = FaceEmbedding.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            FaceEmbedding.student_id,
            FaceEmbedding.embedding_id,
            distance_expr.label("distance"),
        )
        .join(Student, Student.student_id == FaceEmbedding.student_id)
        .where(Student.institution_id == institution_id)
        .order_by(distance_expr.asc())
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [
        RosterMatch(
            student_id=r.student_id, embedding_id=r.embedding_id, distance=float(r.distance)
        )
        for r in rows
    ]
