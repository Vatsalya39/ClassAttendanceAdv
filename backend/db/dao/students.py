"""
backend/db/dao/students.py
──────────────────────────
Student + face-embedding writes (enrollment path).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select
from sqlalchemy.engine import CursorResult

from backend.db.models import FaceEmbedding, Student

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class RollNumberAlreadyExistsError(Exception):
    """Raised when create_student hits the unique roll_number constraint."""


async def get_student_by_roll_number(session: AsyncSession, roll_number: str) -> Student | None:
    stmt = select(Student).where(Student.roll_number == roll_number)
    return (await session.execute(stmt)).scalar_one_or_none()


async def get_student(session: AsyncSession, student_id: uuid.UUID) -> Student | None:
    return await session.get(Student, student_id)


async def create_student(
    session: AsyncSession,
    *,
    roll_number: str,
    name: str,
    password_hash: str,
    institution_id: uuid.UUID,
) -> Student:
    """
    Insert a student row. Raises RollNumberAlreadyExistsError on duplicate
    roll_number; caller decides whether to commit.
    """
    existing = await get_student_by_roll_number(session, roll_number)
    if existing is not None:
        raise RollNumberAlreadyExistsError(f"roll_number '{roll_number}' already registered")

    student = Student(
        student_id=uuid.uuid4(),
        roll_number=roll_number,
        name=name,
        password_hash=password_hash,
        institution_id=institution_id,
    )
    session.add(student)
    await session.flush()
    return student


async def add_embedding(
    session: AsyncSession,
    *,
    student_id: uuid.UUID,
    embedding: list[float],
    source_image_url: str | None = None,
) -> FaceEmbedding:
    """Store one 512-d ArcFace embedding for a student (3-5 per student)."""
    row = FaceEmbedding(
        embedding_id=uuid.uuid4(),
        student_id=student_id,
        embedding=embedding,
        source_image_url=source_image_url,
    )
    session.add(row)
    await session.flush()
    return row


# ─── Biometric purge (Phase 5 — see docs/data-retention.md) ──────────────────


async def list_embedding_image_urls(session: AsyncSession, student_id: uuid.UUID) -> list[str]:
    """All enrollment-image URLs to delete from storage BEFORE the rows go."""
    stmt = select(FaceEmbedding.source_image_url).where(FaceEmbedding.student_id == student_id)
    return [u for u in (await session.execute(stmt)).scalars() if u]


async def delete_embeddings(session: AsyncSession, student_id: uuid.UUID) -> int:
    from sqlalchemy import delete

    result = cast(
        "CursorResult[Any]",
        await session.execute(delete(FaceEmbedding).where(FaceEmbedding.student_id == student_id)),
    )
    return int(result.rowcount)
