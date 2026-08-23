"""
backend/db/dao/attendance.py
────────────────────────────
Class-session creation, idempotent attendance marking, and date reports.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from backend.db.models import Attendance, ClassSession, Enrollment, Student

if TYPE_CHECKING:
    from datetime import date

    from sqlalchemy.engine import CursorResult, Row
    from sqlalchemy.ext.asyncio import AsyncSession


async def create_class_session(
    session: AsyncSession,
    *,
    section_id: uuid.UUID,
    course_name: str,
    session_date: date,
    admin_id: uuid.UUID | None,
) -> ClassSession:
    row = ClassSession(
        session_id=uuid.uuid4(),
        section_id=section_id,
        course_name=course_name,
        session_date=session_date,
        admin_id=admin_id,
    )
    session.add(row)
    await session.flush()
    return row


async def get_class_session(session: AsyncSession, session_id: uuid.UUID) -> ClassSession | None:
    return await session.get(ClassSession, session_id)


async def mark_attendance(
    session: AsyncSession,
    *,
    student_id: uuid.UUID,
    session_id: uuid.UUID,
    session_date: date,
    match_distance: float | None = None,
) -> bool:
    """
    Insert an attendance row; idempotent.

    Re-processing the same job (or a duplicate face of the same student in
    one photo) must not double-write: the INSERT ... ON CONFLICT DO NOTHING
    respects uq_attendance_student_session. Returns True if inserted now,
    False if the row already existed.
    """
    stmt = (
        pg_insert(Attendance)
        .values(
            attendance_id=uuid.uuid4(),
            student_id=student_id,
            session_id=session_id,
            session_date=session_date,
            match_distance=match_distance,
        )
        .on_conflict_do_nothing(index_elements=["student_id", "session_id", "session_date"])
    )
    result = cast("CursorResult[Any]", await session.execute(stmt))
    return bool(result.rowcount)


async def get_attendance_for_date(
    session: AsyncSession,
    *,
    section_id: uuid.UUID,
    session_date: date,
) -> list[Row[Any]]:
    """
    Full roster report for a section on a date: every enrolled student with
    their attendance status (marked_at None ⇒ not marked).

    The attendance side of the outer join carries ``session_date`` so the
    query prunes to a single monthly partition.
    """
    session_ids = (
        select(ClassSession.session_id)
        .where(
            ClassSession.section_id == section_id,
            ClassSession.session_date == session_date,
        )
        .scalar_subquery()
    )
    stmt = (
        select(
            Student.student_id,
            Student.roll_number,
            Student.name.label("student_name"),
            Attendance.session_id,
            Attendance.match_distance,
            Attendance.marked_at,
        )
        .join(Enrollment, Enrollment.student_id == Student.student_id)
        .outerjoin(
            Attendance,
            (Attendance.student_id == Student.student_id)
            & (Attendance.session_date == session_date)
            & (Attendance.session_id.in_(session_ids)),
        )
        .where(Enrollment.section_id == section_id)
        .order_by(Student.roll_number)
    )
    return list((await session.execute(stmt)).all())


# ─── Biometric purge (Phase 5 — see docs/data-retention.md) ──────────────────


async def delete_attendance_for_student(session: AsyncSession, student_id: uuid.UUID) -> int:
    """Hard-delete every attendance row of a student across all partitions."""
    from sqlalchemy import delete

    result = cast(
        "CursorResult[Any]",
        await session.execute(delete(Attendance).where(Attendance.student_id == student_id)),
    )
    return int(result.rowcount)


async def anonymize_attendance_for_student(session: AsyncSession, student_id: uuid.UUID) -> int:
    """
    Strip the student reference but keep the statistical record
    (docs/data-retention.md step 3). student_id is nullable since
    migration 0002_attendance_anonymizable.
    """
    from sqlalchemy import update

    result = cast(
        "CursorResult[Any]",
        await session.execute(
            update(Attendance)
            .where(Attendance.student_id == student_id)
            .values(student_id=None, match_distance=None)
        ),
    )
    return int(result.rowcount)
