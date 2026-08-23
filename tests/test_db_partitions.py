"""
tests/test_db_partitions.py
───────────────────────────
Phase 1 acceptance test (b): attendance partition pruning works — a query
filtered on session_date only scans the relevant monthly partition — plus
mark_attendance idempotency and the DEFAULT catch-all partition.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from backend.db import dao
from backend.db.partitions import add_months, month_start, partition_name

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def test_add_months_clamps_and_wraps() -> None:
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)  # leap year
    assert add_months(date(2026, 12, 15), 2) == date(2027, 2, 15)
    assert month_start(date(2026, 8, 23)) == date(2026, 8, 1)
    assert partition_name(date(2026, 8, 23)) == "attendance_2026_08"


@pytest.fixture
async def world(db_session: AsyncSession) -> dict[str, uuid.UUID]:
    inst = await dao.create_institution(db_session, name="Part U")
    dept = await dao.create_department(db_session, institution_id=inst.institution_id, name="D")
    section = await dao.create_section(
        db_session, department_id=dept.department_id, name="SEC-1", academic_year=2026
    )
    student = await dao.create_student(
        db_session,
        roll_number="P001",
        name="Partition Tester",
        password_hash="x",
        institution_id=inst.institution_id,
    )
    from backend.db.models import Enrollment

    db_session.add(
        Enrollment(
            enrollment_id=uuid.uuid4(),
            student_id=student.student_id,
            section_id=section.section_id,
        )
    )
    await db_session.commit()
    return {
        "section_id": section.section_id,
        "student_id": student.student_id,
    }


async def _explain_partitions(session: AsyncSession, query_date: date) -> str:
    result = await session.execute(
        text(
            f"EXPLAIN SELECT * FROM attendance WHERE session_date = '{query_date:%Y-%m-%d}'"  # noqa: S608
        )
    )
    return "\n".join(str(row[0]) for row in result)


async def test_partition_pruning_scans_only_relevant_month(
    db_session: AsyncSession, world: dict[str, uuid.UUID]
) -> None:
    this_month = month_start(date.today())
    next_month = add_months(this_month, 1)

    # Insert attendance in two different months (separate sessions).
    s1 = await dao.create_class_session(
        db_session,
        section_id=world["section_id"],
        course_name="DBMS",
        session_date=this_month,
        admin_id=None,
    )
    await dao.mark_attendance(
        db_session,
        student_id=world["student_id"],
        session_id=s1.session_id,
        session_date=this_month,
        match_distance=0.32,
    )
    s2 = await dao.create_class_session(
        db_session,
        section_id=world["section_id"],
        course_name="DBMS",
        session_date=add_months(this_month, 2),
        admin_id=None,
    )
    await dao.mark_attendance(
        db_session,
        student_id=world["student_id"],
        session_id=s2.session_id,
        session_date=add_months(this_month, 2),
        match_distance=0.33,
    )
    await db_session.commit()

    plan_this = await _explain_partitions(db_session, this_month)
    plan_next = await _explain_partitions(db_session, next_month)

    assert partition_name(this_month) in plan_this, plan_this
    assert partition_name(add_months(this_month, 2)) not in plan_this, plan_this

    assert partition_name(next_month) in plan_next, plan_next
    assert partition_name(this_month) not in plan_next, plan_next


async def test_mark_attendance_is_idempotent(
    db_session: AsyncSession, world: dict[str, uuid.UUID]
) -> None:
    session_row = await dao.create_class_session(
        db_session,
        section_id=world["section_id"],
        course_name="OS",
        session_date=date.today(),
        admin_id=None,
    )
    sid, stid = (
        session_row.session_id,
        world["student_id"],
    )

    first = await dao.mark_attendance(
        db_session,
        student_id=stid,
        session_id=sid,
        session_date=session_row.session_date,
        match_distance=0.21,
    )
    await db_session.commit()
    second = await dao.mark_attendance(
        db_session,
        student_id=stid,
        session_id=sid,
        session_date=session_row.session_date,
        match_distance=0.99,
    )
    await db_session.commit()

    assert first is True
    assert second is False, "duplicate attendance write must be a no-op"

    count = len(
        (
            await db_session.execute(
                text("SELECT 1 FROM attendance WHERE session_id = :sid"),
                {"sid": str(sid)},
            )
        ).fetchall()
    )
    assert count == 1


async def test_default_partition_catches_stray_dates(
    db_session: AsyncSession, world: dict[str, uuid.UUID]
) -> None:
    """A date with no pre-created partition must not fail the insert."""
    far_future = add_months(month_start(date.today()), 36)
    session_row = await dao.create_class_session(
        db_session,
        section_id=world["section_id"],
        course_name="Far Future",
        session_date=far_future,
        admin_id=None,
    )
    ok = await dao.mark_attendance(
        db_session,
        student_id=world["student_id"],
        session_id=session_row.session_id,
        session_date=far_future,
    )
    await db_session.commit()
    assert ok is True

    rows = (
        await db_session.execute(
            text(
                f"SELECT tableoid::regclass FROM attendance "  # noqa: S608
                f"WHERE session_date = '{far_future:%Y-%m-%d}'"
            )
        )
    ).fetchall()
    assert str(rows[0][0]) == "attendance_default"


async def test_attendance_rows_in_correct_partition(
    db_session: AsyncSession, world: dict[str, uuid.UUID]
) -> None:
    today = date.today()
    session_row = await dao.create_class_session(
        db_session,
        section_id=world["section_id"],
        course_name="CN",
        session_date=today,
        admin_id=None,
    )
    await dao.mark_attendance(
        db_session,
        student_id=world["student_id"],
        session_id=session_row.session_id,
        session_date=today,
    )
    await db_session.commit()

    rows = (
        await db_session.execute(
            text(
                f"SELECT tableoid::regclass FROM attendance "
                f"WHERE session_date = '{today:%Y-%m-%d}'"  # noqa: S608
            )
        )
    ).fetchall()
    assert rows, "expected one attendance row"
    assert str(rows[0][0]) == partition_name(today)
