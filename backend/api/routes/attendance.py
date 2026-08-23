"""
backend/api/routes/attendance.py
────────────────────────────────
Student-facing attendance history (JWT-protected).
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from backend.api.deps import CurrentStudent, DbSession
from backend.api.schemas import AttendanceRecord, StudentAttendanceResponse
from backend.db.models import Attendance, ClassSession

router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.get("/me", response_model=StudentAttendanceResponse)
async def my_attendance(
    student: CurrentStudent,
    session: DbSession,
) -> StudentAttendanceResponse:
    stmt = (
        select(
            ClassSession.session_id,
            ClassSession.course_name,
            ClassSession.session_date,
            Attendance.marked_at,
            Attendance.match_distance,
        )
        .join(Attendance, Attendance.session_id == ClassSession.session_id)
        .where(Attendance.student_id == student.student_id)
        .order_by(ClassSession.session_date.desc())
    )
    rows = (await session.execute(stmt)).all()
    return StudentAttendanceResponse(
        student_id=student.student_id,
        records=[
            AttendanceRecord(
                session_id=r.session_id,
                course_name=r.course_name,
                session_date=r.session_date,
                marked_at=r.marked_at,
                match_distance=r.match_distance,
            )
            for r in rows
        ],
    )
