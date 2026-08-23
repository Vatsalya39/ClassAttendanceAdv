"""
backend/db/dao/org.py
─────────────────────
Institution / department / section management + bulk enrollment.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from backend.db.models import Department, Enrollment, Institution, Section, Student

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# ─── Institutions ────────────────────────────────────────────────────────────


async def create_institution(session: AsyncSession, *, name: str) -> Institution:
    institution = Institution(institution_id=uuid.uuid4(), name=name)
    session.add(institution)
    await session.flush()
    return institution


async def list_institutions(session: AsyncSession) -> list[Institution]:
    return list((await session.execute(select(Institution).order_by(Institution.name))).scalars())


# ─── Departments ─────────────────────────────────────────────────────────────


async def create_department(
    session: AsyncSession, *, institution_id: uuid.UUID, name: str
) -> Department:
    department = Department(department_id=uuid.uuid4(), institution_id=institution_id, name=name)
    session.add(department)
    await session.flush()
    return department


async def list_departments(
    session: AsyncSession, institution_id: uuid.UUID | None = None
) -> list[Department]:
    stmt = select(Department).order_by(Department.name)
    if institution_id is not None:
        stmt = stmt.where(Department.institution_id == institution_id)
    return list((await session.execute(stmt)).scalars())


# ─── Sections ────────────────────────────────────────────────────────────────


async def create_section(
    session: AsyncSession,
    *,
    department_id: uuid.UUID,
    name: str,
    academic_year: int,
) -> Section:
    section = Section(
        section_id=uuid.uuid4(),
        department_id=department_id,
        name=name,
        academic_year=academic_year,
    )
    session.add(section)
    await session.flush()
    return section


async def get_section_by_id(session: AsyncSession, section_id: uuid.UUID) -> Section | None:
    return await session.get(Section, section_id)


async def list_sections(
    session: AsyncSession,
    *,
    institution_id: uuid.UUID | None = None,
    department_id: uuid.UUID | None = None,
) -> list[Section]:
    stmt = select(Section).order_by(Section.name)
    if department_id is not None:
        stmt = stmt.where(Section.department_id == department_id)
    elif institution_id is not None:
        stmt = stmt.join(Department, Department.department_id == Section.department_id).where(
            Department.institution_id == institution_id
        )
    return list((await session.execute(stmt)).scalars())


# ─── Enrollment ──────────────────────────────────────────────────────────────


class StudentNotFoundError(Exception):
    pass


class StudentWrongInstitutionError(Exception):
    """Student exists but belongs to a different institution than the section."""


async def bulk_enroll(
    session: AsyncSession,
    *,
    section_id: uuid.UUID,
    student_ids: list[uuid.UUID],
) -> tuple[int, int]:
    """
    Enroll many students into one section (CSV import path).

    Skips students already enrolled (ON CONFLICT DO NOTHING semantics).
    Returns (enrolled_now, skipped_already_or_invalid).

    Raises StudentNotFoundError if NONE of the ids exist, and
    StudentWrongInstitutionError if any student belongs to another
    institution than the section's — cross-tenant enrollment must never
    happen silently.
    """
    section = await get_section_by_id(session, section_id)
    if section is None:
        raise StudentNotFoundError(f"section {section_id} does not exist")

    dept_stmt = select(Department.institution_id).where(
        Department.department_id == section.department_id
    )
    section_institution_id = (await session.execute(dept_stmt)).scalar_one()

    found: dict[uuid.UUID, uuid.UUID] = {}
    for sid in set(student_ids):
        inst_id = (
            await session.execute(select(Student.institution_id).where(Student.student_id == sid))
        ).scalar_one_or_none()
        if inst_id is not None:
            found[sid] = inst_id

    wrong_tenant = [sid for sid, inst in found.items() if inst != section_institution_id]
    if wrong_tenant:
        raise StudentWrongInstitutionError(
            f"{len(wrong_tenant)} student(s) belong to another institution"
        )

    enrolled = 0
    for sid in found:
        already = (
            await session.execute(
                select(func.count())
                .select_from(Enrollment)
                .where(
                    Enrollment.student_id == sid,
                    Enrollment.section_id == section_id,
                )
            )
        ).scalar_one()
        if already:
            continue
        session.add(
            Enrollment(
                enrollment_id=uuid.uuid4(),
                student_id=sid,
                section_id=section_id,
            )
        )
        enrolled += 1

    await session.flush()
    return enrolled, len(student_ids) - enrolled
