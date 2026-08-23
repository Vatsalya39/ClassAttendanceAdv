"""
backend/api/routes/org.py
─────────────────────────
Institution / department / section CRUD, admin-account creation, and CSV
bulk enrollment — all role- and scope-checked:

    institutions            → super_admin only
    departments             → super_admin or own-institution institution_admin
    sections                → super_admin, institution_admin (own inst),
                              department_admin (own dept)
    department_admin accounts → super_admin / institution_admin
    bulk-enroll (CSV)       → any admin with access to the target section
"""

from __future__ import annotations

import csv
import io
import uuid

from fastapi import APIRouter, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import AdminUser, DbSession, require_role, require_section_access
from backend.api.schemas import (
    AdminCreateRequest,
    BulkEnrollResponse,
    DepartmentCreate,
    DepartmentOut,
    InstitutionCreate,
    InstitutionOut,
    RosterStudentOut,
    SectionCreate,
    SectionOut,
)
from backend.core.security import hash_password
from backend.db import dao
from backend.db.models import AdminRole, Department, Enrollment, Student

router = APIRouter(prefix="/admin", tags=["org"])


async def _audit(
    session: AsyncSession,
    admin: AdminUser,
    action: str,
    target_type: str,
    target_id: uuid.UUID | None,
    detail: str = "",
) -> None:
    await dao.audit_action(
        session,
        admin_id=admin.admin_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=detail,
    )


# ─── Institutions (super_admin) ───────────────────────────────────────────────


@router.post("/institutions", response_model=InstitutionOut)
async def create_institution(
    body: InstitutionCreate, session: DbSession, admin: AdminUser
) -> InstitutionOut:
    await require_role(admin, AdminRole.super_admin)
    inst = await dao.create_institution(session, name=body.name.strip())
    await _audit(
        session, admin, "institution.created", "institution", inst.institution_id, body.name
    )
    await session.commit()
    return InstitutionOut(institution_id=inst.institution_id, name=inst.name)


@router.get("/institutions", response_model=list[InstitutionOut])
async def list_institutions(session: DbSession, admin: AdminUser) -> list[InstitutionOut]:
    if admin.role == AdminRole.super_admin:
        rows = await dao.list_institutions(session)
    elif admin.institution_id is not None:
        rows = [
            i
            for i in await dao.list_institutions(session)
            if i.institution_id == admin.institution_id
        ]
    else:
        rows = []
    return [InstitutionOut(institution_id=i.institution_id, name=i.name) for i in rows]


# ─── Departments ──────────────────────────────────────────────────────────────


@router.post("/departments", response_model=DepartmentOut)
async def create_department(
    body: DepartmentCreate, session: DbSession, admin: AdminUser
) -> DepartmentOut:
    await require_role(admin, AdminRole.super_admin, AdminRole.institution_admin)
    # An institution_admin may only create departments in their OWN institution.
    if admin.role == AdminRole.institution_admin and admin.institution_id != body.institution_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Institution outside your scope"
        )
    dept = await dao.create_department(
        session, institution_id=body.institution_id, name=body.name.strip()
    )
    await _audit(session, admin, "department.created", "department", dept.department_id, body.name)
    await session.commit()
    return DepartmentOut(
        department_id=dept.department_id, institution_id=dept.institution_id, name=dept.name
    )


@router.get("/departments", response_model=list[DepartmentOut])
async def list_departments(
    session: DbSession,
    admin: AdminUser,
    institution_id: uuid.UUID | None = None,
) -> list[DepartmentOut]:
    effective_inst = institution_id
    if admin.role == AdminRole.institution_admin:
        effective_inst = admin.institution_id
    elif admin.role == AdminRole.super_admin and institution_id is None:
        effective_inst = None
    elif (
        admin.role in (AdminRole.department_admin, AdminRole.faculty)
        and admin.department_id is not None
    ):
        dept = await session.get(Department, admin.department_id)
        effective_inst = dept.institution_id if dept else None

    rows = await dao.list_departments(session, institution_id=effective_inst)
    return [
        DepartmentOut(department_id=d.department_id, institution_id=d.institution_id, name=d.name)
        for d in rows
    ]


# ─── Sections ────────────────────────────────────────────────────────────────


@router.post("/sections", response_model=SectionOut)
async def create_section(body: SectionCreate, session: DbSession, admin: AdminUser) -> SectionOut:
    pair_dept = await session.get(Department, body.department_id)
    if pair_dept is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Department not found")

    allowed = False
    if admin.role == AdminRole.super_admin:
        allowed = True
    elif admin.role == AdminRole.institution_admin:
        allowed = admin.institution_id == pair_dept.institution_id
    elif admin.role == AdminRole.department_admin:
        allowed = admin.department_id == pair_dept.department_id
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Department outside your scope"
        )

    section = await dao.create_section(
        session,
        department_id=body.department_id,
        name=body.name.strip(),
        academic_year=body.academic_year,
    )
    await _audit(session, admin, "section.created", "section", section.section_id, section.name)
    await session.commit()
    return SectionOut(
        section_id=section.section_id,
        department_id=section.department_id,
        name=section.name,
        academic_year=section.academic_year,
    )


@router.get("/sections", response_model=list[SectionOut])
async def list_sections(
    session: DbSession,
    admin: AdminUser,
    department_id: uuid.UUID | None = None,
    institution_id: uuid.UUID | None = None,
) -> list[SectionOut]:
    if admin.role == AdminRole.super_admin:
        rows = await dao.list_sections(
            session, department_id=department_id, institution_id=institution_id
        )
    elif admin.role == AdminRole.institution_admin:
        rows = await dao.list_sections(session, institution_id=admin.institution_id)
        if department_id is not None:
            rows = [s for s in rows if s.department_id == department_id]
    else:  # department_admin / faculty → own department only
        rows = await dao.list_sections(session, department_id=admin.department_id)
    return [
        SectionOut(
            section_id=s.section_id,
            department_id=s.department_id,
            name=s.name,
            academic_year=s.academic_year,
        )
        for s in rows
    ]


@router.get("/sections/{section_id}/students", response_model=list[RosterStudentOut])
async def list_section_students(
    section_id: uuid.UUID,
    session: DbSession,
    admin: AdminUser,
) -> list[RosterStudentOut]:
    """Enrolled roster for one section (used by the review-queue assign UI)."""
    await require_section_access(session, admin, section_id)
    stmt = (
        select(Student.student_id, Student.roll_number, Student.name)
        .join(Enrollment, Enrollment.student_id == Student.student_id)
        .where(Enrollment.section_id == section_id)
        .order_by(Student.roll_number)
    )
    rows = (await session.execute(stmt)).all()
    return [
        RosterStudentOut(student_id=r.student_id, roll_number=r.roll_number, name=r.name)
        for r in rows
    ]


# ─── Bulk enroll (CSV of roll numbers) ────────────────────────────────────────


@router.post("/sections/{section_id}/bulk-enroll", response_model=BulkEnrollResponse)
async def bulk_enroll(
    section_id: uuid.UUID,
    session: DbSession,
    admin: AdminUser,
    file: UploadFile,
    has_header: bool = Query(default=False),
) -> BulkEnrollResponse:
    """Upload a single-column CSV of roll numbers to enroll into a section."""
    await require_section_access(session, admin, section_id)

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="CSV must be UTF-8"
        ) from exc

    roll_numbers = [
        row[0].strip() for row in csv.reader(io.StringIO(text)) if row and row[0].strip()
    ]
    if has_header and roll_numbers:
        roll_numbers = roll_numbers[1:]
    if not roll_numbers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No roll numbers found in CSV"
        )

    student_ids: list[uuid.UUID] = []
    missing: list[str] = []
    for rn in roll_numbers[:5000]:
        student = await dao.get_student_by_roll_number(session, rn)
        if student is None:
            missing.append(rn)
        else:
            student_ids.append(student.student_id)

    enrolled = skipped_before = 0
    wrong_tenant = False
    try:
        enrolled, skipped_before = await dao.bulk_enroll(
            session, section_id=section_id, student_ids=student_ids
        )
    except dao.StudentWrongInstitutionError:
        wrong_tenant = True
    except dao.StudentNotFoundError as exc:  # pragma: no cover — section checked above
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    await _audit(
        session,
        admin,
        "section.bulk_enrolled",
        "section",
        section_id,
        f"rows={len(roll_numbers)} enrolled={enrolled} missing={len(missing)}",
    )
    if wrong_tenant:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSV contains students from another institution — nothing was enrolled",
        )
    await session.commit()
    skipped_total = len(missing) + skipped_before
    return BulkEnrollResponse(section_id=section_id, enrolled=enrolled, skipped=skipped_total)


# ─── Admin accounts ──────────────────────────────────────────────────────────


@router.post("/admins")
async def create_admin(
    body: AdminCreateRequest, session: DbSession, admin: AdminUser
) -> dict[str, uuid.UUID | str]:
    role = AdminRole(body.role)

    if admin.role == AdminRole.super_admin:
        pass  # may create any scoped admin
    elif admin.role == AdminRole.institution_admin:
        if body.institution_id != admin.institution_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Institution outside your scope"
            )
        if (
            role == AdminRole.faculty
            and body.department_id is not None
            and body.department_id != admin.department_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Department outside your scope"
            )
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")

    if body.institution_id is None and role != AdminRole.institution_admin:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="institution_id required"
        )
    if role in (AdminRole.department_admin, AdminRole.faculty) and body.department_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="department_id required for this role",
        )

    existing = await dao.get_admin_by_username(session, body.username)
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username taken")

    created = await dao.create_admin(
        session,
        username=body.username.strip(),
        password_hash=hash_password(body.password),
        role=role,
        institution_id=body.institution_id,
        department_id=body.department_id,
    )
    await _audit(session, admin, "admin.created", "admin", created.admin_id, f"role={role.value}")
    await session.commit()
    return {"admin_id": created.admin_id, "username": created.username, "role": created.role.value}
