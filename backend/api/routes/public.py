"""
backend/api/routes/public.py
────────────────────────────
Unauthenticated read-only endpoints needed by the public-facing frontend:

    GET /public/sections — section list (ids + display names only) so the
    registration form can offer a section picker without exposing any
    student data. No personal data is ever returned here.
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from backend.api.deps import DbSession
from backend.api.schemas import PublicSectionOut
from backend.db.models import Department, Institution, Section

router = APIRouter(prefix="/public", tags=["public"])


@router.get("/sections", response_model=list[PublicSectionOut])
async def list_public_sections(session: DbSession) -> list[PublicSectionOut]:
    stmt = (
        select(
            Section.section_id,
            Section.name,
            Section.academic_year,
            Department.name.label("department_name"),
            Institution.name.label("institution_name"),
        )
        .join(Department, Department.department_id == Section.department_id)
        .join(Institution, Institution.institution_id == Department.institution_id)
        .order_by(Institution.name, Department.name, Section.name)
    )
    rows = (await session.execute(stmt)).all()
    return [
        PublicSectionOut(
            section_id=r.section_id,
            name=r.name,
            academic_year=r.academic_year,
            department_name=r.department_name,
            institution_name=r.institution_name,
        )
        for r in rows
    ]
