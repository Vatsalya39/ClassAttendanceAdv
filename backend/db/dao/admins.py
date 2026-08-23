"""
backend/db/dao/admins.py
────────────────────────
Admin accounts, audit logging, and role-scope resolution.

Role → accessible sections (docs/architecture.md):
    super_admin        → ALL sections
    institution_admin  → sections under their institution
    department_admin   → sections under their department
    faculty            → sections under their department (per-section
                         assignment is a future refinement)
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import select

from backend.db.models import Admin, AdminRole, AuditLog, Department, Section

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create_admin(
    session: AsyncSession,
    *,
    username: str,
    password_hash: str,
    role: AdminRole,
    institution_id: uuid.UUID | None = None,
    department_id: uuid.UUID | None = None,
) -> Admin:
    admin = Admin(
        admin_id=uuid.uuid4(),
        username=username,
        password_hash=password_hash,
        role=role,
        institution_id=institution_id,
        department_id=department_id,
    )
    session.add(admin)
    await session.flush()
    return admin


async def get_admin_by_username(session: AsyncSession, username: str) -> Admin | None:
    stmt = select(Admin).where(Admin.username == username)
    return (await session.execute(stmt)).scalar_one_or_none()


async def get_section_with_parents(
    session: AsyncSession, section_id: uuid.UUID
) -> tuple[Section, Department] | None:
    """Section row plus its department (for tenant-scope checks)."""
    stmt = (
        select(Section, Department)
        .join(Department, Department.department_id == Section.department_id)
        .where(Section.section_id == section_id)
    )
    row = (await session.execute(stmt)).first()
    return (row[0], row[1]) if row else None


async def sections_accessible_to_admin(session: AsyncSession, admin: Admin) -> list[uuid.UUID]:
    """
    Section ids the admin may act on. Empty list means NO access;
    callers must distinguish "no access" from "access to everything"
    (super_admin returns the actual full list — bounded by org size and
    only used for listing endpoints).
    """
    stmt = select(Section.section_id).join(
        Department, Department.department_id == Section.department_id
    )
    if admin.role == AdminRole.super_admin:
        stmt = stmt
    elif admin.role == AdminRole.institution_admin:
        if admin.institution_id is None:
            return []
        stmt = stmt.where(Department.institution_id == admin.institution_id)
    elif admin.role in (AdminRole.department_admin, AdminRole.faculty):
        if admin.department_id is None:
            return []
        stmt = stmt.where(Section.department_id == admin.department_id)
    else:  # pragma: no cover — enum is closed
        return []
    return list((await session.execute(stmt)).scalars())


async def audit_action(
    session: AsyncSession,
    *,
    admin_id: uuid.UUID | None,
    action: str,
    target_type: str,
    target_id: uuid.UUID | None = None,
    detail: str | None = None,
) -> AuditLog:
    row = AuditLog(
        log_id=uuid.uuid4(),
        admin_id=admin_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=(detail[:2048] if detail else None),
    )
    session.add(row)
    await session.flush()
    return row
