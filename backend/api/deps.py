"""
backend/api/deps.py
──────────────────
FastAPI dependency wiring: DB session, Redis, storage, queue backends and
JWT-authenticated principals with role/scope enforcement.

Scope rules (docs/architecture.md):
    super_admin        → everything
    institution_admin  → own institution only
    department_admin   → own department only
    faculty            → sections in own department

A department_admin for Dept A gets 403 touching Dept B's resources — the
checks below are the single enforcement point for that rule.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.security import InvalidTokenError, decode_token
from backend.db import dao
from backend.db.models import Admin, AdminRole, Student
from backend.db.session import get_db

_bearer = HTTPBearer(auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db)]


# ── App-state resources (redis / queue / storage live on app.state) ──────────


def get_redis(request: Request) -> Any:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return redis


def get_storage(request: Request) -> Any:
    storage = getattr(request.app.state, "storage", None)
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage unavailable")
    return storage


def get_queue(request: Request) -> Any:
    queue = getattr(request.app.state, "queue", None)
    if queue is None:
        raise HTTPException(status_code=503, detail="Queue unavailable")
    return queue


# ── Token extraction ─────────────────────────────────────────────────────────


async def _decode_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict[str, Any]:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        return decode_token(credentials.credentials, expected_type="access")
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


# ── Students ─────────────────────────────────────────────────────────────────


async def get_current_student(
    claims: Annotated[dict[str, Any], Depends(_decode_bearer)],
    session: DbSession,
) -> Student:
    if claims.get("principal_type") != "student":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Student token required")
    student = await dao.get_student(session, uuid.UUID(claims["sub"]))
    if student is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Student not found")
    return student


# ── Admins ───────────────────────────────────────────────────────────────────


async def get_current_admin(
    claims: Annotated[dict[str, Any], Depends(_decode_bearer)],
    session: DbSession,
) -> Admin:
    """
    Authenticate an admin token and re-verify its scope claims against the
    database row. The DB row is authoritative — a stale/revoked role in an
    old token cannot grant access.
    """
    if claims.get("principal_type") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required")
    admin = await session.get(Admin, uuid.UUID(claims["sub"]))
    if admin is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin not found")
    # Claims must match the stored row (defence against stale tokens).
    if str(admin.role.value) != str(claims.get("role")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Role changed — re-login"
        )
    if claims.get("institution_id") and admin.institution_id != uuid.UUID(
        str(claims["institution_id"])
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Scope changed — re-login"
        )
    if claims.get("department_id") and admin.department_id != uuid.UUID(
        str(claims["department_id"])
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Scope changed — re-login"
        )
    return admin


AdminUser = Annotated[Admin, Depends(get_current_admin)]
CurrentStudent = Annotated[Student, Depends(get_current_student)]
RedisDep = Annotated[Any, Depends(get_redis)]
StorageDep = Annotated[Any, Depends(get_storage)]
QueueDep = Annotated[Any, Depends(get_queue)]


# ── Scope enforcement helpers ────────────────────────────────────────────────


async def accessible_section_ids(session: AsyncSession, admin: Admin) -> list[uuid.UUID]:
    """Section ids this admin may act on ([] ⇒ nothing)."""
    return await dao.sections_accessible_to_admin(session, admin)


async def require_section_access(
    session: AsyncSession, admin: Admin, section_id: uuid.UUID
) -> tuple[Any, Any]:
    """
    Return (section, department) after verifying the admin may touch it.
    Raises 404 for unknown sections and 403 for out-of-scope ones.
    """
    pair = await dao.get_section_with_parents(session, section_id)
    if pair is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section not found")
    section, department = pair
    allowed = False
    if admin.role == AdminRole.super_admin:
        allowed = True
    elif admin.role == AdminRole.institution_admin:
        allowed = admin.institution_id == department.institution_id
    elif admin.role in (AdminRole.department_admin, AdminRole.faculty):
        allowed = admin.department_id == section.department_id
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Section outside your scope"
        )
    return section, department


async def require_role(admin: Admin, *roles: AdminRole) -> None:
    if admin.role not in roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")


# ── Rate limiting dependency ─────────────────────────────────────────────────


async def enforce_rate_limit(
    request: Request,
    redis: RedisDep,
    *,
    name: str,
    limit: int,
    window_seconds: int = 60,
) -> None:
    from backend.core.ratelimit import RateLimitExceededError, check_rate_limit

    client_ip = request.client.host if request.client else "unknown"
    try:
        await check_rate_limit(
            redis, name=f"{name}:{client_ip}", limit=limit, window_seconds=window_seconds
        )
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
