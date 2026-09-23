"""
backend/api/routes/auth.py
──────────────────────────
Student and admin login, plus refresh-token exchange.

Rate limited per IP on settings.rate_limit_auth_rpm.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request, status

from backend.api.deps import DbSession, RedisDep, enforce_rate_limit
from backend.api.schemas import AdminLoginRequest, RefreshRequest, StudentLoginRequest, TokenPair
from backend.core.config import settings
from backend.core.security import (
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from backend.db import dao
from backend.db.models import Admin, Student

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_pair_for_admin(admin: Admin) -> TokenPair:
    access, _ = create_access_token(
        subject=str(admin.admin_id),
        token_type="admin",
        role=admin.role.value,
        institution_id=admin.institution_id,
        department_id=admin.department_id,
    )
    refresh, _ = create_refresh_token(subject=str(admin.admin_id), principal_type="admin")
    return TokenPair(access_token=access, refresh_token=refresh)


def _token_pair_for_student(student: Student) -> TokenPair:
    access, _ = create_access_token(subject=str(student.student_id), token_type="student")
    refresh, _ = create_refresh_token(subject=str(student.student_id), principal_type="student")
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/student/login", response_model=TokenPair)
async def student_login(
    request: Request,
    body: StudentLoginRequest,
    session: DbSession,
    redis: RedisDep,
) -> TokenPair:
    await enforce_rate_limit(request, redis, name="auth", limit=settings.rate_limit_auth_rpm)
    student = await dao.get_student_by_roll_number(session, body.roll_number)
    if student is None or not verify_password(body.password, student.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return _token_pair_for_student(student)


@router.post("/admin/login", response_model=TokenPair)
async def admin_login(
    request: Request,
    body: AdminLoginRequest,
    session: DbSession,
    redis: RedisDep,
) -> TokenPair:
    await enforce_rate_limit(request, redis, name="auth", limit=settings.rate_limit_auth_rpm)
    admin = await dao.get_admin_by_username(session, body.username)
    if admin is None or not verify_password(body.password, admin.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return _token_pair_for_admin(admin)


@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshRequest, session: DbSession) -> TokenPair:
    try:
        claims = decode_token(body.refresh_token, expected_type="refresh")
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    principal_type = claims.get("principal_type")
    subject = uuid.UUID(str(claims["sub"]))

    if principal_type == "admin":
        admin = await session.get(Admin, subject)
        if admin is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account deleted")
        return _token_pair_for_admin(admin)

    if principal_type == "student":
        student = await session.get(Student, subject)
        if student is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account deleted")
        return _token_pair_for_student(student)

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown principal type")
