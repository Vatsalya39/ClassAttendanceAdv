"""
backend/core/security.py
────────────────────────
Password hashing (argon2id) and JWT issuing/verification.

JWT access tokens carry role + tenant-scope claims (institution_id /
department_id). Every admin route re-verifies these claims server-side —
they are NEVER trusted from client input.

Tokens are short-lived; refresh tokens are longer-lived and only accepted
by POST /auth/refresh.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.core.config import settings

# OWASP-recommended argon2id parameters (m=64 MiB, t=3, p=4 — comfortable on
# server hardware; verification of old hashes still works because the params
# are embedded in each hash string).
_pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__type="ID",  # argon2id
    argon2__time_cost=3,
    argon2__memory_cost=65536,
    argon2__parallelism=4,
)

TokenType = Literal["access", "refresh"]


class InvalidTokenError(Exception):
    """Raised when a JWT is missing/expired/has the wrong type."""


# ─── Passwords ────────────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    return str(_pwd_context.hash(plain))


def verify_password(plain: str, password_hash: str) -> bool:
    try:
        return bool(_pwd_context.verify(plain, password_hash))
    except Exception:
        # Malformed hash in the DB must read as auth failure, never 500.
        return False


# ─── JWT ──────────────────────────────────────────────────────────────────────


def _create_token(
    *,
    subject: str,
    token_type: TokenType,
    expires_delta: timedelta,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str, datetime]:
    now = datetime.now(UTC)
    exp = now + expires_delta
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "jti": uuid.uuid4().hex,
        **(extra_claims or {}),
    }
    encoded = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded, exp


def create_access_token(
    *,
    subject: str,
    token_type: Literal["admin", "student"] = "student",
    role: str | None = None,
    institution_id: uuid.UUID | None = None,
    department_id: uuid.UUID | None = None,
    minutes: int | None = None,
) -> tuple[str, datetime]:
    """
    Mint an access token.

    ``role`` / ``institution_id`` / ``department_id`` claims exist for admin
    tokens only. Student tokens carry subject = student_id.
    """
    claims: dict[str, Any] = {"principal_type": token_type}
    if role is not None:
        claims["role"] = role
    if institution_id is not None:
        claims["institution_id"] = str(institution_id)
    if department_id is not None:
        claims["department_id"] = str(department_id)
    return _create_token(
        subject=subject,
        token_type="access",
        expires_delta=timedelta(minutes=minutes or settings.jwt_access_token_expire_minutes),
        extra_claims=claims,
    )


def create_refresh_token(
    *, subject: str, principal_type: Literal["admin", "student"]
) -> tuple[str, datetime]:
    return _create_token(
        subject=subject,
        token_type="refresh",
        expires_delta=timedelta(days=settings.jwt_refresh_token_expire_days),
        extra_claims={"principal_type": principal_type},
    )


def decode_token(token: str, expected_type: TokenType = "access") -> dict[str, Any]:
    """
    Decode and validate a JWT. Raises InvalidTokenError on any problem
    (bad signature, expired, wrong type, missing claims).
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise InvalidTokenError(f"invalid token: {exc}") from exc

    if payload.get("type") != expected_type:
        raise InvalidTokenError(f"expected {expected_type} token")
    if "sub" not in payload:
        raise InvalidTokenError("token missing subject")
    return payload
