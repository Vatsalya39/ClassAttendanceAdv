"""
backend/db/dao/registration_tokens.py
─────────────────────────────────────
DAO layer for issuing and validating student registration tokens.
Enforces that face embeddings are only registered for students whose
identities have been authorized by institution administrators.
"""

from __future__ import annotations

import datetime
import secrets
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import select

from backend.db.models import RegistrationToken

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create_registration_token(
    session: AsyncSession,
    *,
    roll_number: str,
    institution_id: uuid.UUID,
    expires_in_days: int = 30,
) -> RegistrationToken:
    """
    Issue a single-use registration token for a specific student roll number.
    Admin-only operation during student pre-seeding / enrollment onboarding.
    """
    token_str = secrets.token_urlsafe(32)
    now = datetime.datetime.now(datetime.UTC)
    expires_at = now + datetime.timedelta(days=expires_in_days)

    reg_token = RegistrationToken(
        token_id=uuid.uuid4(),
        token=token_str,
        roll_number=roll_number.strip(),
        institution_id=institution_id,
        expires_at=expires_at,
        is_used=False,
    )
    session.add(reg_token)
    await session.flush()
    return reg_token


async def verify_and_consume_registration_token(
    session: AsyncSession,
    *,
    roll_number: str,
    token: str,
    institution_id: uuid.UUID,
) -> bool:
    """
    Atomically verify that a registration token is valid, unexpired, matches the
    roll number and institution, and mark it as consumed (single-use).
    Returns True if successfully verified and consumed, False otherwise.
    """
    now = datetime.datetime.now(datetime.UTC)
    stmt = (
        select(RegistrationToken)
        .where(
            RegistrationToken.token == token.strip(),
            RegistrationToken.roll_number == roll_number.strip(),
            RegistrationToken.institution_id == institution_id,
            RegistrationToken.is_used.is_(False),
            RegistrationToken.expires_at > now,
        )
        .with_for_update()
    )
    res = await session.execute(stmt)
    token_row = res.scalar_one_or_none()
    if token_row is None:
        return False

    token_row.is_used = True
    token_row.used_at = now
    await session.flush()
    return True
