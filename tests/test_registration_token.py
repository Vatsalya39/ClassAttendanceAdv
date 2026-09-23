"""
tests/test_registration_token.py
─────────────────────────────────
Unit tests for Phase B1: Single-use Registration Token Verification.
Verifies that student registration cannot occur without a valid, unexpired,
single-use registration token tied to the student's roll_number and institution_id.
"""

from __future__ import annotations

import datetime
import uuid
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.core.config import settings
from backend.db.dao.registration_tokens import (
    create_registration_token,
    verify_and_consume_registration_token,
)
from backend.db.models import Department, RegistrationToken, Section


class TestTokenCreationAndConsumptionDAO:
    @pytest.mark.asyncio
    async def test_create_registration_token_dao(self) -> None:
        """create_registration_token generates secure token and persists with expiration."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        inst_id = uuid.uuid4()
        roll = "25BLC1105"

        reg_token = await create_registration_token(
            mock_session,
            roll_number=roll,
            institution_id=inst_id,
            expires_in_days=30,
        )

        assert len(reg_token.token) >= 32
        assert reg_token.institution_id == inst_id
        assert reg_token.roll_number == roll
        assert reg_token.is_used is False
        assert reg_token.used_at is None
        assert reg_token.expires_at > datetime.datetime.now(datetime.UTC)
        mock_session.add.assert_called_once_with(reg_token)
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_valid_token_consumed_successfully(self) -> None:
        """A valid unexpired token is consumed (is_used=True, used_at set) and returns True."""
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()
        inst_id = uuid.uuid4()
        token_str = "sec_test_token_abc123"

        token_row = RegistrationToken(
            token_id=uuid.uuid4(),
            token=token_str,
            institution_id=inst_id,
            roll_number="25BLC1105",
            expires_at=datetime.datetime.now(datetime.UTC) + timedelta(days=5),
            is_used=False,
            used_at=None,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = token_row
        mock_session.execute.return_value = mock_result

        success = await verify_and_consume_registration_token(
            mock_session,
            roll_number="25BLC1105",
            token=token_str,
            institution_id=inst_id,
        )

        assert success is True
        assert token_row.is_used is True
        assert token_row.used_at is not None
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_nonexistent_or_unmatched_token_rejected(self) -> None:
        """If query returns None (due to mismatched token, roll number, expired, or already used), return False."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        success = await verify_and_consume_registration_token(
            mock_session,
            roll_number="25BLC1105",
            token="invalid_token",
            institution_id=uuid.uuid4(),
        )

        assert success is False
        mock_session.flush.assert_not_called()


class TestRegisterEndpointEnforcement:
    """Tests the FastAPI route handler enforcement in backend/api/routes/register.py."""

    @pytest.mark.asyncio
    async def test_register_missing_token_when_required_raises_403(self) -> None:
        from backend.api.routes.register import register

        mock_request = MagicMock()
        mock_session = AsyncMock()
        mock_redis = AsyncMock()
        mock_storage = MagicMock()

        mock_file = MagicMock()
        mock_file.filename = "face.jpg"
        mock_file.content_type = "image/jpeg"

        inst_id = uuid.uuid4()
        sec_id = uuid.uuid4()
        dept_id = uuid.uuid4()

        mock_sec = Section(
            section_id=sec_id, department_id=dept_id, name="CSE-A", academic_year=2026
        )
        mock_dept = Department(department_id=dept_id, institution_id=inst_id, name="CSE")

        with (
            patch("backend.api.routes.register.enforce_rate_limit", new=AsyncMock()),
            patch(
                "backend.api.routes.register.dao.get_section_with_parents",
                new=AsyncMock(return_value=(mock_sec, mock_dept)),
            ),
            patch.object(settings, "require_registration_token", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await register(
                request=mock_request,
                session=mock_session,
                redis=mock_redis,
                storage=mock_storage,
                roll_number="25BLC1105",
                name="Alice",
                password="password123",
                section_id=sec_id,
                images=[mock_file, mock_file, mock_file],
                registration_token="",  # EMPTY
            )
        assert exc_info.value.status_code == 403
        assert "Registration token is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_register_invalid_token_raises_403(self) -> None:
        from backend.api.routes.register import register

        mock_request = MagicMock()
        mock_session = AsyncMock()
        mock_redis = AsyncMock()
        mock_storage = MagicMock()

        mock_file = MagicMock()
        mock_file.filename = "face.jpg"
        mock_file.content_type = "image/jpeg"

        inst_id = uuid.uuid4()
        sec_id = uuid.uuid4()
        dept_id = uuid.uuid4()

        mock_sec = Section(
            section_id=sec_id, department_id=dept_id, name="CSE-A", academic_year=2026
        )
        mock_dept = Department(department_id=dept_id, institution_id=inst_id, name="CSE")

        with (
            patch("backend.api.routes.register.enforce_rate_limit", new=AsyncMock()),
            patch(
                "backend.api.routes.register.dao.get_section_with_parents",
                new=AsyncMock(return_value=(mock_sec, mock_dept)),
            ),
            patch(
                "backend.api.routes.register.dao.verify_and_consume_registration_token",
                new=AsyncMock(return_value=False),
            ),
            patch.object(settings, "require_registration_token", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await register(
                request=mock_request,
                session=mock_session,
                redis=mock_redis,
                storage=mock_storage,
                roll_number="25BLC1105",
                name="Alice",
                password="password123",
                section_id=sec_id,
                images=[mock_file, mock_file, mock_file],
                registration_token="invalid_or_consumed_token",
            )
        assert exc_info.value.status_code == 403
        assert "Invalid, expired, or already consumed" in exc_info.value.detail
