"""registration_tokens table — student identity verification gate

Revision ID: 0003_registration_tokens
Revises: 0002_attendance_anonymizable
Create Date: 2026-09-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0003_registration_tokens"
down_revision = "0002_attendance_anonymizable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "registration_tokens",
        sa.Column("token_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("token", sa.String(64), nullable=False, unique=True),
        sa.Column("roll_number", sa.String(64), nullable=False),
        sa.Column(
            "institution_id",
            UUID(as_uuid=True),
            sa.ForeignKey("institutions.institution_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_used", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_registration_tokens_token",
        "registration_tokens",
        ["token"],
        unique=True,
    )
    op.create_index(
        "ix_registration_tokens_roll_number",
        "registration_tokens",
        ["roll_number"],
    )


def downgrade() -> None:
    op.drop_table("registration_tokens")
