"""attendance.student_id nullable — supports biometric anonymization

Phase 5 data-retention requirement (docs/data-retention.md): on a deletion
request, attendance history may be ANONYMISED instead of deleted so
institution-level statistics survive. That requires student_id to be NULLable.

Revision ID: 0002_attendance_anonymizable
Revises: 0001_initial
"""

from __future__ import annotations

from alembic import op

revision = "0002_attendance_anonymizable"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Propagates to all monthly partitions automatically.
    op.execute("ALTER TABLE attendance ALTER COLUMN student_id DROP NOT NULL")


def downgrade() -> None:
    # Only safe if no anonymised (NULL) rows remain.
    op.execute("DELETE FROM attendance WHERE student_id IS NULL")
    op.execute("ALTER TABLE attendance ALTER COLUMN student_id SET NOT NULL")
