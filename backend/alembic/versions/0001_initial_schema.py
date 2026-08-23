"""initial schema — org hierarchy, students, pgvector embeddings, partitioned attendance

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-23
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811

from backend.db.models import EMBEDDING_DIM, AdminRole, ReviewStatus
from backend.db.partitions import ensure_partitions

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── pgvector ──────────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Organisational hierarchy ──────────────────────────────────────────────
    op.create_table(
        "institutions",
        sa.Column("institution_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "departments",
        sa.Column("department_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column(
            "institution_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("institutions.institution_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("institution_id", "name", name="uq_departments_inst_name"),
    )
    op.create_table(
        "sections",
        sa.Column("section_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column(
            "department_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("departments.department_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("academic_year", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "admins",
        sa.Column("admin_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(100), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column(
            "role",
            sa.Enum(AdminRole, name="admin_role", native_enum=True),
            nullable=False,
        ),
        sa.Column(
            "institution_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("institutions.institution_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "department_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("departments.department_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Students & embeddings ─────────────────────────────────────────────────
    op.create_table(
        "students",
        sa.Column("student_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column("roll_number", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column(
            "institution_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("institutions.institution_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "face_embeddings",
        sa.Column("embedding_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column(
            "student_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("students.student_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("source_image_url", sa.String(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_face_embeddings_student_id", "face_embeddings", ["student_id"])
    #
    # THE single HNSW index is sufficient at 50k+ students because it is
    # NEVER queried unfiltered: find_nearest_embeddings_for_roster() joins
    # face_embeddings → enrollments ON section_id BEFORE the cosine-distance
    # ordering, so only that section's ~60-70 students' embeddings (a few
    # hundred rows) are ever ranked. The one institution-wide search is an
    # explicitly-named, rarely-used admin fallback.
    #
    op.execute(
        "CREATE INDEX ix_face_embeddings_embedding_hnsw "
        "ON face_embeddings USING hnsw (embedding vector_cosine_ops)"
    )
    op.create_table(
        "enrollments",
        sa.Column("enrollment_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column(
            "student_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("students.student_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("sections.section_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("enrolled_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("student_id", "section_id", name="uq_enrollments_student_section"),
    )
    op.create_index("ix_enrollments_section_id", "enrollments", ["section_id"])

    # ── Sessions & attendance (partitioned) ───────────────────────────────────
    op.create_table(
        "class_sessions",
        sa.Column("session_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column(
            "section_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("sections.section_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("course_name", sa.String(255), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column(
            "admin_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("admins.admin_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_class_sessions_section_date",
        "class_sessions",
        ["section_id", "session_date"],
    )
    # PARTITION BY RANGE (session_date) — monthly partitions created below and
    # by the scheduled maintenance job. PK/UNIQUE must include the partition
    # key (Postgres rule); session_id ⇒ session_date is 1:1 so idempotency of
    # (student_id, session_id) is preserved.
    op.create_table(
        "attendance",
        sa.Column("attendance_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column("session_date", sa.Date(), primary_key=True),
        sa.Column(
            "student_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("students.student_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("class_sessions.session_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("match_distance", sa.Float(), nullable=True),
        sa.Column("marked_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "student_id",
            "session_id",
            "session_date",
            name="uq_attendance_student_session",
        ),
        postgresql_partition_by="RANGE (session_date)",
    )
    op.create_index("ix_attendance_session_id", "attendance", ["session_id"])
    ensure_partitions(op.get_bind(), months_ahead=3)

    # ── Audit & review queue ──────────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("log_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column(
            "admin_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("admins.admin_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("target_type", sa.String(100), nullable=False),
        sa.Column("target_id", PgUUID(as_uuid=True), nullable=True),
        sa.Column("detail", sa.String(2048), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])
    op.create_index("ix_audit_log_admin_id", "audit_log", ["admin_id"])

    op.create_table(
        "review_queue",
        sa.Column("review_id", PgUUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("class_sessions.session_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("face_crop_url", sa.String(1024), nullable=True),
        sa.Column("face_embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "best_guess_student_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("students.student_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("best_guess_distance", sa.Float(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(ReviewStatus, name="review_status", native_enum=True),
            nullable=False,
        ),
        sa.Column(
            "resolved_student_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("students.student_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "resolved_by_admin_id",
            PgUUID(as_uuid=True),
            sa.ForeignKey("admins.admin_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_review_queue_session_id", "review_queue", ["session_id"])
    op.create_index("ix_review_queue_status", "review_queue", ["status"])


def downgrade() -> None:
    op.drop_index("ix_review_queue_status", table_name="review_queue")
    op.drop_index("ix_review_queue_session_id", table_name="review_queue")
    op.drop_table("review_queue")
    op.drop_index("ix_audit_log_admin_id", table_name="audit_log")
    op.drop_index("ix_audit_log_created_at", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_attendance_session_id", table_name="attendance")
    op.execute("DROP TABLE IF EXISTS attendance CASCADE")  # drops partitions too
    op.drop_table("class_sessions")
    op.drop_index("ix_enrollments_section_id", table_name="enrollments")
    op.drop_table("enrollments")
    op.execute("DROP INDEX IF EXISTS ix_face_embeddings_embedding_hnsw")
    op.drop_index("ix_face_embeddings_student_id", table_name="face_embeddings")
    op.drop_table("face_embeddings")
    op.drop_table("students")
    op.drop_table("admins")
    op.drop_table("sections")
    op.drop_table("departments")
    op.drop_table("institutions")
    op.execute("DROP TYPE IF EXISTS admin_role")
    op.execute("DROP TYPE IF EXISTS review_status")
