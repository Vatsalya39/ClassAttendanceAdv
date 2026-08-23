"""
backend/db/models.py
────────────────────
SQLAlchemy models for the multi-campus attendance system.

Organisational hierarchy:
    Institution → Department → Section (~60-70 students) → Class Session

THE key design rule encoded in the DAO layer (backend/db/dao/matching.py):
face matching is ALWAYS roster-scoped via a join to ``enrollments`` filtered
by section_id — never a global vector search.

Partitioning note
─────────────────
``attendance`` is PARTITIONED BY RANGE (session_date), one monthly partition.
PostgreSQL requires every UNIQUE constraint / PK on a partitioned table to
include the partition key, so:
    PRIMARY KEY (attendance_id, session_date)
    UNIQUE      (student_id, session_id, session_date)
The extra session_date column does not weaken the idempotency guarantee:
session_id already determines session_date 1:1 (see class_sessions), so
(student_id, session_id) collisions are still impossible.
Individual partitions are created by backend.db.partitions.ensure_partitions().
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime  # noqa: TCH003 — needed at runtime by the mapper

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base

EMBEDDING_DIM = 512


class AdminRole(str, enum.Enum):
    super_admin = "super_admin"
    institution_admin = "institution_admin"
    department_admin = "department_admin"
    faculty = "faculty"


class ReviewStatus(str, enum.Enum):
    pending = "pending"
    resolved = "resolved"


# ─── Organisational hierarchy ─────────────────────────────────────────────────


class Institution(Base):
    __tablename__ = "institutions"

    institution_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    departments: Mapped[list[Department]] = relationship(back_populates="institution")
    students: Mapped[list[Student]] = relationship(back_populates="institution")


class Department(Base):
    __tablename__ = "departments"
    __table_args__ = (UniqueConstraint("institution_id", "name", name="uq_departments_inst_name"),)

    department_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    institution_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("institutions.institution_id", ondelete="CASCADE"),
    )
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    institution: Mapped[Institution] = relationship(back_populates="departments")
    sections: Mapped[list[Section]] = relationship(back_populates="department")


class Section(Base):
    """The ~60-70-student cluster that face matching is always scoped to."""

    __tablename__ = "sections"

    section_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    department_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("departments.department_id", ondelete="CASCADE"),
    )
    name: Mapped[str] = mapped_column(String(100))  # e.g. "CSE-3B"
    academic_year: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    department: Mapped[Department] = relationship(back_populates="sections")
    enrollments: Mapped[list[Enrollment]] = relationship(back_populates="section")
    class_sessions: Mapped[list[ClassSession]] = relationship(back_populates="section")


class Enrollment(Base):
    __tablename__ = "enrollments"
    __table_args__ = (
        UniqueConstraint("student_id", "section_id", name="uq_enrollments_student_section"),
        Index("ix_enrollments_section_id", "section_id"),
    )

    enrollment_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("students.student_id", ondelete="CASCADE"),
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("sections.section_id", ondelete="CASCADE"),
    )
    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    student: Mapped[Student] = relationship(back_populates="enrollments")
    section: Mapped[Section] = relationship(back_populates="enrollments")


# ─── Core entities ────────────────────────────────────────────────────────────


class Student(Base):
    __tablename__ = "students"

    student_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    roll_number: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(255))
    institution_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("institutions.institution_id", ondelete="CASCADE"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    institution: Mapped[Institution] = relationship(back_populates="students")
    enrollments: Mapped[list[Enrollment]] = relationship(back_populates="student")
    embeddings: Mapped[list[FaceEmbedding]] = relationship(back_populates="student")


class FaceEmbedding(Base):
    """
    One ArcFace embedding per enrolment image; 3-5 rows per student.

    The single HNSW index on ``embedding`` is sufficient at 50k+ students
    because it is NEVER queried unfiltered — see
    find_nearest_embeddings_for_roster() in backend/db/dao/matching.py,
    which joins to enrollments on section_id before any vector ordering.
    """

    __tablename__ = "face_embeddings"

    embedding_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("students.student_id", ondelete="CASCADE"),
    )
    # Cosine distance ops (<=>) — matches match_embedding()'s distance metric.
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    source_image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_face_embeddings_student_id", "student_id"),
        Index(
            "ix_face_embeddings_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    student: Mapped[Student] = relationship(back_populates="embeddings")


class ClassSession(Base):
    __tablename__ = "class_sessions"
    __table_args__ = (Index("ix_class_sessions_section_date", "section_id", "session_date"),)

    session_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("sections.section_id", ondelete="CASCADE"),
    )
    course_name: Mapped[str] = mapped_column(String(255))
    session_date: Mapped[date] = mapped_column(Date)
    admin_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("admins.admin_id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    section: Mapped[Section] = relationship(back_populates="class_sessions")
    attendance_rows: Mapped[list[Attendance]] = relationship(back_populates="class_session")


class Attendance(Base):
    """
    Partitioned monthly by RANGE (session_date) — ~10M rows/year at 50k
    students. See module docstring for why the PK / UNIQUE include
    session_date.
    """

    __tablename__ = "attendance"
    __table_args__ = (
        UniqueConstraint(
            "student_id",
            "session_id",
            "session_date",
            name="uq_attendance_student_session",
        ),
        Index("ix_attendance_session_id", "session_id"),
        {"postgresql_partition_by": "RANGE (session_date)"},
    )

    attendance_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_date: Mapped[date] = mapped_column(Date, primary_key=True)  # partition key
    # Nullable: anonymised on biometric-purge requests (migration 0002) so
    # institution-level statistics survive a deletion request.
    student_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("students.student_id", ondelete="CASCADE"),
        nullable=True,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("class_sessions.session_id", ondelete="CASCADE"),
    )
    match_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    marked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    class_session: Mapped[ClassSession] = relationship(back_populates="attendance_rows")


# ─── Admins & audit ───────────────────────────────────────────────────────────


class Admin(Base):
    __tablename__ = "admins"

    admin_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(100), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[AdminRole] = mapped_column(Enum(AdminRole, name="admin_role"))
    # Nullable: super_admin belongs to no institution/department.
    institution_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("institutions.institution_id", ondelete="CASCADE"),
        nullable=True,
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("departments.department_id", ondelete="CASCADE"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_created_at", "created_at"),
        Index("ix_audit_log_admin_id", "admin_id"),
    )

    log_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    admin_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("admins.admin_id", ondelete="SET NULL"),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String(100))
    target_type: Mapped[str] = mapped_column(String(100))
    # Polymorphic reference — may point at any table, so no FK.
    target_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    detail: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ─── Review queue ─────────────────────────────────────────────────────────────


class ReviewQueue(Base):
    """
    Unmatched faces land here instead of being force-matched below threshold.

    ``face_embedding`` stores the unmatched face's own 512-d vector so the
    admin's explicit "search all students" action can run
    find_nearest_embeddings_institution_wide WITHOUT re-running the ML model
    inside the API tier.
    """

    __tablename__ = "review_queue"
    __table_args__ = (
        Index("ix_review_queue_session_id", "session_id"),
        Index("ix_review_queue_status", "status"),
    )

    review_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("class_sessions.session_id", ondelete="CASCADE"),
    )
    face_crop_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    face_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    best_guess_student_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("students.student_id", ondelete="SET NULL"),
        nullable=True,
    )
    best_guess_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[ReviewStatus] = mapped_column(
        Enum(ReviewStatus, name="review_status"), default=ReviewStatus.pending
    )
    resolved_student_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("students.student_id", ondelete="SET NULL"),
        nullable=True,
    )
    resolved_by_admin_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("admins.admin_id", ondelete="SET NULL"),
        nullable=True,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
