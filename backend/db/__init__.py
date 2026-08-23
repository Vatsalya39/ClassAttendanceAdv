"""
backend.db
──────────
Database layer: SQLAlchemy models, async session management, partition
maintenance helpers, and the DAO (data-access object) layer.

Import models here so Alembic autogenerate / metadata.create_all sees them:
    from backend.db import Base  # noqa: F401  (registers all models)
"""

from backend.db.base import Base
from backend.db.models import (
    Admin,
    AdminRole,
    Attendance,
    AuditLog,
    ClassSession,
    Department,
    Enrollment,
    FaceEmbedding,
    Institution,
    ReviewQueue,
    ReviewStatus,
    Section,
    Student,
)

__all__ = [
    "Admin",
    "AdminRole",
    "Attendance",
    "AuditLog",
    "Base",
    "ClassSession",
    "Department",
    "Enrollment",
    "FaceEmbedding",
    "Institution",
    "ReviewQueue",
    "ReviewStatus",
    "Section",
    "Student",
]
