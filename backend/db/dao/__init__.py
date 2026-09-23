"""
backend.db.dao
──────────────
Data-access layer. Every function takes an AsyncSession as its first
argument and commits nothing unless documented — callers own transaction
boundaries.

CRITICAL (see docs/architecture.md): the ONLY two vector-search entry
points are find_nearest_embeddings_for_roster (section-scoped, fast, always
used by the capture worker) and find_nearest_embeddings_institution_wide
(deliberately slow fallback for the manual review-queue "search all
students" action). Any new code that queries face_embeddings must go
through one of these two.
"""

from backend.db.dao.admins import (
    audit_action,
    create_admin,
    get_admin_by_username,
    get_section_with_parents,
    sections_accessible_to_admin,
)
from backend.db.dao.attendance import (
    anonymize_attendance_for_student,
    create_class_session,
    delete_attendance_for_student,
    get_attendance_for_date,
    get_class_session,
    mark_attendance,
)
from backend.db.dao.matching import (
    RosterMatch,
    find_nearest_embeddings_for_roster,
    find_nearest_embeddings_for_student,
    find_nearest_embeddings_institution_wide,
)
from backend.db.dao.org import (
    StudentNotFoundError,
    StudentWrongInstitutionError,
    bulk_enroll,
    create_department,
    create_institution,
    create_section,
    get_section_by_id,
    list_departments,
    list_institutions,
    list_sections,
)
from backend.db.dao.registration_tokens import (
    create_registration_token,
    verify_and_consume_registration_token,
)
from backend.db.dao.review import (
    ReviewAlreadyResolvedError,
    delete_reviews_for_student,
    enqueue_review,
    get_review_item,
    list_pending_reviews,
    resolve_review,
)
from backend.db.dao.students import (
    RollNumberAlreadyExistsError,
    add_embedding,
    create_student,
    delete_embeddings,
    get_student,
    get_student_by_roll_number,
    list_embedding_image_urls,
)

__all__ = [
    "RosterMatch",
    "add_embedding",
    "anonymize_attendance_for_student",
    "audit_action",
    "bulk_enroll",
    "create_admin",
    "create_class_session",
    "create_department",
    "create_institution",
    "create_registration_token",
    "create_section",
    "create_student",
    "delete_attendance_for_student",
    "delete_embeddings",
    "delete_reviews_for_student",
    "enqueue_review",
    "find_nearest_embeddings_for_roster",
    "find_nearest_embeddings_for_student",
    "find_nearest_embeddings_institution_wide",
    "get_admin_by_username",
    "get_attendance_for_date",
    "get_class_session",
    "get_review_item",
    "get_section_by_id",
    "get_section_with_parents",
    "get_student",
    "get_student_by_roll_number",
    "list_departments",
    "list_embedding_image_urls",
    "list_institutions",
    "list_pending_reviews",
    "list_sections",
    "ReviewAlreadyResolvedError",
    "RollNumberAlreadyExistsError",
    "mark_attendance",
    "resolve_review",
    "sections_accessible_to_admin",
    "StudentNotFoundError",
    "StudentWrongInstitutionError",
    "verify_and_consume_registration_token",
]
