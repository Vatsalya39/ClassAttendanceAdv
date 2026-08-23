"""
backend/api/schemas.py
──────────────────────
Pydantic request/response models for the HTTP API.

Face embeddings and images are biometric data — they are NEVER included in
any response schema (see docs/data-retention.md).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field

# ─── Auth ────────────────────────────────────────────────────────────────────


class StudentLoginRequest(BaseModel):
    roll_number: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=1024)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


# ─── Registration ────────────────────────────────────────────────────────────


class RegisterResponse(BaseModel):
    student_id: uuid.UUID
    roll_number: str
    name: str
    embeddings_stored: int


# ─── Student attendance ──────────────────────────────────────────────────────


class AttendanceRecord(BaseModel):
    session_id: uuid.UUID
    course_name: str
    session_date: date
    marked_at: datetime | None
    match_distance: float | None


class StudentAttendanceResponse(BaseModel):
    student_id: uuid.UUID
    records: list[AttendanceRecord]


# ─── Capture ─────────────────────────────────────────────────────────────────


class CaptureAcceptedResponse(BaseModel):
    job_id: uuid.UUID
    session_id: uuid.UUID
    status: str = "queued"


class CaptureResultStudent(BaseModel):
    student_id: uuid.UUID
    roll_number: str | None = None
    name: str | None = None
    match_distance: float | None = None


class CaptureStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: str  # queued | processing | done | failed
    error: str | None = None
    matched: list[CaptureResultStudent] = []
    review_needed: list[CaptureResultStudent] = []


# ─── Review queue ────────────────────────────────────────────────────────────


class ReviewItem(BaseModel):
    review_id: uuid.UUID
    session_id: uuid.UUID
    section_id: uuid.UUID
    section_name: str
    face_crop_url: str | None
    best_guess_student_id: uuid.UUID | None
    best_guess_distance: float | None
    created_at: datetime


class ReviewResolveRequest(BaseModel):
    """Assign the face to a roster student, or discard it."""

    action: str = Field(pattern="^(assign|discard)$")
    student_id: uuid.UUID | None = None
    # Explicit "search all students": run the slower institution-wide vector
    # search using the stored unmatched-face embedding and return candidates
    # for the admin to pick from. Never automatic.
    search_all: bool = False


class ReviewSearchAllCandidate(BaseModel):
    student_id: uuid.UUID
    roll_number: str
    name: str
    distance: float


class ReviewSearchAllResponse(BaseModel):
    review_id: uuid.UUID
    candidates: list[ReviewSearchAllCandidate]


class ReviewResolvedResponse(BaseModel):
    review_id: uuid.UUID
    status: str
    resolved_student_id: uuid.UUID | None


# ─── Attendance reports ──────────────────────────────────────────────────────


class AttendanceReportRow(BaseModel):
    student_id: uuid.UUID
    roll_number: str
    name: str
    marked: bool
    match_distance: float | None
    marked_at: datetime | None


class AttendanceReportResponse(BaseModel):
    section_id: uuid.UUID
    date: date
    total_enrolled: int
    present: int
    rows: list[AttendanceReportRow]


# ─── Public (registration helper) ─────────────────────────────────────────────


class PublicSectionOut(BaseModel):
    section_id: uuid.UUID
    name: str
    academic_year: int
    department_name: str
    institution_name: str


# ─── Org CRUD ────────────────────────────────────────────────────────────────


class InstitutionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class InstitutionOut(BaseModel):
    institution_id: uuid.UUID
    name: str


class DepartmentCreate(BaseModel):
    institution_id: uuid.UUID
    name: str = Field(min_length=1, max_length=255)


class DepartmentOut(BaseModel):
    department_id: uuid.UUID
    institution_id: uuid.UUID
    name: str


class SectionCreate(BaseModel):
    department_id: uuid.UUID
    name: str = Field(min_length=1, max_length=100)
    academic_year: int


class SectionOut(BaseModel):
    section_id: uuid.UUID
    department_id: uuid.UUID
    name: str
    academic_year: int


class RosterStudentOut(BaseModel):
    student_id: uuid.UUID
    roll_number: str
    name: str


class BulkEnrollResponse(BaseModel):
    section_id: uuid.UUID
    enrolled: int
    skipped: int


class AdminCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=1024)
    role: str = Field(pattern="^(institution_admin|department_admin|faculty)$")
    institution_id: uuid.UUID | None = None
    department_id: uuid.UUID | None = None
