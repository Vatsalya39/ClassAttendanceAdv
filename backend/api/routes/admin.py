"""
backend/api/routes/admin.py
───────────────────────────
Admin-facing endpoints:

    POST /admin/session/capture              → enqueue async capture job
    GET  /admin/session/capture/{job_id}     → poll job status/results
    GET  /admin/review-queue                 → pending unmatched faces (scoped)
    POST /admin/review-queue/{id}/resolve    → assign or discard
    POST /admin/review-queue/{id}/search-all → explicit institution-wide search
    GET  /admin/attendance                   → section+date report

Every route re-verifies role/scope server-side against the admin's DB row;
a department_admin for Dept A gets 403 touching Dept B's sections.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select

from backend.api.deps import (
    AdminUser,
    DbSession,
    QueueDep,
    RedisDep,
    StorageDep,
    enforce_rate_limit,
    require_role,
    require_section_access,
)
from backend.api.schemas import (
    AttendanceReportResponse,
    AttendanceReportRow,
    CaptureAcceptedResponse,
    CaptureResultStudent,
    CaptureStatusResponse,
    CreateRegistrationTokenRequest,
    RegistrationTokenResponse,
    ReviewItem,
    ReviewResolvedResponse,
    ReviewResolveRequest,
    ReviewSearchAllCandidate,
    ReviewSearchAllResponse,
)
from backend.core.config import settings
from backend.core.jobstatus import JobStatus, get_job_scope, get_status, set_job_scope, set_status
from backend.core.queue import CaptureJobMessage
from backend.db import dao
from backend.db.dao.matching import find_nearest_embeddings_institution_wide
from backend.db.models import AdminRole, ClassSession, Enrollment, ReviewQueue, Section, Student
from ml.utils import load_image_from_bytes, validate_image_resolution

router = APIRouter(prefix="/admin", tags=["admin"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


# ─── Async capture ────────────────────────────────────────────────────────────


@router.post("/session/capture", response_model=CaptureAcceptedResponse)
async def capture(
    request: Request,
    session: DbSession,
    admin: AdminUser,
    redis: RedisDep,
    queue: QueueDep,
    storage: StorageDep,
    section_id: uuid.UUID = Form(...),
    course_name: str = Form("General"),
    image: UploadFile | None = File(None),
    frames: list[UploadFile] | None = File(None),
) -> CaptureAcceptedResponse:
    """
    Validate + store one classroom photo (or a 1-5 frame burst, Module 4),
    enqueue the processing job, and return immediately. The ML pipeline
    NEVER runs inside this request.

    ``image`` is the primary frame (legacy single-photo clients keep
    working). ``frames`` optionally supplies 2-5 burst photos of the same
    session — the worker processes each and unions recognized students via
    the idempotent attendance constraint.
    """
    await require_section_access(session, admin, section_id)

    # Per-admin AND per-section limits so a runaway client can't flood the queue.
    await enforce_rate_limit(
        request,
        redis,
        name=f"capture-admin:{admin.admin_id}",
        limit=settings.rate_limit_capture_rpm,
    )
    await enforce_rate_limit(
        request,
        redis,
        name=f"capture-section:{section_id}",
        limit=settings.rate_limit_capture_rpm * 4,
    )

    uploads: list[UploadFile] = []
    if image is not None:
        uploads.append(image)
    if frames:
        uploads.extend(f for f in frames if f is not None)

    if not uploads:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one classroom photo is required",
        )
    if len(uploads) > settings.capture_max_frames:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Too many frames — capture accepts up to {settings.capture_max_frames} "
                "burst photos per session"
            ),
        )

    frame_urls: list[str] = []
    for upload in uploads:
        if (upload.content_type or "").lower() not in _ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only JPEG/PNG classroom photos are accepted",
            )
        data = await upload.read()
        if len(data) > settings.upload_max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds {settings.upload_max_bytes // (1024 * 1024)} MB limit",
            )
        if not data:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file"
            )
        # Resolution floor: back-row faces below ~1024px on the shorter side
        # embed unreliably and flood the review queue. Reject early with an
        # actionable message (the frontend camera requests 1080p+).
        try:
            decoded = load_image_from_bytes(data)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not decode image — unsupported format or corrupted data",
            ) from None
        ok, (w, h) = validate_image_resolution(decoded, settings.capture_min_image_side)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Photo resolution too low ({w}x{h}px) — at least "
                    f"{settings.capture_min_image_side}px on the shorter side is required "
                    "so back-row students are recognisable. Capture at 1080p or higher."
                ),
            )
        frame_urls.append(
            await storage.upload(
                data=data,
                filename=f"capture_{uuid.uuid4().hex[:12]}_{len(frame_urls)}.jpg",
                container=settings.azure_blob_container_sessions,
            )
        )

    class_session = await dao.create_class_session(
        session,
        section_id=section_id,
        course_name=course_name.strip() or "General",
        session_date=date.today(),
        admin_id=admin.admin_id,
    )
    job_id = uuid.uuid4()
    message = CaptureJobMessage(
        job_id=job_id,
        session_id=class_session.session_id,
        section_id=section_id,
        image_url=frame_urls[0],
        frame_urls=frame_urls,
        admin_id=admin.admin_id,
    )
    await dao.audit_action(
        session,
        admin_id=admin.admin_id,
        action="capture.enqueued",
        target_type="class_session",
        target_id=class_session.session_id,
        detail=f"job={job_id} section={section_id} frames={len(frame_urls)}",
    )
    await set_status(
        redis, str(job_id), JobStatus.queued, ttl_seconds=settings.job_status_ttl_seconds
    )
    # Tenant marker: polling admins must have scope over THIS section.
    await set_job_scope(
        redis, str(job_id), str(section_id), ttl_seconds=settings.job_status_ttl_seconds
    )
    await queue.enqueue(message)
    await session.commit()

    return CaptureAcceptedResponse(
        job_id=job_id,
        session_id=class_session.session_id,
        status=JobStatus.queued.value,
    )


async def _enrich_students(
    session: DbSession, entries: list[dict[str, Any]]
) -> list[CaptureResultStudent]:
    """Attach roll_number/name to worker result entries (ids only in Redis)."""
    out: list[CaptureResultStudent] = []
    for entry in entries:
        student = await dao.get_student(session, uuid.UUID(str(entry["student_id"])))
        raw_distance = entry.get("distance")
        out.append(
            CaptureResultStudent(
                student_id=uuid.UUID(str(entry["student_id"])),
                roll_number=student.roll_number if student else None,
                name=student.name if student else None,
                match_distance=float(raw_distance) if raw_distance is not None else None,
            )
        )
    return out


@router.get("/session/capture/{job_id}", response_model=CaptureStatusResponse)
async def capture_status(
    job_id: uuid.UUID,
    session: DbSession,
    admin: AdminUser,
    redis: RedisDep,
) -> CaptureStatusResponse:
    status_data = await get_status(redis, str(job_id))
    if status_data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job")

    # Tenant isolation on the status channel itself: a scoped-out admin must
    # not even observe another department's job progress.
    section_scope = await get_job_scope(redis, str(job_id))
    if section_scope:
        await require_section_access(session, admin, uuid.UUID(section_scope))

    matched: list[CaptureResultStudent] = []
    review_needed: list[CaptureResultStudent] = []
    result = status_data.get("result") or {}
    if status_data["status"] == JobStatus.done.value:
        matched = await _enrich_students(session, list(result.get("matched", [])))
        review_needed = await _enrich_students(session, list(result.get("review_needed", [])))

    return CaptureStatusResponse(
        job_id=job_id,
        status=str(status_data["status"]),
        error=status_data.get("error"),
        matched=matched,
        review_needed=review_needed,
    )


# ─── Review queue ─────────────────────────────────────────────────────────────


def _review_to_item(row: ReviewQueue, section: Section) -> ReviewItem:
    return ReviewItem(
        review_id=row.review_id,
        session_id=row.session_id,
        section_id=section.section_id,
        section_name=section.name,
        face_crop_url=row.face_crop_url,
        best_guess_student_id=row.best_guess_student_id,
        best_guess_distance=row.best_guess_distance,
        created_at=row.created_at,
    )


@router.get("/review-queue", response_model=list[ReviewItem])
async def list_review_queue(
    session: DbSession,
    admin: AdminUser,
    section_id: uuid.UUID | None = None,
    limit: int = 100,
) -> list[ReviewItem]:
    if section_id is not None:
        await require_section_access(session, admin, section_id)
        scoped_ids = [section_id]
    else:
        scoped_ids = await dao.sections_accessible_to_admin(session, admin)
        if not scoped_ids and admin.role.value != "super_admin":
            return []

    stmt = (
        select(ReviewQueue, ClassSession)
        .join(ClassSession, ClassSession.session_id == ReviewQueue.session_id)
        .where(
            ClassSession.section_id.in_(scoped_ids),
            ReviewQueue.status == "pending",
        )
        .order_by(ReviewQueue.created_at.asc())
        .limit(min(limit, 500))
    )
    rows = (await session.execute(stmt)).all()

    items: list[ReviewItem] = []
    for review, cs in rows:
        section = await dao.get_section_by_id(session, cs.section_id)
        assert section is not None
        items.append(_review_to_item(review, section))
    return items


async def _load_scoped_review(
    session: DbSession, admin: AdminUser, review_id: uuid.UUID
) -> ReviewQueue:
    review = await dao.get_review_item(session, review_id)
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found")
    cs = await dao.get_class_session(session, review.session_id)
    if cs is None:  # pragma: no cover — FK guarantees existence
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session missing")
    await require_section_access(session, admin, cs.section_id)
    return review


@router.post("/review-queue/{review_id}/resolve", response_model=ReviewResolvedResponse)
async def resolve_review(
    review_id: uuid.UUID,
    body: ReviewResolveRequest,
    session: DbSession,
    admin: AdminUser,
) -> ReviewResolvedResponse:
    review = await _load_scoped_review(session, admin, review_id)
    cs = await dao.get_class_session(session, review.session_id)
    assert cs is not None

    resolved_student_id: uuid.UUID | None = None
    if body.action == "assign":
        if body.student_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="student_id is required when assigning",
            )
        student = await dao.get_student(session, body.student_id)
        if student is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

        # Tenant check: student must belong to the session's institution —
        # roster members always do; institution-wide picks are validated too.
        pair = await dao.get_section_with_parents(session, cs.section_id)
        assert pair is not None
        _, department = pair
        if student.institution_id != department.institution_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Student belongs to another institution",
            )

        if not body.search_all:
            # Default path: assignment must come from THIS section's roster.
            enrolled = (
                await session.execute(
                    select(Enrollment.enrollment_id).where(
                        Enrollment.student_id == student.student_id,
                        Enrollment.section_id == cs.section_id,
                    )
                )
            ).scalar_one_or_none()
            if enrolled is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Student is not in this section's roster (use search_all for exceptions)",
                )
        resolved_student_id = student.student_id

    try:
        await dao.resolve_review(
            session,
            review_id=review.review_id,
            admin_id=admin.admin_id,
            resolved_student_id=resolved_student_id,
        )
    except dao.ReviewAlreadyResolvedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if resolved_student_id is not None:
        cs_row = await dao.get_class_session(session, review.session_id)
        assert cs_row is not None
        await dao.mark_attendance(
            session,
            student_id=resolved_student_id,
            session_id=review.session_id,
            session_date=cs_row.session_date,
            match_distance=None,
        )

    await dao.audit_action(
        session,
        admin_id=admin.admin_id,
        action="review.resolved",
        target_type="review_queue",
        target_id=review.review_id,
        detail=f"action={body.action} student={resolved_student_id}",
    )
    await session.commit()

    return ReviewResolvedResponse(
        review_id=review.review_id,
        status="resolved",
        resolved_student_id=resolved_student_id,
    )


@router.post("/review-queue/{review_id}/search-all", response_model=ReviewSearchAllResponse)
async def review_search_all(
    review_id: uuid.UUID,
    session: DbSession,
    admin: AdminUser,
) -> ReviewSearchAllResponse:
    """
    The ONE legitimate broader search (docs/architecture.md): an explicit,
    per-item, institution-scoped vector lookup used only while manually
    identifying an unmatched face. Deliberately slower than roster search.
    """
    review = await _load_scoped_review(session, admin, review_id)
    cs = await dao.get_class_session(session, review.session_id)
    assert cs is not None
    pair = await dao.get_section_with_parents(session, cs.section_id)
    assert pair is not None
    _, department = pair

    if review.face_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No embedding stored for this review item",
        )

    matches = await find_nearest_embeddings_institution_wide(
        session,
        list(review.face_embedding),
        institution_id=department.institution_id,
        top_k=10,
    )
    candidates: list[ReviewSearchAllCandidate] = []
    for m in matches:
        student = await dao.get_student(session, m.student_id)
        if student is None:
            continue
        candidates.append(
            ReviewSearchAllCandidate(
                student_id=student.student_id,
                roll_number=student.roll_number,
                name=student.name,
                distance=m.distance,
            )
        )
    return ReviewSearchAllResponse(review_id=review.review_id, candidates=candidates)


# ─── Attendance report ────────────────────────────────────────────────────────


@router.get("/attendance", response_model=AttendanceReportResponse)
async def attendance_report(
    session: DbSession,
    admin: AdminUser,
    section_id: uuid.UUID,
    date: date,
) -> AttendanceReportResponse:
    await require_section_access(session, admin, section_id)
    rows = await dao.get_attendance_for_date(session, section_id=section_id, session_date=date)

    present = sum(1 for r in rows if r.marked_at is not None)
    return AttendanceReportResponse(
        section_id=section_id,
        date=date,
        total_enrolled=len(rows),
        present=present,
        rows=[
            AttendanceReportRow(
                student_id=r.student_id,
                roll_number=r.roll_number,
                name=r.student_name,
                marked=r.marked_at is not None,
                match_distance=r.match_distance,
                marked_at=r.marked_at,
            )
            for r in rows
        ],
    )


# ─── Biometric purge (Phase 5 — docs/data-retention.md) ───────────────────────


class PurgeBiometricsRequest(BaseModel):
    attendance_action: Literal["anonymize", "delete"] = "anonymize"
    delete_account: bool = False


class PurgeBiometricsResponse(BaseModel):
    student_id: uuid.UUID
    embeddings_deleted: int
    images_deleted: int
    reviews_deleted: int
    attendance_affected: int
    account_deleted: bool


@router.post("/students/{student_id}/purge-biometrics", response_model=PurgeBiometricsResponse)
async def purge_biometrics(
    student_id: uuid.UUID,
    body: PurgeBiometricsRequest,
    session: DbSession,
    admin: AdminUser,
    storage: StorageDep,
) -> PurgeBiometricsResponse:
    """
    Right-to-erasure workflow (docs/data-retention.md):

        1. delete all face_embeddings for the student
        2. delete enrollment images from blob storage
        3. anonymise (or hard-delete) partitioned attendance history
        4. delete review-queue items referencing the student
        5. write an audit-log entry

    super_admin / institution_admin of the student's institution ONLY — a
    department_admin or faculty may never trigger biometric deletion.
    """
    await require_role(admin, AdminRole.super_admin, AdminRole.institution_admin)

    student = await dao.get_student(session, student_id)
    if student is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    if admin.role == AdminRole.institution_admin and admin.institution_id != student.institution_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student belongs to another institution",
        )

    # 1+2. Collect image URLs BEFORE deleting the embedding rows, then drop
    # the blobs from storage. Blob failures are logged but do not block the
    # DB purge (a lifecycle policy backstop cleans orphans in production).
    image_urls = await dao.list_embedding_image_urls(session, student.student_id)
    embeddings_deleted = await dao.delete_embeddings(session, student.student_id)

    images_deleted = 0
    for url in image_urls:
        try:
            await storage.delete(url)
            images_deleted += 1
        except Exception:  # noqa: BLE001 — storage errors must not block erasure
            logging.getLogger(__name__).warning("purge: failed to delete blob %s", url)

    # 3. Attendance history.
    if body.attendance_action == "delete":
        attendance_affected = await dao.delete_attendance_for_student(session, student.student_id)
    else:
        attendance_affected = await dao.anonymize_attendance_for_student(
            session, student.student_id
        )

    # 4. Review queue references.
    reviews_deleted = await dao.delete_reviews_for_student(session, student.student_id)

    # Optional: remove the account row itself (cascades enrollments etc.).
    account_deleted = False
    if body.delete_account:
        # Core DELETE so the DB's ON DELETE CASCADE handles enrollments etc.
        # (session.delete() would attempt to NULL out child FKs first.)
        from sqlalchemy import delete as sa_delete

        await session.execute(sa_delete(Student).where(Student.student_id == student.student_id))
        account_deleted = True

    # 5. Audit trail.
    await dao.audit_action(
        session,
        admin_id=admin.admin_id,
        action="student.biometrics_purged",
        target_type="student",
        target_id=student_id,
        detail=(
            f"attendance={body.attendance_action} images={images_deleted} "
            f"embeddings={embeddings_deleted} account_deleted={account_deleted}"
        ),
    )
    await session.commit()

    return PurgeBiometricsResponse(
        student_id=student_id,
        embeddings_deleted=embeddings_deleted,
        images_deleted=images_deleted,
        reviews_deleted=reviews_deleted,
        attendance_affected=attendance_affected,
        account_deleted=account_deleted,
    )


@router.post("/registration-tokens", response_model=RegistrationTokenResponse)
async def issue_registration_token(
    body: CreateRegistrationTokenRequest,
    session: DbSession,
    admin: AdminUser,
) -> RegistrationTokenResponse:
    """
    Issue a single-use registration token for a student roll number.
    Ensures enrollment requests can be tied back to an authorized identity.
    """
    await require_role(admin, AdminRole.super_admin, AdminRole.institution_admin)
    if admin.institution_id is None and admin.role != AdminRole.super_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Institution required")

    inst_id = admin.institution_id
    if inst_id is None:
        from backend.db.models import Institution

        inst = (await session.execute(select(Institution))).scalars().first()
        if inst is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="No institution found"
            )
        inst_id = inst.institution_id

    token_row = await dao.create_registration_token(
        session,
        roll_number=body.roll_number,
        institution_id=inst_id,
        expires_in_days=body.expires_in_days,
    )
    await dao.audit_action(
        session,
        admin_id=admin.admin_id,
        action="registration_token.issued",
        target_type="student",
        target_id=None,
        detail=f"roll_number={body.roll_number} expires_at={token_row.expires_at}",
    )
    await session.commit()
    return RegistrationTokenResponse(
        token=token_row.token,
        roll_number=token_row.roll_number,
        expires_at=token_row.expires_at,
    )
