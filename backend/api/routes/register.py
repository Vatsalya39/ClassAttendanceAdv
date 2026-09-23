"""
backend/api/routes/register.py
──────────────────────────────
Student self-registration: runs the face pipeline on 3-5 enrollment images
and stores the resulting embeddings.

Registration is synchronous (a handful of portrait images), unlike session
capture which is queued. InsightFace must be importable in the API process
for this route to work — in docker-compose both containers share the same
image which includes ml/requirements.txt.
"""

from __future__ import annotations

import uuid

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from backend.api.deps import DbSession, RedisDep, StorageDep, enforce_rate_limit
from backend.api.schemas import RegisterResponse
from backend.core.config import settings
from backend.db import dao
from backend.db.models import Enrollment
from ml.pipeline import batch_process
from ml.utils import load_image_from_bytes, resize_for_detection

router = APIRouter(tags=["registration"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


def _validate_upload(file: UploadFile, data: bytes) -> None:
    if (file.content_type or "").lower() not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG/PNG enrollment images are accepted",
        )
    if len(data) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.upload_max_bytes // (1024 * 1024)} MB limit",
        )


@router.post("/register", response_model=RegisterResponse)
async def register(
    request: Request,
    session: DbSession,
    redis: RedisDep,
    storage: StorageDep,
    roll_number: str = Form(..., min_length=1, max_length=64),
    name: str = Form(..., min_length=1, max_length=255),
    password: str = Form(..., min_length=8, max_length=1024),
    section_id: uuid.UUID = Form(...),
    images: list[UploadFile] = File(...),
    registration_token: str | None = Form(None, max_length=128),
) -> RegisterResponse:
    # Brute-force / abuse guard: registration is unauthenticated.
    await enforce_rate_limit(
        request, redis, name="register", limit=settings.rate_limit_register_rpm
    )

    if not (settings.enrollment_min_images <= len(images) <= settings.enrollment_max_images):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"{settings.enrollment_min_images}-{settings.enrollment_max_images} "
                "enrollment images are required"
            ),
        )

    section_pair = await dao.get_section_with_parents(session, section_id)
    if section_pair is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section not found")
    _, department = section_pair

    # ── Identity verification (Phase B1 guardrail) ──────────────────────────
    # Ensure the registering client possesses an admin-issued single-use token
    # tied to this roll number and institution.
    if settings.require_registration_token:
        if not registration_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Registration token is required. Request a registration token from your institution administrator.",
            )
        token_valid = await dao.verify_and_consume_registration_token(
            session,
            roll_number=roll_number.strip(),
            token=registration_token.strip(),
            institution_id=department.institution_id,
        )
        if not token_valid:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid, expired, or already consumed registration token for this roll number.",
            )

    from backend.core.security import hash_password

    try:
        student = await dao.create_student(
            session,
            roll_number=roll_number.strip(),
            name=name.strip(),
            password_hash=hash_password(password),
            institution_id=department.institution_id,
        )
    except dao.RollNumberAlreadyExistsError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Roll number already registered"
        ) from exc

    embeddings_stored = 0
    for upload in images:
        data = await upload.read()
        _validate_upload(upload, data)

        # Enrollment photos are single-student close-ups: cap the decode size
        # (Module 2 — only session photos keep native resolution).
        image = load_image_from_bytes(data)  # raises ValueError on garbage
        image = resize_for_detection(image)
        faces = batch_process(image, model_pack=settings.insightface_model_pack)
        if len(faces) != 1:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Enrollment photo '{upload.filename}' must contain exactly one "
                    f"face (found {len(faces)}). Retake with a clear frontal shot."
                ),
            )

        # ── Enrollment hygiene (Module 4) ─────────────────────────────────
        # A NEW enrollment photo must sit closer to this student's existing
        # references than to any OTHER student's references. Catches
        # mislabeled uploads and accidental duplicate registrations of an
        # already-enrolled person before the embeddings poison the roster.
        new_embedding = faces[0].embedding
        if embeddings_stored > 0:
            # Gather THIS student's stored embeddings via a roster-agnostic
            # student-scoped lookup (own references) and the institution's
            # nearest OTHER-student reference via the explicitly-named
            # institution-wide DAO entry point (admin-review-grade query,
            # used here at registration time where volume is trivial).
            own_rows = await dao.find_nearest_embeddings_for_student(
                session, new_embedding, student_id=student.student_id, top_k=5
            )
            nearest_other = await dao.find_nearest_embeddings_institution_wide(
                session,
                new_embedding,
                institution_id=department.institution_id,
                top_k=1,
            )
            nearest_other = [m for m in nearest_other if m.student_id != student.student_id]

            if own_rows and nearest_other:
                own_dist = min(r.distance for r in own_rows)
                other_dist = min(r.distance for r in nearest_other)
                # The new photo must be closer to its OWN student than to
                # anyone else. Use the match threshold as the tolerance: if
                # another student's reference is CLOSER than ours, or our
                # best distance is already outside tolerance, reject.
                if other_dist < own_dist or own_dist > settings.face_match_threshold:
                    await session.rollback()
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            "Enrollment photo does not match this student's previous "
                            "photos (or matches another student more closely). "
                            "Retake with a clear frontal photo of the SAME person."
                        ),
                    )

        url = await storage.upload(
            data=data,
            filename=f"{roll_number}_{uuid.uuid4().hex[:8]}.jpg",
            container=settings.azure_blob_container_enrollment,
        )
        await dao.add_embedding(
            session,
            student_id=student.student_id,
            embedding=_as_list(faces[0].embedding),
            source_image_url=url,
        )
        embeddings_stored += 1

        # Dual-write during trial window: write both buffalo_l and antelopev2 embeddings
        if settings.dual_enrollment_write:
            alt_pack = (
                "antelopev2" if settings.insightface_model_pack == "buffalo_l" else "buffalo_l"
            )
            try:
                alt_faces = batch_process(image, model_pack=alt_pack)
                if alt_faces:
                    await dao.add_embedding(
                        session,
                        student_id=student.student_id,
                        embedding=_as_list(alt_faces[0].embedding),
                        source_image_url=url,
                    )
                    embeddings_stored += 1
            except Exception:  # noqa: BLE001
                pass

    # Enroll into the requested section as part of registration.
    session.add(
        Enrollment(
            enrollment_id=uuid.uuid4(),
            student_id=student.student_id,
            section_id=section_id,
        )
    )

    await session.commit()
    return RegisterResponse(
        student_id=student.student_id,
        roll_number=student.roll_number,
        name=student.name,
        embeddings_stored=embeddings_stored,
    )


def _as_list(embedding: list[float]) -> list[float]:
    """pgvector accepts plain python floats; guard against numpy scalars."""
    return [float(v) for v in np.asarray(embedding).ravel()]
