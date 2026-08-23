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
from ml.utils import load_image_from_bytes

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

        image = load_image_from_bytes(data)  # raises ValueError on garbage
        faces = batch_process(image)
        if len(faces) != 1:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Enrollment photo '{upload.filename}' must contain exactly one "
                    f"face (found {len(faces)}). Retake with a clear frontal shot."
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
