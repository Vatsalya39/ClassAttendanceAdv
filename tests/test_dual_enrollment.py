"""
tests/test_dual_enrollment.py
──────────────────────────────
Verifies dual-enrollment writing behavior during model migration window.

Ensures:
1. When dual_enrollment_write=True, registration writes both buffalo_l
   and antelopev2 embeddings for each enrollment photo.
2. Both embeddings are 512-dimensional and unit-normalised.
3. When dual_enrollment_write=False, only the active pack embedding is written.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.api.routes.register import register
from backend.core.config import settings
from backend.db.models import Department, Section, Student


@pytest.mark.asyncio
async def test_dual_enrollment_writes_both_backbone_embeddings() -> None:
    """Verify that registration stores two embeddings per photo when dual_enrollment_write=True."""
    mock_request = MagicMock()
    mock_session = AsyncMock()
    mock_redis = AsyncMock()
    mock_storage = MagicMock()
    mock_storage.upload = AsyncMock(return_value="https://blob.example.com/face.jpg")

    mock_file = MagicMock()
    mock_file.filename = "enroll.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.read = AsyncMock(return_value=b"fake_jpeg_bytes")

    inst_id = uuid.uuid4()
    sec_id = uuid.uuid4()
    dept_id = uuid.uuid4()
    stud_id = uuid.uuid4()

    mock_sec = Section(section_id=sec_id, department_id=dept_id, name="CSE-A", academic_year=2026)
    mock_dept = Department(department_id=dept_id, institution_id=inst_id, name="CSE")
    mock_student = Student(
        student_id=stud_id, roll_number="25BLC9999", name="Test Student", institution_id=inst_id
    )

    # 512-d mock embeddings
    emb_buffalo = list(np.random.randn(512).astype(np.float32))
    emb_buffalo = [float(v / np.linalg.norm(emb_buffalo)) for v in emb_buffalo]

    emb_antelope = list(np.random.randn(512).astype(np.float32))
    emb_antelope = [float(v / np.linalg.norm(emb_antelope)) for v in emb_antelope]

    mock_face_buffalo = MagicMock()
    mock_face_buffalo.embedding = emb_buffalo

    mock_face_antelope = MagicMock()
    mock_face_antelope.embedding = emb_antelope

    def mock_batch_process(img, model_pack="buffalo_l", **kwargs):
        if model_pack == "buffalo_l":
            return [mock_face_buffalo]
        elif model_pack == "antelopev2":
            return [mock_face_antelope]
        return [mock_face_buffalo]

    added_embeddings: list[list[float]] = []

    async def mock_add_embedding(session, student_id, embedding, source_image_url):
        added_embeddings.append(embedding)

    with (
        patch("backend.api.routes.register.enforce_rate_limit", new=AsyncMock()),
        patch(
            "backend.api.routes.register.dao.get_section_with_parents",
            new=AsyncMock(return_value=(mock_sec, mock_dept)),
        ),
        patch(
            "backend.api.routes.register.dao.create_student",
            new=AsyncMock(return_value=mock_student),
        ),
        patch(
            "backend.api.routes.register.dao.verify_and_consume_registration_token",
            new=AsyncMock(return_value=True),
        ),
        patch("backend.api.routes.register.dao.add_embedding", side_effect=mock_add_embedding),
        patch("backend.api.routes.register._validate_upload", return_value=None),
        patch(
            "backend.api.routes.register.load_image_from_bytes",
            return_value=np.zeros((200, 200, 3), dtype=np.uint8),
        ),
        patch("backend.api.routes.register.resize_for_detection", side_effect=lambda x: x),
        patch("backend.api.routes.register.batch_process", side_effect=mock_batch_process),
        patch.object(settings, "require_registration_token", True),
        patch.object(settings, "dual_enrollment_write", True),
        patch.object(settings, "insightface_model_pack", "buffalo_l"),
        patch.object(settings, "enrollment_min_images", 1),
    ):
        resp = await register(
            request=mock_request,
            session=mock_session,
            redis=mock_redis,
            storage=mock_storage,
            roll_number="25BLC9999",
            name="Test Student",
            password="password123",
            section_id=sec_id,
            images=[mock_file],
            registration_token="VALID-TOKEN",
        )

    assert (
        resp.embeddings_stored == 2
    ), f"Expected 2 embeddings stored, got {resp.embeddings_stored}"
    assert len(added_embeddings) == 2
    # Verify buffalo_l and antelopev2 embeddings were both written
    assert np.allclose(added_embeddings[0], emb_buffalo)
    assert np.allclose(added_embeddings[1], emb_antelope)


@pytest.mark.asyncio
async def test_single_enrollment_when_dual_write_disabled() -> None:
    """Verify only 1 embedding per photo is stored when dual_enrollment_write=False."""
    mock_request = MagicMock()
    mock_session = AsyncMock()
    mock_redis = AsyncMock()
    mock_storage = MagicMock()
    mock_storage.upload = AsyncMock(return_value="https://blob.example.com/face.jpg")

    mock_file = MagicMock()
    mock_file.filename = "enroll.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.read = AsyncMock(return_value=b"fake_jpeg_bytes")

    inst_id = uuid.uuid4()
    sec_id = uuid.uuid4()
    dept_id = uuid.uuid4()
    stud_id = uuid.uuid4()

    mock_sec = Section(section_id=sec_id, department_id=dept_id, name="CSE-A", academic_year=2026)
    mock_dept = Department(department_id=dept_id, institution_id=inst_id, name="CSE")
    mock_student = Student(
        student_id=stud_id, roll_number="25BLC9999", name="Test Student", institution_id=inst_id
    )

    emb_buffalo = list(np.random.randn(512).astype(np.float32))
    emb_buffalo = [float(v / np.linalg.norm(emb_buffalo)) for v in emb_buffalo]
    mock_face = MagicMock()
    mock_face.embedding = emb_buffalo

    added_embeddings: list[list[float]] = []

    async def mock_add_embedding(session, student_id, embedding, source_image_url):
        added_embeddings.append(embedding)

    with (
        patch("backend.api.routes.register.enforce_rate_limit", new=AsyncMock()),
        patch(
            "backend.api.routes.register.dao.get_section_with_parents",
            new=AsyncMock(return_value=(mock_sec, mock_dept)),
        ),
        patch(
            "backend.api.routes.register.dao.create_student",
            new=AsyncMock(return_value=mock_student),
        ),
        patch(
            "backend.api.routes.register.dao.verify_and_consume_registration_token",
            new=AsyncMock(return_value=True),
        ),
        patch("backend.api.routes.register.dao.add_embedding", side_effect=mock_add_embedding),
        patch("backend.api.routes.register._validate_upload", return_value=None),
        patch(
            "backend.api.routes.register.load_image_from_bytes",
            return_value=np.zeros((200, 200, 3), dtype=np.uint8),
        ),
        patch("backend.api.routes.register.resize_for_detection", side_effect=lambda x: x),
        patch("backend.api.routes.register.batch_process", return_value=[mock_face]),
        patch.object(settings, "require_registration_token", True),
        patch.object(settings, "dual_enrollment_write", False),
        patch.object(settings, "enrollment_min_images", 1),
    ):
        resp = await register(
            request=mock_request,
            session=mock_session,
            redis=mock_redis,
            storage=mock_storage,
            roll_number="25BLC9999",
            name="Test Student",
            password="password123",
            section_id=sec_id,
            images=[mock_file],
            registration_token="VALID-TOKEN",
        )

    assert resp.embeddings_stored == 1
    assert len(added_embeddings) == 1
