"""
tests/test_security_hardening.py
────────────────────────────────
Phase 5 hardening acceptance tests:
  - /register is rate limited per IP
  - short passwords are rejected by input validation
  - security headers on every response
  - purge-biometrics: anonymize / delete attendance, blob deletion,
    review cleanup, audit entry, account deletion, role restrictions
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from backend.core.config import settings
from backend.core.security import hash_password
from backend.db import dao
from backend.db.models import AdminRole, Attendance, AuditLog, Enrollment


def make_jpeg(size: int = 32) -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((size, size, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
async def purge_world(db_session: Any, tmp_storage: Any) -> dict[str, Any]:
    """Student with embedding+blob, attendance row, review item, admins."""
    inst = await dao.create_institution(db_session, name="Purge U")
    dept = await dao.create_department(db_session, institution_id=inst.institution_id, name="D")
    section = await dao.create_section(
        db_session, department_id=dept.department_id, name="P-1", academic_year=2026
    )
    inst_admin = await dao.create_admin(
        db_session,
        username="purge_inst",
        password_hash=hash_password("purge-inst-pass"),
        role=AdminRole.institution_admin,
        institution_id=inst.institution_id,
        department_id=None,
    )
    dept_admin = await dao.create_admin(
        db_session,
        username="purge_dept",
        password_hash=hash_password("purge-dept-pass"),
        role=AdminRole.department_admin,
        institution_id=inst.institution_id,
        department_id=dept.department_id,
    )
    student = await dao.create_student(
        db_session,
        roll_number="PURGE1",
        name="Purge Me",
        password_hash="x",
        institution_id=inst.institution_id,
    )
    db_session.add(
        Enrollment(
            enrollment_id=uuid.uuid4(),
            student_id=student.student_id,
            section_id=section.section_id,
        )
    )
    # Real enrollment blob through the overridden storage backend.
    url = await tmp_storage.upload(
        data=make_jpeg(), filename="enroll.jpg", container="enrollment-images"
    )
    await dao.add_embedding(
        db_session,
        student_id=student.student_id,
        embedding=[0.1] * 512,
        source_image_url=url,
    )
    cs = await dao.create_class_session(
        db_session,
        section_id=section.section_id,
        course_name="C",
        session_date=date.today(),
        admin_id=None,
    )
    await dao.mark_attendance(
        db_session,
        student_id=student.student_id,
        session_id=cs.session_id,
        session_date=date.today(),
        match_distance=0.2,
    )
    await dao.enqueue_review(
        db_session,
        session_id=cs.session_id,
        best_guess_student_id=student.student_id,
    )
    await db_session.commit()
    return {
        "inst": inst,
        "section": section,
        "inst_admin": inst_admin,
        "dept_admin": dept_admin,
        "student": student,
        "blob_url": url,
        "tmp_storage": tmp_storage,
        "session": cs,
    }


async def _login(client: AsyncClient, username: str, password: str) -> str:
    resp = await client.post("/auth/admin/login", json={"username": username, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ─── Rate limiting & validation ──────────────────────────────────────────────


async def test_register_is_rate_limited(
    client: AsyncClient,
    redis_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "rate_limit_register_rpm", 2)
    files = [("images", (f"e{i}.jpg", make_jpeg(), "image/jpeg")) for i in range(3)]
    payload = {
        "roll_number": "RL1",
        "name": "Rate Limited",
        "password": "long-enough-password",
        "section_id": str(uuid.uuid4()),
    }
    for _ in range(2):
        resp = await client.post("/register", data=payload, files=files)
        # Fails later (unknown section / no faces) but counts against the limit.
        assert resp.status_code in (404, 422)
    resp = await client.post("/register", data=payload, files=files)
    assert resp.status_code == 429


async def test_short_password_rejected(client: AsyncClient, redis_client: Any) -> None:
    files = [("images", (f"e{i}.jpg", make_jpeg(), "image/jpeg")) for i in range(3)]
    resp = await client.post(
        "/register",
        data={
            "roll_number": "SP1",
            "name": "Short Password",
            "password": "short",
            "section_id": str(uuid.uuid4()),
        },
        files=files,
    )
    assert resp.status_code == 422
    assert "password" in resp.text.lower()


async def test_security_headers_present(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=(self)" in resp.headers["Permissions-Policy"]


# ─── Purge-biometrics flows ──────────────────────────────────────────────────


async def test_purge_anonymizes_attendance_and_deletes_biometrics(
    client: AsyncClient, db_session: Any, purge_world: dict[str, Any]
) -> None:
    token = await _login(client, "purge_inst", "purge-inst-pass")
    blob_path = purge_world["tmp_storage"]._root / purge_world["blob_url"].removeprefix("/static/")
    assert blob_path.exists(), "blob should exist before purge"

    resp = await client.post(
        f"/admin/students/{purge_world['student'].student_id}/purge-biometrics",
        json={"attendance_action": "anonymize"},
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["embeddings_deleted"] == 1
    assert body["images_deleted"] == 1
    assert body["reviews_deleted"] == 1
    assert body["attendance_affected"] == 1
    assert body["account_deleted"] is False

    # Blob removed from storage.
    assert not blob_path.exists()

    # Embeddings gone; student row retained.
    from backend.db.models import FaceEmbedding

    emb_count = (
        await db_session.execute(
            select(FaceEmbedding.embedding_id).where(
                FaceEmbedding.student_id == purge_world["student"].student_id
            )
        )
    ).all()
    assert emb_count == []
    assert await dao.get_student(db_session, purge_world["student"].student_id) is not None

    # Attendance anonymised: row survives, student reference + distance gone.
    rows = (
        (
            await db_session.execute(
                select(Attendance).where(Attendance.session_id == purge_world["session"].session_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].student_id is None
    assert rows[0].match_distance is None

    # Audit entry written.
    logs = (
        (
            await db_session.execute(
                select(AuditLog).where(AuditLog.action == "student.biometrics_purged")
            )
        )
        .scalars()
        .all()
    )
    assert len(logs) == 1


async def test_purge_delete_mode_removes_attendance(
    client: AsyncClient, db_session: Any, purge_world: dict[str, Any]
) -> None:
    token = await _login(client, "purge_inst", "purge-inst-pass")
    resp = await client.post(
        f"/admin/students/{purge_world['student'].student_id}/purge-biometrics",
        json={"attendance_action": "delete"},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    assert resp.json()["attendance_affected"] == 1

    rows = (await db_session.execute(select(Attendance))).scalars().all()
    assert rows == []


async def test_purge_with_account_deletion(
    client: AsyncClient, db_session: Any, purge_world: dict[str, Any]
) -> None:
    token = await _login(client, "purge_inst", "purge-inst-pass")
    resp = await client.post(
        f"/admin/students/{purge_world['student'].student_id}/purge-biometrics",
        json={"attendance_action": "delete", "delete_account": True},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    assert resp.json()["account_deleted"] is True
    assert await dao.get_student(db_session, purge_world["student"].student_id) is None


async def test_purge_unknown_student_404(client: AsyncClient, purge_world: dict[str, Any]) -> None:
    token = await _login(client, "purge_inst", "purge-inst-pass")
    resp = await client.post(
        f"/admin/students/{uuid.uuid4()}/purge-biometrics",
        json={"attendance_action": "delete"},
        headers=_auth(token),
    )
    assert resp.status_code == 404
