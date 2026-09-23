"""
tests/test_api_endpoints.py
───────────────────────────
Phase 3 API integration tests.

Covers:
  - admin/student login issuing scoped JWTs
  - POST /admin/session/capture: enqueue-only (no inline ML), image
    validation, job-status polling
  - TENANT ISOLATION: a department_admin for Dept B gets 403 touching
    Dept A's sections (capture + review queue + reports)
  - review-queue resolve (assign within roster / discard)
  - attendance report endpoint

Uses the real app with dependency overrides: DB → test database, Redis →
live test Redis (skips if down), storage → tmp dir, queue → in-memory.
InsightFace is NOT required — the capture path is enqueue-only by design.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pytest
from httpx import AsyncClient

from backend.core.config import settings
from backend.core.queue import CaptureJobMessage
from backend.core.security import hash_password
from backend.db import dao
from backend.db.models import AdminRole, Enrollment


def make_jpeg_bytes(size: int = 1080) -> bytes:
    """Session-photo fixture: default 1080x1080 clears the capture
    resolution floor (settings.capture_min_image_side = 1024)."""
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((size, size, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


# ─── Helpers ─────────────────────────────────────────────────────────────────


# ─── World builder ───────────────────────────────────────────────────────────


@pytest.fixture
async def tenant_world(db_session: Any) -> dict[str, Any]:
    """
    Two departments, two sections, two department_admins, one student per
    section — the minimum for tenant-isolation assertions.
    """
    ids: dict[str, Any] = {}
    inst = await dao.create_institution(db_session, name="Tenant U")
    ids["institution"] = inst.institution_id
    for tag in ("a", "b"):
        dept = await dao.create_department(
            db_session, institution_id=inst.institution_id, name=f"Dept {tag.upper()}"
        )
        section = await dao.create_section(
            db_session,
            department_id=dept.department_id,
            name=f"SEC-{tag.upper()}",
            academic_year=2026,
        )
        admin = await dao.create_admin(
            db_session,
            username=f"admin_{tag}",
            password_hash=hash_password(f"pw-{tag}-password"),
            role=AdminRole.department_admin,
            institution_id=inst.institution_id,
            department_id=dept.department_id,
        )
        student = await dao.create_student(
            db_session,
            roll_number=f"T{tag.upper()}001",
            name=f"Tenant {tag.upper()} Student",
            password_hash=hash_password("student-pw"),
            institution_id=inst.institution_id,
        )
        db_session.add(
            Enrollment(
                enrollment_id=uuid.uuid4(),
                student_id=student.student_id,
                section_id=section.section_id,
            )
        )
        ids[f"dept_{tag}"] = dept.department_id
        ids[f"section_{tag}"] = section.section_id
        ids[f"admin_{tag}"] = admin
        ids[f"student_{tag}"] = student
    await db_session.commit()
    return ids


async def login_admin(client: AsyncClient, username: str, password: str) -> str:
    resp = await client.post("/auth/admin/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    data: dict[str, str] = resp.json()
    return data["access_token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ─── Auth endpoints ──────────────────────────────────────────────────────────


async def test_admin_login_issues_scoped_jwt(
    client: AsyncClient, tenant_world: dict[str, Any]
) -> None:
    resp = await client.post(
        "/auth/admin/login", json={"username": "admin_a", "password": "pw-a-password"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"

    from backend.core.security import decode_token

    claims = decode_token(body["access_token"])
    assert claims["role"] == "department_admin"
    assert claims["department_id"] == str(tenant_world["dept_a"])


async def test_student_login_works(client: AsyncClient, tenant_world: dict[str, Any]) -> None:
    resp = await client.post(
        "/auth/student/login", json={"roll_number": "TA001", "password": "student-pw"}
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


async def test_wrong_password_rejected(client: AsyncClient, tenant_world: dict[str, Any]) -> None:
    resp = await client.post("/auth/admin/login", json={"username": "admin_a", "password": "wrong"})
    assert resp.status_code == 401


# ─── Tenant isolation ────────────────────────────────────────────────────────


async def test_dept_admin_cannot_capture_other_dept_section(
    client: AsyncClient, tenant_world: dict[str, Any], fake_queue: Any
) -> None:
    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(tenant_world["section_b"]), "course_name": "HACK"},
        files={"image": ("photo.jpg", make_jpeg_bytes(), "image/jpeg")},
        headers=auth(token),
    )
    assert resp.status_code == 403
    assert fake_queue.messages == []  # nothing was enqueued cross-tenant


async def test_dept_admin_cannot_read_other_dept_review_queue(
    client: AsyncClient, tenant_world: dict[str, Any], db_session: Any
) -> None:
    # Seed a pending review item in section A's session.
    cs = await dao.create_class_session(
        db_session,
        section_id=tenant_world["section_a"],
        course_name="X",
        session_date=date.today(),
        admin_id=None,
    )
    await dao.enqueue_review(db_session, session_id=cs.session_id)
    await db_session.commit()

    token_b = await login_admin(client, "admin_b", "pw-b-password")
    resp = await client.get("/admin/review-queue", headers=auth(token_b))
    assert resp.status_code == 200
    assert all(item["section_id"] == str(tenant_world["section_b"]) for item in resp.json())

    # Directly requesting section A's queue is also blocked.
    resp = await client.get(
        f"/admin/review-queue?section_id={tenant_world['section_a']}", headers=auth(token_b)
    )
    assert resp.status_code == 403


async def test_dept_admin_cannot_read_other_dept_attendance(
    client: AsyncClient, tenant_world: dict[str, Any]
) -> None:
    token_b = await login_admin(client, "admin_b", "pw-b-password")
    resp = await client.get(
        f"/admin/attendance?section_id={tenant_world['section_a']}&date={date.today()}",
        headers=auth(token_b),
    )
    assert resp.status_code == 403


# ─── Capture flow ────────────────────────────────────────────────────────────


async def test_capture_enqueues_job_and_returns_immediately(
    client: AsyncClient,
    tenant_world: dict[str, Any],
    fake_queue: Any,
) -> None:
    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(tenant_world["section_a"]), "course_name": "DBMS"},
        files={"image": ("photo.jpg", make_jpeg_bytes(), "image/jpeg")},
        headers=auth(token),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    assert len(fake_queue.messages) == 1

    msg = fake_queue.messages[0]
    assert isinstance(msg, CaptureJobMessage)
    assert msg.section_id == tenant_world["section_a"]
    assert msg.image_url.startswith("/static/session-images/")

    # Poll: status is visible in Redis.
    status_resp = await client.get(f"/admin/session/capture/{body['job_id']}", headers=auth(token))
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "queued"


async def test_capture_rejects_non_image(client: AsyncClient, tenant_world: dict[str, Any]) -> None:
    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(tenant_world["section_a"])},
        files={"image": ("notes.txt", b"hello", "text/plain")},
        headers=auth(token),
    )
    assert resp.status_code == 415


async def test_capture_requires_auth(client: AsyncClient, tenant_world: dict[str, Any]) -> None:
    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(tenant_world["section_a"])},
        files={"image": ("photo.jpg", make_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 401


async def test_capture_rate_limited(
    client: AsyncClient,
    tenant_world: dict[str, Any],
    fake_queue: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "rate_limit_capture_rpm", 2)
    token = await login_admin(client, "admin_a", "pw-a-password")
    for _ in range(2):
        resp = await client.post(
            "/admin/session/capture",
            data={"section_id": str(tenant_world["section_a"])},
            files={"image": ("photo.jpg", make_jpeg_bytes(), "image/jpeg")},
            headers=auth(token),
        )
        assert resp.status_code == 200
    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(tenant_world["section_a"])},
        files={"image": ("photo.jpg", make_jpeg_bytes(), "image/jpeg")},
        headers=auth(token),
    )
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


# ─── Capture resolution floor ─────────────────────────────────────────────────


async def test_capture_rejects_low_resolution_photo(
    client: AsyncClient,
    tenant_world: dict[str, Any],
) -> None:
    """Below the 1024px floor: 422 with an actionable message."""
    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(tenant_world["section_a"])},
        files={"image": ("lowres.jpg", make_jpeg_bytes(size=640), "image/jpeg")},
        headers=auth(token),
    )
    assert resp.status_code == 422
    assert "resolution too low" in resp.json()["detail"]
    assert "1024" in resp.json()["detail"]


async def test_capture_accepts_boundary_resolution_photo(
    client: AsyncClient,
    tenant_world: dict[str, Any],
    fake_queue: Any,
) -> None:
    """Exactly at the floor (1024px shorter side) passes."""
    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(tenant_world["section_a"])},
        files={"image": ("hd.jpg", make_jpeg_bytes(size=1024), "image/jpeg")},
        headers=auth(token),
    )
    assert resp.status_code == 200, resp.text
    assert len(fake_queue.messages) == 1


async def test_capture_rejects_undecodable_image(
    client: AsyncClient,
    tenant_world: dict[str, Any],
) -> None:
    """Valid content-type header but garbage bytes: 422, not 500."""
    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(tenant_world["section_a"])},
        files={"image": ("corrupt.jpg", b"\xff\xd8garbage-not-a-jpeg", "image/jpeg")},
        headers=auth(token),
    )
    assert resp.status_code == 422
    assert "decode" in resp.json()["detail"].lower()


# ─── Review queue resolve ────────────────────────────────────────────────────


async def test_resolve_assigns_roster_student_and_marks_attendance(
    client: AsyncClient,
    tenant_world: dict[str, Any],
    db_session: Any,
) -> None:
    cs = await dao.create_class_session(
        db_session,
        section_id=tenant_world["section_a"],
        course_name="Review",
        session_date=date.today(),
        admin_id=None,
    )
    review = await dao.enqueue_review(
        db_session,
        session_id=cs.session_id,
        best_guess_student_id=tenant_world["student_a"].student_id,
        best_guess_distance=0.61,
    )
    await db_session.commit()

    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.post(
        f"/admin/review-queue/{review.review_id}/resolve",
        json={"action": "assign", "student_id": str(tenant_world["student_a"].student_id)},
        headers=auth(token),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["resolved_student_id"] == str(tenant_world["student_a"].student_id)

    # Assignment also marks attendance for the session.
    rows = await dao.get_attendance_for_date(
        db_session, section_id=tenant_world["section_a"], session_date=date.today()
    )
    marked = [r for r in rows if r.student_id == tenant_world["student_a"].student_id]
    assert len(marked) == 1 and marked[0].marked_at is not None


async def test_resolve_rejects_student_outside_roster(
    client: AsyncClient,
    tenant_world: dict[str, Any],
    db_session: Any,
) -> None:
    cs = await dao.create_class_session(
        db_session,
        section_id=tenant_world["section_a"],
        course_name="Review",
        session_date=date.today(),
        admin_id=None,
    )
    review = await dao.enqueue_review(db_session, session_id=cs.session_id)
    await db_session.commit()

    token = await login_admin(client, "admin_a", "pw-a-password")
    # student_b is in section B — NOT in section A's roster.
    resp = await client.post(
        f"/admin/review-queue/{review.review_id}/resolve",
        json={"action": "assign", "student_id": str(tenant_world["student_b"].student_id)},
        headers=auth(token),
    )
    assert resp.status_code == 422


async def test_resolve_discard(
    client: AsyncClient, tenant_world: dict[str, Any], db_session: Any
) -> None:
    cs = await dao.create_class_session(
        db_session,
        section_id=tenant_world["section_a"],
        course_name="Review",
        session_date=date.today(),
        admin_id=None,
    )
    review = await dao.enqueue_review(db_session, session_id=cs.session_id)
    await db_session.commit()

    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.post(
        f"/admin/review-queue/{review.review_id}/resolve",
        json={"action": "discard"},
        headers=auth(token),
    )
    assert resp.status_code == 200
    assert resp.json()["resolved_student_id"] is None


# ─── Attendance report ───────────────────────────────────────────────────────


async def test_attendance_report_shape(
    client: AsyncClient, tenant_world: dict[str, Any], db_session: Any
) -> None:
    cs = await dao.create_class_session(
        db_session,
        section_id=tenant_world["section_a"],
        course_name="Report",
        session_date=date.today(),
        admin_id=None,
    )
    await dao.mark_attendance(
        db_session,
        student_id=tenant_world["student_a"].student_id,
        session_id=cs.session_id,
        session_date=date.today(),
        match_distance=0.30,
    )
    await db_session.commit()

    token = await login_admin(client, "admin_a", "pw-a-password")
    resp = await client.get(
        f"/admin/attendance?section_id={tenant_world['section_a']}&date={date.today()}",
        headers=auth(token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_enrolled"] == 1
    assert body["present"] == 1
    assert body["rows"][0]["roll_number"] == "TA001"


# ─── Student attendance ──────────────────────────────────────────────────────


async def test_student_attendance_me(
    client: AsyncClient, tenant_world: dict[str, Any], db_session: Any
) -> None:
    cs = await dao.create_class_session(
        db_session,
        section_id=tenant_world["section_a"],
        course_name="Hist",
        session_date=date.today(),
        admin_id=None,
    )
    await dao.mark_attendance(
        db_session,
        student_id=tenant_world["student_a"].student_id,
        session_id=cs.session_id,
        session_date=date.today(),
    )
    await db_session.commit()

    login = await client.post(
        "/auth/student/login", json={"roll_number": "TA001", "password": "student-pw"}
    )
    token = login.json()["access_token"]
    resp = await client.get("/attendance/me", headers=auth(token))
    assert resp.status_code == 200
    records = resp.json()["records"]
    assert len(records) == 1
    assert records[0]["course_name"] == "Hist"
