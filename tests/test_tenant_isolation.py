"""
tests/test_tenant_isolation.py
──────────────────────────────
Phase 5 requirement: for EVERY admin-scoped endpoint, assert that a token
scoped to institution/department A cannot read or write data belonging to
institution/department B.

World: two institutions (A, B), each with a department, a section, students,
a class session with a pending review item and a queued capture job. Three
token flavours are exercised:
    - dept_admin_a   (department scope, institution A)
    - inst_admin_a   (institution scope, institution A)
    - superadmin     (global)
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pytest
from httpx import AsyncClient

from backend.core.security import hash_password
from backend.db import dao
from backend.db.models import AdminRole, Enrollment


@pytest.fixture
async def isolation_world(db_session: Any) -> dict[str, Any]:
    ids: dict[str, Any] = {}
    passwords = {"dept": "dept-pass-123", "inst": "inst-pass-123"}

    for tag in ("a", "b"):
        inst = await dao.create_institution(db_session, name=f"Inst {tag.upper()}")
        dept = await dao.create_department(
            db_session, institution_id=inst.institution_id, name=f"Dept {tag.upper()}"
        )
        section = await dao.create_section(
            db_session,
            department_id=dept.department_id,
            name=f"SEC-{tag.upper()}",
            academic_year=2026,
        )
        await dao.create_admin(
            db_session,
            username=f"iso_dept_{tag}",
            password_hash=hash_password(passwords["dept"] + tag),
            role=AdminRole.department_admin,
            institution_id=inst.institution_id,
            department_id=dept.department_id,
        )
        await dao.create_admin(
            db_session,
            username=f"iso_inst_{tag}",
            password_hash=hash_password(passwords["inst"] + tag),
            role=AdminRole.institution_admin,
            institution_id=inst.institution_id,
            department_id=None,
        )
        student = await dao.create_student(
            db_session,
            roll_number=f"ISO{tag.upper()}1",
            name=f"Iso Student {tag.upper()}",
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
        cs = await dao.create_class_session(
            db_session,
            section_id=section.section_id,
            course_name="ISO",
            session_date=date.today(),
            admin_id=None,
        )
        review = await dao.enqueue_review(db_session, session_id=cs.session_id)

        ids[f"inst_{tag}"] = inst.institution_id
        ids[f"dept_{tag}"] = dept.department_id
        ids[f"section_{tag}"] = section.section_id
        ids[f"student_{tag}"] = student.student_id
        ids[f"session_{tag}"] = cs.session_id
        ids[f"review_{tag}"] = review.review_id
        ids[f"job_{tag}"] = uuid.uuid4()  # simulated queued job in section

    await db_session.commit()
    ids["passwords"] = {"dept": passwords["dept"], "inst": passwords["inst"]}
    return ids


async def _login(client: AsyncClient, username: str, password: str) -> str:
    resp = await client.post("/auth/admin/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def tokens(
    client: AsyncClient, redis_client: Any, isolation_world: dict[str, Any]
) -> dict[str, str]:
    """Login all three admin flavours; also seed job status/scope in Redis."""
    pw = isolation_world["passwords"]
    t = {
        "dept_a": await _login(client, "iso_dept_a", pw["dept"] + "a"),
        "dept_b": await _login(client, "iso_dept_b", pw["dept"] + "b"),
        "inst_a": await _login(client, "iso_inst_a", pw["inst"] + "a"),
    }
    # Seed Redis job status + tenant marker for both sections' jobs.
    for tag in ("a", "b"):
        job_id = str(ids_job(isolation_world, tag))
        await redis_client.set(f"job:{job_id}", '{"status": "queued"}')
        await redis_client.set(f"job:{job_id}:section", str(isolation_world[f"section_{tag}"]))
    return t


def ids_job(world: dict[str, Any], tag: str) -> uuid.UUID:
    return world[f"job_{tag}"]


# ─── Cross-tenant matrix: dept_admin A against everything of dept B ──────────


async def test_capture_section_b_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    import io

    resp = await client.post(
        "/admin/session/capture",
        data={"section_id": str(isolation_world["section_b"])},
        files={"image": ("p.jpg", io.BytesIO(b"\xff\xd8fake").getvalue(), "image/jpeg")},
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


async def test_job_status_of_b_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.get(
        f"/admin/session/capture/{ids_job(isolation_world, 'b')}",
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


async def test_own_job_status_allowed(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.get(
        f"/admin/session/capture/{ids_job(isolation_world, 'a')}",
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


async def test_review_queue_filter_b_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.get(
        f"/admin/review-queue?section_id={isolation_world['section_b']}",
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


async def test_resolve_b_review_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.post(
        f"/admin/review-queue/{isolation_world['review_b']}/resolve",
        json={"action": "discard"},
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


async def test_search_all_b_review_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.post(
        f"/admin/review-queue/{isolation_world['review_b']}/search-all",
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


async def test_attendance_report_b_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.get(
        f"/admin/attendance?section_id={isolation_world['section_b']}&date={date.today()}",
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


async def test_roster_b_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.get(
        f"/admin/sections/{isolation_world['section_b']}/students",
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


async def test_bulk_enroll_b_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.post(
        f"/admin/sections/{isolation_world['section_b']}/bulk-enroll",
        files={"file": ("r.csv", b"X001\nX002\n", "text/csv")},
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


async def test_create_section_in_b_forbidden(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.post(
        "/admin/sections",
        json={
            "department_id": str(isolation_world["dept_b"]),
            "name": "HACK",
            "academic_year": 2026,
        },
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


# ─── Institution-scope checks ────────────────────────────────────────────────


async def test_inst_admin_a_cannot_create_department_in_b(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.post(
        "/admin/departments",
        json={"institution_id": str(isolation_world["inst_b"]), "name": "HACK"},
        headers=_auth(tokens["inst_a"]),
    )
    assert resp.status_code == 403


async def test_inst_admin_a_cannot_create_admin_in_b(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.post(
        "/admin/admins",
        json={
            "username": "hacker_admin",
            "password": "whatever-123",
            "role": "department_admin",
            "institution_id": str(isolation_world["inst_b"]),
            "department_id": str(isolation_world["dept_b"]),
        },
        headers=_auth(tokens["inst_a"]),
    )
    assert resp.status_code == 403


async def test_inst_admin_a_cannot_purge_student_in_b(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.post(
        f"/admin/students/{isolation_world['student_b']}/purge-biometrics",
        json={"attendance_action": "delete"},
        headers=_auth(tokens["inst_a"]),
    )
    assert resp.status_code == 403


async def test_dept_admin_cannot_purge_anyone(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.post(
        f"/admin/students/{isolation_world['student_a']}/purge-biometrics",
        json={"attendance_action": "delete"},
        headers=_auth(tokens["dept_a"]),
    )
    assert resp.status_code == 403


# ─── Listing endpoints never leak other tenants' data ────────────────────────


async def test_sections_list_scoped_to_department(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.get("/admin/sections", headers=_auth(tokens["dept_a"]))
    assert resp.status_code == 200
    returned = {s["section_id"] for s in resp.json()}
    assert returned == {str(isolation_world["section_a"])}


async def test_departments_list_scoped_to_institution(
    client: AsyncClient, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    resp = await client.get("/admin/departments", headers=_auth(tokens["inst_a"]))
    assert resp.status_code == 200
    returned = {d["department_id"] for d in resp.json()}
    assert returned == {str(isolation_world["dept_a"])}


async def test_superadmin_sees_everything(
    client: AsyncClient, db_session: Any, isolation_world: dict[str, Any], tokens: dict[str, str]
) -> None:
    # Create a real super admin directly through the DAO.
    await dao.create_admin(
        db_session,
        username="iso_super",
        password_hash=hash_password("super-pass-123"),
        role=AdminRole.super_admin,
        institution_id=None,
        department_id=None,
    )
    await db_session.commit()
    token = await _login(client, "iso_super", "super-pass-123")
    resp = await client.get("/admin/sections", headers=_auth(token))
    assert resp.status_code == 200
    returned = {s["section_id"] for s in resp.json()}
    assert returned == {
        str(isolation_world["section_a"]),
        str(isolation_world["section_b"]),
    }
