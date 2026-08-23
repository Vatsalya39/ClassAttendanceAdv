"""
tests/test_public_and_roster_endpoints.py
─────────────────────────────────────────
Phase 4 additions:
  - GET /public/sections (no auth, names only — no student data)
  - GET /admin/sections/{id}/students (scoped roster for review assignment)
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from httpx import AsyncClient

from backend.core.security import hash_password
from backend.db import dao
from backend.db.models import AdminRole, Enrollment


@pytest.fixture
async def roster_world(db_session: Any) -> dict[str, Any]:
    inst = await dao.create_institution(db_session, name="Roster U")
    dept = await dao.create_department(
        db_session, institution_id=inst.institution_id, name="Dept R"
    )
    section = await dao.create_section(
        db_session, department_id=dept.department_id, name="RS-1", academic_year=2026
    )
    admin = await dao.create_admin(
        db_session,
        username="roster_admin",
        password_hash=hash_password("pw-roster-admin"),
        role=AdminRole.department_admin,
        institution_id=inst.institution_id,
        department_id=dept.department_id,
    )
    students = []
    for i in range(2):
        s = await dao.create_student(
            db_session,
            roll_number=f"R{i:03d}",
            name=f"Roster Student {i}",
            password_hash="x",
            institution_id=inst.institution_id,
        )
        db_session.add(
            Enrollment(
                enrollment_id=uuid.uuid4(),
                student_id=s.student_id,
                section_id=section.section_id,
            )
        )
        students.append(s)
    # One student NOT enrolled anywhere (must not appear in the roster).
    outsider = await dao.create_student(
        db_session,
        roll_number="OUT",
        name="Outsider",
        password_hash="x",
        institution_id=inst.institution_id,
    )
    await db_session.commit()
    return {
        "section": section,
        "admin": admin,
        "students": students,
        "outsider": outsider,
    }


async def _admin_headers(client: AsyncClient) -> dict[str, str]:
    resp = await client.post(
        "/auth/admin/login", json={"username": "roster_admin", "password": "pw-roster-admin"}
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def test_public_sections_lists_names_without_auth(
    client: AsyncClient, roster_world: dict[str, Any]
) -> None:
    resp = await client.get("/public/sections")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["name"] == "RS-1"
    assert items[0]["institution_name"] == "Roster U"
    assert items[0]["department_name"] == "Dept R"
    # No student data may ever leak through this endpoint.
    assert "students" not in items[0]
    assert "roll_number" not in str(items)


async def test_section_roster_returns_only_enrolled(
    client: AsyncClient, roster_world: dict[str, Any]
) -> None:
    headers = await _admin_headers(client)
    resp = await client.get(
        f"/admin/sections/{roster_world['section'].section_id}/students", headers=headers
    )
    assert resp.status_code == 200
    rows = resp.json()
    roll_numbers = {r["roll_number"] for r in rows}
    assert roll_numbers == {"R000", "R001"}
    assert "OUT" not in roll_numbers


async def test_section_roster_requires_auth(
    client: AsyncClient, roster_world: dict[str, Any]
) -> None:
    resp = await client.get(f"/admin/sections/{roster_world['section'].section_id}/students")
    assert resp.status_code == 401


async def test_openapi_has_no_spurious_request_params() -> None:
    """
    Regression: get_db(request: Request) imported Request only under
    TYPE_CHECKING, which made FastAPI turn it into a required *query* param
    on every route in production (tests masked it via dependency overrides).
    The served OpenAPI schema must contain no such artifact.
    """
    from backend.api.main import app

    spec = app.openapi()
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue
            bogus = [
                p
                for p in op.get("parameters", [])
                if p["name"] == "request"
                or p.get("schema", {}).get("type") == "object"
                and p["in"] == "query"
            ]
            assert not bogus, f"{method.upper()} {path} has bogus params: {bogus}"
