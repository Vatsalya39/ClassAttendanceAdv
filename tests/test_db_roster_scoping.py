"""
tests/test_db_roster_scoping.py
───────────────────────────────
Phase 1 acceptance test (a): roster-scoped vector search only ever considers
embeddings belonging to students enrolled in the given section — even when
other sections' embeddings exist in the same table and are strictly closer
to the query.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np
import pytest

from backend.db import dao
from backend.db.models import Enrollment

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


async def seed_two_sections(session: AsyncSession) -> dict[str, uuid.UUID]:
    """Institution with two sections; each student gets one embedding."""
    inst = await dao.create_institution(session, name="Test University")
    dept = await dao.create_department(
        session, institution_id=inst.institution_id, name="Computer Science"
    )
    sec_a = await dao.create_section(
        session, department_id=dept.department_id, name="CSE-3A", academic_year=2026
    )
    sec_b = await dao.create_section(
        session, department_id=dept.department_id, name="CSE-3B", academic_year=2026
    )

    ids: dict[str, uuid.UUID] = {
        "institution": inst.institution_id,
        "section_a": sec_a.section_id,
        "section_b": sec_b.section_id,
    }

    for tag, section_id, base_seed in (
        ("a", sec_a.section_id, 100),
        ("b", sec_b.section_id, 200),
    ):
        for i in range(4):
            student = await dao.create_student(
                session,
                roll_number=f"{tag.upper()}{i:03d}",
                name=f"Student {tag}{i}",
                password_hash="argon2$fake$hash",
                institution_id=inst.institution_id,
            )
            emb = await dao.add_embedding(
                session, student_id=student.student_id, embedding=unit_vec(base_seed + i)
            )
            session.add(
                Enrollment(
                    enrollment_id=uuid.uuid4(),
                    student_id=student.student_id,
                    section_id=section_id,
                )
            )
            await session.commit()
            ids[f"{tag}_student_{i}"] = student.student_id
            ids[f"{tag}_emb_{i}"] = emb.embedding_id

    return ids


async def test_roster_search_never_crosses_section(db_session: AsyncSession) -> None:
    ids = await seed_two_sections(db_session)

    # Query = section-A student 0's embedding + tiny noise ⇒ that student is
    # the closest embedding in the WHOLE table.
    base = unit_vec(100)
    noise = np.random.default_rng(999).standard_normal(512).astype(np.float32)
    noise *= 0.01 / np.linalg.norm(noise)
    query = (np.asarray(base) + noise).tolist()

    # Scoped to section A → finds A0.
    in_a = await dao.find_nearest_embeddings_for_roster(
        db_session, query, section_id=ids["section_a"], top_k=5
    )
    assert in_a, "roster search returned no candidates"
    assert in_a[0].student_id == ids["a_student_0"]

    # Scoped to section B → must NEVER return any section-A student, even
    # though A0's embedding is globally closest to the query.
    in_b = await dao.find_nearest_embeddings_for_roster(
        db_session, query, section_id=ids["section_b"], top_k=10
    )
    b_students = {ids[f"b_student_{i}"] for i in range(4)}
    returned = {m.student_id for m in in_b}
    assert returned <= b_students, "roster-scoped search leaked embeddings from another section!"


async def test_roster_search_empty_section_returns_nothing(
    db_session: AsyncSession,
) -> None:
    inst = await dao.create_institution(db_session, name="Empty U")
    dept = await dao.create_department(db_session, institution_id=inst.institution_id, name="D")
    empty_section = await dao.create_section(
        db_session, department_id=dept.department_id, name="EMPTY-1", academic_year=2026
    )
    results = await dao.find_nearest_embeddings_for_roster(
        db_session, unit_vec(1), section_id=empty_section.section_id, top_k=5
    )
    assert results == []


async def test_institution_wide_finds_global_closest(db_session: AsyncSession) -> None:
    ids = await seed_two_sections(db_session)

    query = unit_vec(202)
    matches = await dao.find_nearest_embeddings_institution_wide(
        db_session, query, institution_id=ids["institution"], top_k=5
    )
    assert matches[0].student_id == ids["b_student_2"]
    # All candidates belong to this institution's students.
    allowed = {ids[f"{t}_student_{i}"] for t in ("a", "b") for i in range(4)}
    assert {m.student_id for m in matches} <= allowed


async def test_bulk_enroll_is_idempotent(db_session: AsyncSession) -> None:
    ids = await seed_two_sections(db_session)
    students = [ids[f"a_student_{i}"] for i in range(4)]

    enrolled_first, _ = await dao.bulk_enroll(
        db_session, section_id=ids["section_a"], student_ids=students
    )
    await db_session.commit()
    assert enrolled_first == 0  # seed_two_sections already enrolled them

    # New students not yet enrolled anywhere
    new_ids = []
    for i in range(3):
        s = await dao.create_student(
            db_session,
            roll_number=f"N{i:03d}",
            name=f"New {i}",
            password_hash="x",
            institution_id=ids["institution"],
        )
        new_ids.append(s.student_id)
    await db_session.commit()

    enrolled_now, skipped = await dao.bulk_enroll(
        db_session, section_id=ids["section_a"], student_ids=new_ids
    )
    await db_session.commit()
    assert enrolled_now == 3
    assert skipped == 0

    enrolled_again, skipped_again = await dao.bulk_enroll(
        db_session, section_id=ids["section_a"], student_ids=new_ids
    )
    await db_session.commit()
    assert enrolled_again == 0
    assert skipped_again == 3


async def test_bulk_enroll_rejects_cross_tenant(db_session: AsyncSession) -> None:
    ids = await seed_two_sections(db_session)

    other_inst = await dao.create_institution(db_session, name="Other University")
    outsider = await dao.create_student(
        db_session,
        roll_number="OUT001",
        name="Outsider",
        password_hash="x",
        institution_id=other_inst.institution_id,
    )
    await db_session.commit()

    with pytest.raises(dao.StudentWrongInstitutionError):
        await dao.bulk_enroll(
            db_session,
            section_id=ids["section_a"],
            student_ids=[outsider.student_id],
        )
