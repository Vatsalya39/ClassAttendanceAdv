"""
tests/test_capture_isolation.py
───────────────────────────────
Phase 3 acceptance test: captures submitted for two different sections
CONCURRENTLY must each only match against their own roster — the core
roster-scoping rule exercised through the same service functions the worker
uses (services.capture.match_faces_to_roster / write_capture_results).

No InsightFace needed: synthetic embeddings play the role of batch_process
output, which is exactly the contract between ml.pipeline and this layer.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date
from typing import Any

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.db import dao
from backend.db.models import Enrollment
from backend.services.capture import match_faces_to_roster, write_capture_results

THRESHOLD = 0.45


def unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


def perturb(vec: list[float], seed: int, scale: float = 0.02) -> list[float]:
    """A noisy capture of the same face stays well under threshold."""
    noise = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    noise *= scale / np.linalg.norm(noise)
    return (np.asarray(vec) + noise).tolist()


@pytest.fixture
async def two_sections(db_session: AsyncSession) -> dict[str, uuid.UUID]:
    inst = await dao.create_institution(db_session, name="Isolation U")
    dept = await dao.create_department(db_session, institution_id=inst.institution_id, name="CS")
    ids: dict[str, uuid.UUID] = {}
    for tag, seed_base in (("a", 10), ("b", 20)):
        section = await dao.create_section(
            db_session,
            department_id=dept.department_id,
            name=f"CSE-3{tag.upper()}",
            academic_year=2026,
        )
        ids[f"section_{tag}"] = section.section_id
        for i in range(3):
            student = await dao.create_student(
                db_session,
                roll_number=f"{tag.upper()}{i}",
                name=f"Student {tag}{i}",
                password_hash="x",
                institution_id=inst.institution_id,
            )
            await dao.add_embedding(
                db_session, student_id=student.student_id, embedding=unit_vec(seed_base + i)
            )
            db_session.add(
                Enrollment(
                    enrollment_id=uuid.uuid4(),
                    student_id=student.student_id,
                    section_id=section.section_id,
                )
            )
            ids[f"{tag}_student_{i}"] = student.student_id
    await db_session.commit()
    return ids


async def test_concurrent_captures_never_cross_rosters(
    db_session: AsyncSession,
    db_engine: Any,
    two_sections: dict[str, uuid.UUID],
) -> None:
    """
    Both sections "capture" at the same instant — gathered coroutines, each
    with its OWN session, exactly as concurrent worker jobs would run.
    Section A's photo contains faces of A-students; B's of B-students.
    """
    factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async def capture_for(
        section_key: str, photo: list[tuple[tuple[int, int, int, int], list[float]]]
    ) -> list:
        async with factory() as job_session:
            return await match_faces_to_roster(
                job_session,
                section_id=two_sections[section_key],
                face_embeddings=photo,
                threshold=THRESHOLD,
            )

    photo_a = [
        ((100, 100, 200, 200), perturb(unit_vec(10), 101)),  # A0 present
        ((300, 100, 400, 200), perturb(unit_vec(11), 111)),  # A1 present
    ]
    photo_b = [
        ((100, 100, 200, 200), perturb(unit_vec(20), 201)),  # B0 present
        ((500, 300, 600, 400), perturb(unit_vec(21), 211)),  # B1 present
    ]

    outcomes_a, outcomes_b = await asyncio.gather(
        capture_for("section_a", photo_a),
        capture_for("section_b", photo_b),
    )

    # Every face matched within its own roster.
    assert all(o.matched for o in outcomes_a)
    assert all(o.matched for o in outcomes_b)

    matched_ids_a = {o.student_id for o in outcomes_a}
    matched_ids_b = {o.student_id for o in outcomes_b}
    expected_a = {two_sections["a_student_0"], two_sections["a_student_1"]}
    expected_b = {two_sections["b_student_0"], two_sections["b_student_1"]}

    assert matched_ids_a == expected_a
    assert matched_ids_b == expected_b
    assert matched_ids_a.isdisjoint(matched_ids_b), "cross-section match leaked!"


async def test_unknown_face_goes_to_review_not_forced_match(
    db_session: AsyncSession, two_sections: dict[str, uuid.UUID]
) -> None:
    """An intruder's face (not close to any roster member) must land in review.

    Module 3: the face is never force-matched — but the nearest roster
    student is attached as a best guess (with its distance) so the admin's
    review UI can suggest a candidate. No attendance may be written.
    """
    intruder_face = [((1, 1, 50, 50), unit_vec(9999))]

    session_row = await dao.create_class_session(
        db_session,
        section_id=two_sections["section_a"],
        course_name="Security",
        session_date=date.today(),
        admin_id=None,
    )
    outcomes = await match_faces_to_roster(
        db_session,
        section_id=two_sections["section_a"],
        face_embeddings=[(intruder_face[0][0], unit_vec(9999))],
        threshold=THRESHOLD,
    )
    assert len(outcomes) == 1
    assert not outcomes[0].matched
    # Above-threshold faces keep their best guess + distance for the review UI.
    assert outcomes[0].review_reason == "above-threshold"

    outcome = await write_capture_results(
        db_session,
        session_id=session_row.session_id,
        outcomes=outcomes,
        crop_urls={0: "/static/review-crops/x.jpg"},
    )
    await db_session.commit()

    # Module 3 invariant: every face lands in exactly one bucket.
    assert outcome.faces_total == 1
    assert not outcome.matched
    assert len(outcome.review_needed) == 1
    assert outcome.faces_total == len(outcome.matched) + len(outcome.review_needed)

    from sqlalchemy import func, select

    from backend.db.models import Attendance, ReviewQueue

    rows = list((await db_session.execute(select(ReviewQueue))).scalars())
    assert len(rows) == 1
    assert rows[0].status.value == "pending"
    # Best guess is attached (nearest roster student) with its distance…
    assert rows[0].best_guess_student_id is not None
    assert rows[0].best_guess_distance is not None
    # …but NO attendance row may exist for the guessed student.
    attendance_count = (
        await db_session.execute(select(func.count()).select_from(Attendance))
    ).scalar_one()
    assert attendance_count == 0


async def test_reprocessing_same_job_does_not_double_write(
    db_session: AsyncSession, two_sections: dict[str, uuid.UUID]
) -> None:
    """Idempotency: re-running write_capture_results must not duplicate rows."""
    session_row = await dao.create_class_session(
        db_session,
        section_id=two_sections["section_a"],
        course_name="DBMS",
        session_date=date.today(),
        admin_id=None,
    )
    outcomes = await match_faces_to_roster(
        db_session,
        section_id=two_sections["section_a"],
        face_embeddings=[
            ((10, 10, 90, 90), perturb(unit_vec(10), 11)),
            ((110, 10, 190, 90), perturb(unit_vec(11), 12)),
        ],
        threshold=THRESHOLD,
    )

    first = await write_capture_results(
        db_session, session_id=session_row.session_id, outcomes=outcomes
    )
    await db_session.commit()
    second = await write_capture_results(
        db_session, session_id=session_row.session_id, outcomes=outcomes
    )
    await db_session.commit()

    assert len(first.matched) == 2  # inserted now
    # Module 3 semantic change: on re-processing, faces that match cleanly
    # are still accounted as matched (the burst-union semantics) even when
    # the attendance insert is conflict-skipped — the STUDENT is present,
    # the write is simply idempotent. Attendance rows must not duplicate.
    assert len(second.matched) == 2
    assert len(second.review_needed) == 0

    from sqlalchemy import func, select

    from backend.db.models import Attendance

    count = (
        await db_session.execute(
            select(func.count())
            .select_from(Attendance)
            .where(Attendance.session_id == session_row.session_id)
        )
    ).scalar_one()
    assert count == 2
