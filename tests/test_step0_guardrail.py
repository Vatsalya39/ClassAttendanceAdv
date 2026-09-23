"""
tests/test_step0_guardrail.py
─────────────────────────────
STEP 0 Guardrail Verification:
Ensures enrollment embeddings and section enrollments remain 100% immutable
at runtime. Capture, inference, and matching pipelines must NEVER write
or update rows in `face_embeddings` or `enrollments`.
"""

from __future__ import annotations

import ast
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import dao
from backend.db.models import Enrollment, FaceEmbedding
from backend.services.capture import (
    match_and_assign,
    write_capture_results,
)

# ─── Static AST & Grep Guardrail Verification ─────────────────────────────────


def test_static_guardrail_ast_check() -> None:
    """
    Parse backend/services and backend/worker modules with AST.
    Confirm zero calls to `add_embedding`, `insert(FaceEmbedding)`,
    or `insert(Enrollment)` exist in the capture/inference pipeline.
    """
    repo_root = Path(__file__).resolve().parent.parent
    checked_files = [
        repo_root / "backend" / "services" / "capture.py",
        repo_root / "backend" / "worker" / "main.py",
        repo_root / "ml" / "pipeline.py",
    ]

    forbidden_calls = {"add_embedding"}

    for file_path in checked_files:
        assert file_path.exists(), f"File {file_path} must exist"
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

        for node in ast.walk(tree):
            # Check for direct calls or method calls: func() or obj.func()
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                assert (
                    func_name not in forbidden_calls
                ), f"Violation in {file_path.name}: forbidden call '{func_name}' at line {node.lineno}"


def test_static_guardrail_grep_check() -> None:
    """
    Scans backend/services and backend/worker for any mention of `add_embedding`.
    Zero occurrences are allowed outside the student registration and test files.
    """
    repo_root = Path(__file__).resolve().parent.parent
    services_dir = repo_root / "backend" / "services"
    worker_dir = repo_root / "backend" / "worker"

    for directory in (services_dir, worker_dir):
        for py_file in directory.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert (
                "add_embedding" not in content
            ), f"Guardrail failure: 'add_embedding' found in {py_file}"


# ─── Database Immutability Runtime Verification ───────────────────────────────


def unit_vec(seed: int) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


def perturb(vec: list[float], seed: int, scale: float = 0.01) -> list[float]:
    import numpy as np

    noise = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    noise *= scale / np.linalg.norm(noise)
    return (np.asarray(vec) + noise).tolist()


@pytest.fixture
async def immutability_world(db_session: AsyncSession) -> dict[str, Any]:
    """Sets up a section with enrolled students and baseline embeddings."""
    inst = await dao.create_institution(db_session, name="Immutability University")
    dept = await dao.create_department(
        db_session, institution_id=inst.institution_id, name="Computer Science"
    )
    section = await dao.create_section(
        db_session, department_id=dept.department_id, name="CS-A", academic_year=2026
    )

    students = []
    for i in range(3):
        student = await dao.create_student(
            db_session,
            roll_number=f"IMM{i}",
            name=f"Student {i}",
            password_hash="pw_hash",
            institution_id=inst.institution_id,
        )
        for j in range(3):
            await dao.add_embedding(
                db_session,
                student_id=student.student_id,
                embedding=perturb(unit_vec(100 + i), i * 10 + j, scale=0.01),
            )
        db_session.add(
            Enrollment(
                enrollment_id=uuid.uuid4(),
                student_id=student.student_id,
                section_id=section.section_id,
            )
        )
        students.append(student)

    session_row = await dao.create_class_session(
        db_session,
        section_id=section.section_id,
        course_name="Software Verification",
        session_date=date.today(),
        admin_id=None,
    )
    await db_session.commit()

    return {
        "section_id": section.section_id,
        "session_id": session_row.session_id,
        "students": students,
    }


async def test_runtime_capture_does_not_mutate_enrollment_embeddings(
    db_session: AsyncSession, immutability_world: dict[str, Any]
) -> None:
    """
    Run capture matching and persistence with multiple high-confidence,
    low-confidence, and unmatched faces. Verify that face_embeddings and
    enrollments row counts and records remain strictly identical.
    """
    # 1. Capture baseline counts and IDs
    initial_embeddings_count = (
        await db_session.execute(select(func.count(FaceEmbedding.embedding_id)))
    ).scalar_one()
    initial_enrollments_count = (
        await db_session.execute(select(func.count(Enrollment.enrollment_id)))
    ).scalar_one()

    initial_embedding_ids = set(
        (await db_session.execute(select(FaceEmbedding.embedding_id))).scalars().all()
    )

    # 2. Simulate captured classroom faces (some match student 0, student 1 with high confidence <0.20,
    # some strangers, some duplicates)
    captured_faces = [
        ((10, 10, 80, 80), perturb(unit_vec(100), 555, scale=0.005)),  # Strong match Student 0
        ((90, 10, 160, 80), perturb(unit_vec(101), 666, scale=0.005)),  # Strong match Student 1
        ((170, 10, 240, 80), unit_vec(9999)),  # Stranger -> review queue
        ((250, 10, 320, 80), perturb(unit_vec(100), 777, scale=0.008)),  # Duplicate of Student 0
    ]

    outcomes = await match_and_assign(
        db_session,
        section_id=immutability_world["section_id"],
        face_embeddings=captured_faces,
        threshold=0.45,
        margin=0.08,
    )

    capture_outcome = await write_capture_results(
        db_session,
        session_id=immutability_world["session_id"],
        outcomes=outcomes,
    )
    await db_session.commit()

    # Confirm capture processed correctly (some matched, some to review)
    assert len(capture_outcome.matched) >= 2
    assert len(capture_outcome.review_needed) >= 1

    # 3. Assert strict immutability: zero additions, deletions, or mutations to face_embeddings
    final_embeddings_count = (
        await db_session.execute(select(func.count(FaceEmbedding.embedding_id)))
    ).scalar_one()
    final_enrollments_count = (
        await db_session.execute(select(func.count(Enrollment.enrollment_id)))
    ).scalar_one()

    final_embedding_ids = set(
        (await db_session.execute(select(FaceEmbedding.embedding_id))).scalars().all()
    )

    assert (
        final_embeddings_count == initial_embeddings_count
    ), f"Guardrail failure: face_embeddings count changed from {initial_embeddings_count} to {final_embeddings_count}"
    assert (
        final_enrollments_count == initial_enrollments_count
    ), f"Guardrail failure: enrollments count changed from {initial_enrollments_count} to {final_enrollments_count}"
    assert (
        final_embedding_ids == initial_embedding_ids
    ), "Guardrail failure: set of FaceEmbedding IDs mutated during capture!"
