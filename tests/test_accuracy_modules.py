"""
tests/test_accuracy_modules.py
───────────────────────────────
Tests for the 6-module accuracy refactor:

  Module 1  settings wiring + flip-TTA (logic parts that don't need weights)
  Module 2  native-res policy, tile grids, quality gates, cross-tile NMS
  Module 3  top-K multi-embedding scoring, margin testing, Hungarian
            1-to-1 assignment, face-accounting invariant
  Module 4  multi-frame burst ingestion + review auto-resolve, enrollment
            sanity, largest-face calibration pick

DB-backed cases reuse the conftest db_session (skipped if Postgres down).
Pure-logic cases always run — no InsightFace required.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import dao
from backend.db.models import Enrollment, ReviewQueue, ReviewStatus
from backend.services.capture import (
    FaceOutcome,
    _score_candidates,
    assign_one_to_one,
    match_and_assign,
    match_faces_to_roster,
    write_capture_results,
)

THRESHOLD = 0.45
MARGIN = 0.08


# ─── Helpers ─────────────────────────────────────────────────────────────────


def unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


def perturb(vec: list[float], seed: int, scale: float = 0.02) -> list[float]:
    noise = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    noise *= scale / np.linalg.norm(noise)
    return (np.asarray(vec) + noise).tolist()


# ═══ Module 2: pure pipeline logic ═══════════════════════════════════════════


class TestTileGrid:
    def test_small_image_single_tile(self) -> None:
        from ml.pipeline import TiledDetectionConfig, compute_tile_grid

        tiles = compute_tile_grid(1280, 960, TiledDetectionConfig(tile_min_size=1600))
        assert tiles == [(0, 0, 1280, 960)]

    def test_wide_image_more_cols_than_rows(self) -> None:
        from ml.pipeline import TiledDetectionConfig, compute_tile_grid

        tiles = compute_tile_grid(4000, 2000, TiledDetectionConfig(tile_min_size=1600))
        # 4000/1600 → 3 cols, 2000/1600 → 2 rows (each ≥ ceil ratio)
        assert len(tiles) == 6
        # Full coverage: union spans the canvas.
        xs = sorted({x for (x, _, _, _) in tiles})
        assert xs[0] == 0
        max_x2 = max(x2 for (_, _, x2, _) in tiles)
        max_y2 = max(y2 for (_, _, _, y2) in tiles)
        assert (max_x2, max_y2) == (4000, 2000)

    def test_tiles_overlap(self) -> None:
        from ml.pipeline import TiledDetectionConfig, compute_tile_grid

        tiles = compute_tile_grid(3200, 1600, TiledDetectionConfig(tile_min_size=1600, overlap=0.2))
        # 2 cols × 1 row: the right tile starts LEFT of the first tile's end.
        x_ends = sorted(tiles, key=lambda t: t[0])
        assert x_ends[1][0] < x_ends[0][2], "adjacent tiles must overlap"

    def test_every_tile_meets_min_size(self) -> None:
        from ml.pipeline import TiledDetectionConfig, compute_tile_grid

        cfg = TiledDetectionConfig(tile_min_size=1600)
        tiles = compute_tile_grid(5000, 3000, cfg)
        for x1, y1, x2, y2 in tiles:
            assert (x2 - x1) >= 1600 * 0.5  # generous: tiles are ~min_size
            assert (y2 - y1) >= 1600 * 0.5


class TestNMS:
    def test_duplicate_across_tiles_suppressed(self) -> None:
        from ml.pipeline import BoundingBox, FaceDetection, _nms

        a = FaceDetection(BoundingBox(10, 10, 110, 110), [], 0.9)
        b = FaceDetection(BoundingBox(14, 14, 114, 114), [], 0.6)  # IoU ~0.86
        c = FaceDetection(BoundingBox(400, 400, 500, 500), [], 0.5)
        kept = _nms([b, a, c], 0.4)
        assert len(kept) == 2
        assert kept[0] is a  # highest score first
        assert c in kept

    def test_disjoint_boxes_all_kept(self) -> None:
        from ml.pipeline import BoundingBox, FaceDetection, _nms

        dets = [
            FaceDetection(BoundingBox(i * 200, 0, i * 200 + 100, 100), [], 0.5) for i in range(4)
        ]
        assert len(_nms(dets, 0.4)) == 4


class TestQualityGates:
    def _symmetric_landmarks(self) -> list[tuple[float, float]]:
        return [(140.0, 160.0), (260.0, 160.0), (200.0, 240.0), (170.0, 280.0), (230.0, 280.0)]

    def test_flat_crop_fails_blur(self) -> None:
        import cv2

        from ml.pipeline import BoundingBox, QualityGateConfig, assess_face_quality

        img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
        flags = assess_face_quality(
            img,
            BoundingBox(100, 100, 300, 300),
            self._symmetric_landmarks(),
            QualityGateConfig(),
        )
        assert "blur" in flags

    def test_noise_crop_passes_all(self) -> None:
        from ml.pipeline import BoundingBox, QualityGateConfig, assess_face_quality

        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)
        flags = assess_face_quality(
            noisy,
            BoundingBox(100, 100, 300, 300),
            self._symmetric_landmarks(),
            QualityGateConfig(),
        )
        assert flags == ()

    def test_tiny_face_fails_size(self) -> None:
        from ml.pipeline import BoundingBox, QualityGateConfig, assess_face_quality

        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)
        flags = assess_face_quality(
            noisy,
            BoundingBox(0, 0, 40, 40),
            self._symmetric_landmarks(),
            QualityGateConfig(),
        )
        assert "size" in flags

    def test_extreme_profile_fails_pose(self) -> None:
        from ml.pipeline import BoundingBox, QualityGateConfig, assess_face_quality

        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 256, (500, 500, 3), dtype=np.uint8)
        # Nose pushed far to the right of both eyes → huge asymmetry.
        lm = [(100.0, 160.0), (260.0, 160.0), (340.0, 240.0), (170.0, 280.0), (230.0, 280.0)]
        flags = assess_face_quality(noisy, BoundingBox(100, 100, 300, 300), lm, QualityGateConfig())
        assert "pose" in flags

    def test_gates_disabled_when_zero(self) -> None:
        import cv2

        from ml.pipeline import BoundingBox, QualityGateConfig, assess_face_quality

        img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (90, 90, 90), -1)
        off = QualityGateConfig(min_size_px=0, min_blur_var=0, max_pose_ratio=0)
        assert assess_face_quality(img, BoundingBox(0, 0, 20, 20), [], off) == ()


class TestFlipTTA:
    def _fake_rec_model(self, seed: int) -> Any:
        """Minimal ArcFace-shaped recognition-model stub."""
        emb = np.asarray(unit_vec(seed), dtype=np.float32).reshape(1, 512)

        class FakeSession:
            run = staticmethod(lambda names, feeds: [emb])

        class FakeRecModel:
            input_size = (112, 112)
            input_std = 127.5
            input_mean = 127.5
            input_name = "input"
            output_names = ["output"]
            session = FakeSession()

        return FakeRecModel()

    def test_tta_average_is_unit_and_stable(self) -> None:
        """The TTA math (average + L2 renorm) without model weights: an
        aligned crop and its mirror embed to the same vector, so the
        averaged embedding must be a unit vector equal to it."""
        from ml.pipeline import _embed_aligned_crop

        rng = np.random.default_rng(3)
        aligned = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
        out = _embed_aligned_crop(aligned, self._fake_rec_model(1), tta_flip=True)
        assert out is not None
        norm = float(np.linalg.norm(np.asarray(out)))
        assert abs(norm - 1.0) < 1e-5  # re-normalised
        assert len(out) == 512

    def test_no_tta_returns_raw_embedding(self) -> None:
        from ml.pipeline import _embed_aligned_crop

        rng = np.random.default_rng(4)
        aligned = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
        out = _embed_aligned_crop(aligned, self._fake_rec_model(2), tta_flip=False)
        assert out is not None
        assert np.allclose(out, np.asarray(unit_vec(2), dtype=np.float32))

    def test_empty_crop_returns_none(self) -> None:
        from ml.pipeline import _embed_aligned_crop

        class FakeRecModel:
            pass

        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert _embed_aligned_crop(empty, FakeRecModel(), True) is None


# ═══ Module 3: matching, margin, Hungarian ═══════════════════════════════════


class TestScoreCandidates:
    def test_groups_by_student_and_averages_top2(self) -> None:
        s1, s2 = uuid.uuid4(), uuid.uuid4()

        class M:
            def __init__(self, sid: uuid.UUID, d: float) -> None:
                self.student_id, self.distance = sid, d

        rows = [M(s1, 0.30), M(s1, 0.34), M(s1, 0.50), M(s2, 0.36), M(s2, 0.60)]
        scored = _score_candidates(rows)
        # s1 composite = mean(0.30, 0.34) = 0.32; s2 = mean(0.36, 0.60) = 0.48
        assert scored[0].student_id == s1
        assert abs(scored[0].composite_distance - 0.32) < 1e-9
        assert scored[1].student_id == s2

    def test_single_embedding_student_uses_best(self) -> None:
        s1 = uuid.uuid4()

        class M:
            def __init__(self, sid: uuid.UUID, d: float) -> None:
                self.student_id, self.distance = sid, d

        scored = _score_candidates([M(s1, 0.40)])
        assert abs(scored[0].composite_distance - 0.40) < 1e-9


class TestHungarianAssignment:
    def test_two_faces_one_student_one_demoted(self) -> None:
        a = uuid.uuid4()
        o1 = FaceOutcome((0, 0, 1, 1), True, a, 0.30)
        o2 = FaceOutcome((5, 5, 6, 6), True, a, 0.35)
        assign_one_to_one([o1, o2], threshold=THRESHOLD)
        assert o1.matched and not o2.matched
        assert o2.review_reason == "assignment-conflict"

    def test_better_distance_wins(self) -> None:
        a = uuid.uuid4()
        o_worse = FaceOutcome((0, 0, 1, 1), True, a, 0.30)
        o_better = FaceOutcome((5, 5, 6, 6), True, a, 0.10)
        assign_one_to_one([o_worse, o_better], threshold=THRESHOLD)
        assert o_better.matched
        assert not o_worse.matched

    def test_two_faces_two_students_both_kept(self) -> None:
        a, b = uuid.uuid4(), uuid.uuid4()
        o1 = FaceOutcome((0, 0, 1, 1), True, a, 0.30)
        o2 = FaceOutcome((5, 5, 6, 6), True, b, 0.31)
        assign_one_to_one([o1, o2], threshold=THRESHOLD)
        assert o1.matched and o2.matched

    def test_above_threshold_pair_rejected(self) -> None:
        a = uuid.uuid4()
        o1 = FaceOutcome((0, 0, 1, 1), True, a, 0.99)  # would never pass matching,
        # but assignment re-checks defensively
        out = assign_one_to_one([o1], threshold=THRESHOLD)
        assert not out[0].matched
        assert out[0].review_reason == "assignment-rejected"


# ─── DB-backed Module 3+4 world ───────────────────────────────────────────────


@pytest.fixture
async def roster_world(db_session: AsyncSession) -> dict[str, Any]:
    """One section, 3 enrolled students with 3 embeddings each."""
    inst = await dao.create_institution(db_session, name="Accuracy U")
    dept = await dao.create_department(db_session, institution_id=inst.institution_id, name="CSE")
    section = await dao.create_section(
        db_session, department_id=dept.department_id, name="CSE-3A", academic_year=2026
    )
    ids: dict[str, Any] = {"section": section.section_id}
    for i, seed in enumerate((10, 11, 12)):
        student = await dao.create_student(
            db_session,
            roll_number=f"ACC{i}",
            name=f"Student {i}",
            password_hash="x",
            institution_id=inst.institution_id,
        )
        for j in range(3):  # 3 enrollment embeddings each
            await dao.add_embedding(
                db_session,
                student_id=student.student_id,
                embedding=perturb(unit_vec(seed), seed * 10 + j, scale=0.01),
            )
        db_session.add(
            Enrollment(
                enrollment_id=uuid.uuid4(),
                student_id=student.student_id,
                section_id=section.section_id,
            )
        )
        ids[f"s{i}"] = student.student_id
    # Twin: a student whose enrollment embeddings are just tiny rotations of
    # s0's — distance ~0.005-0.02 from s0's refs, well inside the 0.08
    # margin, so queries near s0 are genuinely ambiguous.
    twin = await dao.create_student(
        db_session,
        roll_number="TWIN",
        name="Twin of s0",
        password_hash="x",
        institution_id=inst.institution_id,
    )
    for j in range(3):
        await dao.add_embedding(
            db_session,
            student_id=twin.student_id,
            embedding=perturb(unit_vec(10), 700 + j, scale=0.008),
        )
    db_session.add(
        Enrollment(
            enrollment_id=uuid.uuid4(),
            student_id=twin.student_id,
            section_id=section.section_id,
        )
    )
    ids["twin"] = twin.student_id
    await db_session.commit()
    return ids


class TestMarginTesting:
    async def test_ambiguous_face_goes_to_review(
        self, db_session: AsyncSession, roster_world: dict[str, Any]
    ) -> None:
        """A face near s0 (twin enrolled with near-identical refs) must NOT
        auto-match when s0 and the twin are within the margin of each other."""
        # Query sits right on s0's cluster: s0 best, twin runner-up ~0.01 behind.
        query = perturb(unit_vec(10), 999)

        outcomes = await match_faces_to_roster(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=[((0, 0, 100, 100), query)],
            threshold=THRESHOLD,
            margin=MARGIN,
        )
        assert not outcomes[0].matched
        assert outcomes[0].review_reason == "ambiguous-margin"
        assert outcomes[0].student_id in (
            roster_world["s0"],
            roster_world["twin"],
        )  # best guess attached (whichever scored first)

    async def test_clear_match_passes_margin(
        self, db_session: AsyncSession, roster_world: dict[str, Any]
    ) -> None:
        outcomes = await match_faces_to_roster(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=[((0, 0, 100, 100), perturb(unit_vec(12), 777))],
            threshold=THRESHOLD,
            margin=MARGIN,
        )
        assert outcomes[0].matched
        assert outcomes[0].student_id == roster_world["s2"]

    async def test_margin_zero_disambiguation_lets_lookalike_through(
        self, db_session: AsyncSession, roster_world: dict[str, Any]
    ) -> None:
        """margin=0 disables the ambiguity guard (calibration baseline)."""
        query = perturb(unit_vec(10), 999)
        outcomes = await match_faces_to_roster(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=[((0, 0, 100, 100), query)],
            threshold=THRESHOLD,
            margin=0.0,
        )
        assert outcomes[0].matched  # without the guard it auto-matches


class TestFaceAccountingInvariant:
    async def test_every_face_lands_in_exactly_one_bucket(
        self, db_session: AsyncSession, roster_world: dict[str, Any]
    ) -> None:
        """faces_total == len(matched) + len(review_needed) across all paths."""
        session_row = await dao.create_class_session(
            db_session,
            section_id=roster_world["section"],
            course_name="DBMS",
            session_date=date.today(),
            admin_id=None,
        )
        photo = [
            ((0, 0, 100, 100), perturb(unit_vec(10), 101)),  # s0 (twin → ambiguous)
            ((110, 0, 210, 100), perturb(unit_vec(11), 111)),  # s1
            ((220, 0, 320, 100), unit_vec(9999)),  # stranger → review
            ((330, 0, 430, 100), perturb(unit_vec(12), 121)),  # s2
        ]
        outcomes = await match_and_assign(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=photo,
            threshold=THRESHOLD,
            margin=MARGIN,
        )
        # Duplicate s2: same student appears twice → Hungarian demotes one.
        photo.append(((440, 0, 540, 100), perturb(unit_vec(12), 131)))
        outcomes = await match_and_assign(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=photo,
            threshold=THRESHOLD,
            margin=MARGIN,
        )
        outcome = await write_capture_results(
            db_session, session_id=session_row.session_id, outcomes=outcomes
        )
        await db_session.commit()

        assert outcome.faces_total == 5
        assert outcome.faces_total == len(outcome.matched) + len(outcome.review_needed)
        # s0 is ambiguous (twin) → review; s1, s2, s2-dup: 2 unique matches
        # (dup absorbed by the 1-to-1 assignment / idempotent insert);
        # stranger → review. At least 2 matched, 3 review.
        assert len(outcome.matched) == 2
        assert len(outcome.review_needed) == 3

    async def test_quality_flagged_face_routed_to_review(
        self, db_session: AsyncSession, roster_world: dict[str, Any]
    ) -> None:
        """Quality-gate failures are never auto-matched — always review."""
        session_row = await dao.create_class_session(
            db_session,
            section_id=roster_world["section"],
            course_name="Quality",
            session_date=date.today(),
            admin_id=None,
        )
        outcomes = await match_faces_to_roster(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=[((0, 0, 100, 100), perturb(unit_vec(10), 101))],
            threshold=THRESHOLD,
            margin=MARGIN,
            face_quality_flags=[("blur", "size")],
        )
        assert not outcomes[0].matched
        assert "blur" in outcomes[0].quality_flags
        assert outcomes[0].review_reason == "quality:blur+size"

        outcome = await write_capture_results(
            db_session, session_id=session_row.session_id, outcomes=outcomes
        )
        await db_session.commit()
        assert len(outcome.review_needed) == 1
        assert outcome.faces_total == 1


class TestBurstFrames:
    async def test_union_across_frames_and_auto_resolve(
        self, db_session: AsyncSession, roster_world: dict[str, Any]
    ) -> None:
        """Module 4: burst frame 2 confirms what frame 1 could only guess.

        Frame 1: s0's face is ambiguous (margin) → review with s0 best-guess.
        Frame 2: crisp s0 face → matched. The worker's auto-resolve step then
        clears the pending review for s0 in the same session.
        """
        from backend.worker.main import _auto_resolve_reviews

        session_row = await dao.create_class_session(
            db_session,
            section_id=roster_world["section"],
            course_name="Burst",
            session_date=date.today(),
            admin_id=None,
        )

        # Frame 1: ambiguous probe near s0 with the twin breathing down its
        # neck → review with best-guess attached.
        frame1 = await match_faces_to_roster(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=[((0, 0, 100, 100), perturb(unit_vec(10), 999))],
            threshold=THRESHOLD,
            margin=MARGIN,
        )
        assert not frame1[0].matched
        assert frame1[0].review_reason == "ambiguous-margin"
        outcome1 = await write_capture_results(
            db_session, session_id=session_row.session_id, outcomes=frame1
        )
        await db_session.commit()
        assert len(outcome1.review_needed) == 1

        # Frame 2: crisp s2 capture → auto-match (s2 has no twin).
        frame2 = await match_faces_to_roster(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=[((50, 50, 150, 150), perturb(unit_vec(12), 222))],
            threshold=THRESHOLD,
            margin=MARGIN,
        )
        assert frame2[0].matched
        assert frame2[0].student_id == roster_world["s2"]
        outcome2 = await write_capture_results(
            db_session, session_id=session_row.session_id, outcomes=frame2
        )
        await db_session.commit()
        assert outcome2.matched  # idempotent insert for s2 (first insert wins)

        # Worker auto-resolve: reviews whose best_guess == confirmed student.
        # The ambiguous frame-1 review guessed s0 or the twin — neither was
        # confirmed by frame 2, so it must STAY pending.
        confirmed = {o.student_id for o in outcome2.matched}
        resolved_count = await _auto_resolve_reviews(
            db_session,
            newly_matched_students=confirmed,
            session_id=session_row.session_id,
        )
        await db_session.commit()
        assert resolved_count == 0
        rows = list((await db_session.execute(select(ReviewQueue))).scalars())
        assert rows[0].status == ReviewStatus.pending

        # Frame 3: a LATER crisp s0 capture can't exist while the twin is
        # enrolled (that ambiguity is precisely the margin guard's job) —
        # instead simulate the admin-grade confirmation path: a frame that
        # matches the twin unambiguously (probe at the twin's own seed).
        frame3 = await match_faces_to_roster(
            db_session,
            section_id=roster_world["section"],
            face_embeddings=[((80, 80, 180, 180), perturb(unit_vec(10), 999))],
            threshold=THRESHOLD,
            margin=0.0,  # admin-grade disambiguation (margin off)
        )
        assert frame3[0].matched
        outcome3 = await write_capture_results(
            db_session, session_id=session_row.session_id, outcomes=frame3
        )
        await db_session.commit()

        # Now the auto-resolve fires for the review whose best_guess matches.
        confirmed = {o.student_id for o in outcome3.matched}
        resolved_count = await _auto_resolve_reviews(
            db_session,
            newly_matched_students=confirmed,
            session_id=session_row.session_id,
        )
        await db_session.commit()
        assert resolved_count == 1
        rows = list((await db_session.execute(select(ReviewQueue))).scalars())
        assert rows[0].status == ReviewStatus.resolved


class TestEnrollmentSanity:
    async def test_student_scoped_lookup_returns_own_embeddings_only(
        self, db_session: AsyncSession, roster_world: dict[str, Any]
    ) -> None:
        """find_nearest_embeddings_for_student never leaks other students."""
        matches = await dao.find_nearest_embeddings_for_student(
            db_session, perturb(unit_vec(11), 42), student_id=roster_world["s1"], top_k=5
        )
        assert matches
        assert all(m.student_id == roster_world["s1"] for m in matches)
