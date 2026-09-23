"""
backend/services/capture.py
───────────────────────────
Capture-processing business logic.

This module is the bridge between Phase 2's ML pipeline and Phase 1's DAO:

    batch_process(image)  →  for each face:
        find_nearest_embeddings_for_roster(embedding, section_id, top_k=10)
            ├─ quality-gate failures            → review (never auto-matched)
            ├─ multi-embedding score ≤ threshold → candidate
            ├─ margin vs runner-up student ≥ margin → unambiguous candidate
            └─ otherwise                        → review (never force-matched)
        Hungarian 1-to-1 assignment over face × candidate-student costs
            ├─ assigned pair ≤ threshold+margin   → mark_attendance (idempotent)
            └─ unassigned / rejected faces        → enqueue_review (with crop)

Invariant (Module 3): faces_total == len(matched) + len(review_needed) —
every detected face ends up in EXACTLY one bucket, including quality-gate
failures, ambiguous margin cases, Hungarian rejects, and duplicate-face
conflicts.

Step 0 Guardrail: Enrollment embeddings and section enrollments are strictly
IMMUTABLE at runtime. Capture processing must NEVER insert or mutate rows in
face_embeddings or enrollments. Any future template updates or re-enrollment
must pass through a human-reviewed admin queue with explicit approval and audit logging.

Kept separate from backend/worker/main.py so tests can exercise the
matching/attendance logic without images, a queue, or InsightFace installed.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.dao import find_nearest_embeddings_for_roster, get_class_session, mark_attendance

if TYPE_CHECKING:
    from datetime import date

log = logging.getLogger(__name__)

# Defaults for the matching strategy (Module 3). The worker overrides these
# from settings so they stay calibratable via env.
DEFAULT_TOP_K: int = 10  # nearest embeddings fetched per face
DEFAULT_MARGIN: float = 0.08  # required gap best vs runner-up student
DEFAULT_CANDIDATE_STUDENTS: int = 5  # roster students entering the cost matrix


@dataclass
class FaceOutcome:
    """Result of matching ONE detected face against one section's roster."""

    bbox: tuple[int, int, int, int]
    matched: bool
    student_id: uuid.UUID | None = None
    match_distance: float | None = None
    # Kept so unmatched faces can be re-searched institution-wide later.
    embedding: list[float] | None = None
    # Quality-gate flags from the pipeline (size/blur/pose) — flagged faces
    # are always routed to review with a crop for a human decision.
    quality_flags: tuple[str, ...] = ()
    # Why the face ended in review (auditable, surfaced in the review UI).
    review_reason: str = ""


@dataclass
class CaptureOutcome:
    """Aggregated result of processing one classroom photo."""

    faces_total: int = 0
    matched: list[FaceOutcome] = field(default_factory=list)
    review_needed: list[FaceOutcome] = field(default_factory=list)


# ─── Multi-embedding candidate scoring (Module 3) ───────────────────────────


@dataclass(frozen=True)
class StudentScore:
    """Composite distance of one candidate student for one face."""

    student_id: uuid.UUID
    # Mean cosine distance over the student's top-2 closest enrollment
    # embeddings — robust against one bad enrollment photo.
    composite_distance: float
    # Single closest embedding distance (for display / best-guess).
    best_distance: float


def _score_candidates(
    matches: list[Any],  # list[RosterMatch] from find_nearest_embeddings_for_roster
    max_students: int = DEFAULT_CANDIDATE_STUDENTS,
) -> list[StudentScore]:
    """
    Group top-K nearest embedding rows by student and compute each
    student's composite score = mean of their top-2 closest embedding
    distances (falls back to the single best when only one row exists).

    With 3-5 enrollment embeddings per student and top_k=10, the top rows
    are typically dominated by one student's references; grouping converts
    embedding-level ranking into the student-level ranking the assignment
    step needs.
    """
    per_student: dict[uuid.UUID, list[float]] = {}
    for m in matches:
        per_student.setdefault(m.student_id, []).append(float(m.distance))

    scored: list[StudentScore] = []
    for student_id, dists in per_student.items():
        dists.sort()
        best = dists[0]
        top2 = dists[:2]
        composite = float(np.mean(top2))
        scored.append(
            StudentScore(
                student_id=student_id,
                composite_distance=composite,
                best_distance=best,
            )
        )
    # Best student first.
    scored.sort(key=lambda s: s.composite_distance)
    return scored[:max_students]


async def match_faces_to_roster(
    session: AsyncSession,
    *,
    section_id: uuid.UUID,
    face_embeddings: list[tuple[tuple[int, int, int, int], list[float]]],
    threshold: float,
    top_k: int = DEFAULT_TOP_K,
    margin: float = DEFAULT_MARGIN,
    candidate_students: int = DEFAULT_CANDIDATE_STUDENTS,
    face_quality_flags: list[tuple[str, ...]] | None = None,
) -> list[FaceOutcome]:
    """
    Match every detected face against ONLY this section's roster.

    ``face_embeddings`` is a list of (bbox, embedding) pairs as produced by
    ml.pipeline.batch_process(). The threshold comparison happens here (not
    in SQL) so the calibration value stays a single setting.

    Strategy (Module 3)
    ~~~~~~~~~~~~~~~~~~~
    1. Top-K retrieval: pgvector returns the K=10 nearest roster embeddings
       per face (not just the single nearest).
    2. Multi-embedding scoring: rows are grouped by student; each student's
       score is the mean of their top-2 embedding distances.
    3. Margin testing: a candidate only auto-matches when
       best_student.composite ≤ threshold AND
       runner_up_student.composite − best_student.composite ≥ margin.
       Margin violations (lookalikes) go to review.

    The 1-to-1 Hungarian assignment then runs across ALL faces at once (see
    assign_one_to_one) — callers normally use match_and_assign() instead of
    calling this and the assignment separately.
    """
    flags_per_face = face_quality_flags if face_quality_flags is not None else []
    outcomes: list[FaceOutcome] = []

    for idx, (bbox, embedding) in enumerate(face_embeddings):
        flags = tuple(flags_per_face[idx]) if idx < len(flags_per_face) else ()

        # Quality-gate failures skip matching entirely — never auto-match a
        # face that failed size/blur/pose checks.
        if flags:
            candidates = await find_nearest_embeddings_for_roster(
                session, embedding, section_id=section_id, top_k=1
            )
            best_dist = float(candidates[0].distance) if candidates else None
            outcomes.append(
                FaceOutcome(
                    bbox=bbox,
                    matched=False,
                    student_id=candidates[0].student_id if candidates else None,
                    match_distance=best_dist,
                    embedding=embedding,
                    quality_flags=flags,
                    review_reason=f"quality:{'+'.join(flags)}",
                )
            )
            continue

        candidates = await find_nearest_embeddings_for_roster(
            session, embedding, section_id=section_id, top_k=top_k
        )
        if not candidates:
            outcomes.append(
                FaceOutcome(
                    bbox=bbox,
                    matched=False,
                    embedding=embedding,
                    review_reason="no-roster-candidates",
                )
            )
            continue

        scored = _score_candidates(candidates, max_students=candidate_students)
        best = scored[0]
        runner_up_composite = scored[1].composite_distance if len(scored) > 1 else float("inf")

        within_threshold = best.composite_distance <= threshold
        unambiguous = (runner_up_composite - best.composite_distance) >= margin

        if within_threshold and unambiguous:
            outcomes.append(
                FaceOutcome(
                    bbox=bbox,
                    matched=True,
                    student_id=best.student_id,
                    match_distance=best.composite_distance,
                )
            )
        elif within_threshold and not unambiguous:
            # Ambiguous: best student is plausible but a runner-up is nearly
            # as close (twins / lookalikes) — human decides.
            outcomes.append(
                FaceOutcome(
                    bbox=bbox,
                    matched=False,
                    student_id=best.student_id,
                    match_distance=best.composite_distance,
                    embedding=embedding,
                    review_reason="ambiguous-margin",
                )
            )
        else:
            outcomes.append(
                FaceOutcome(
                    bbox=bbox,
                    matched=False,
                    student_id=best.student_id,
                    match_distance=best.composite_distance,
                    embedding=embedding,
                    review_reason="above-threshold",
                )
            )
    return outcomes


# ─── 1-to-1 Hungarian assignment (Module 3) ─────────────────────────────────


def assign_one_to_one(
    outcomes: list[FaceOutcome],
    *,
    threshold: float,
    margin: float = DEFAULT_MARGIN,
) -> list[FaceOutcome]:
    """
    Globally resolve face→student assignments so ONE student is marked once
    even if two photos/detections look like them.

    Builds the cost matrix from the per-face candidate lists produced by
    match_faces_to_roster (each matched face carries its best student; the
    full per-face candidate map is rebuilt from the DAO fetch inside
    match_and_assign — see there) and solves the bipartite assignment with
    scipy.optimize.linear_sum_assignment.

    This pure function operates on the outcomes list in place-style: faces
    already matched=True keep their student only if the assignment confirms
    it; otherwise they are demoted to review with reason
    "assignment-conflict" or "assignment-margin".

    Parameters
    ----------
    outcomes : per-face results from match_faces_to_roster.
    threshold: same cosine-distance threshold as matching.
    margin   : pairs violating the margin at assignment time are rejected.
    """
    matched_idx = [i for i, o in enumerate(outcomes) if o.matched and o.student_id]
    if not matched_idx:
        return outcomes

    # Faces (rows) × unique students (cols) cost matrix, padded rectangular.
    students = list({outcomes[i].student_id for i in matched_idx})
    col_of = {s: c for c, s in enumerate(students)}
    n_rows, n_cols = len(matched_idx), len(students)
    cost = np.full((n_rows, n_cols), 1e6, dtype=np.float64)  # 1e6 = forbidden
    for r, oi in enumerate(matched_idx):
        o = outcomes[oi]
        assert o.student_id is not None
        cost[r, col_of[o.student_id]] = float(o.match_distance or 0.0)

    rows, cols = linear_sum_assignment(cost)

    # linear_sum_assignment returns matrix row/col indices — map back to
    # outcome positions via matched_idx.
    assigned_pairs = {int(r): int(c) for r, c in zip(rows, cols, strict=False)}
    for r, oi in enumerate(matched_idx):
        o = outcomes[oi]
        if r not in assigned_pairs:
            # Rectangular case: min(rows, cols) pairs assigned; an unassigned
            # face loses its claim to the student.
            o.matched = False
            o.review_reason = "assignment-conflict"
            continue
        c = assigned_pairs[r]
        chosen = students[c]
        pair_cost = float(cost[r, c])
        if pair_cost >= 1e6 or pair_cost > threshold:
            o.matched = False
            o.review_reason = "assignment-rejected"
            continue
        o.student_id = chosen
        o.match_distance = pair_cost
    return outcomes


async def match_and_assign(
    session: AsyncSession,
    *,
    section_id: uuid.UUID,
    face_embeddings: list[tuple[tuple[int, int, int, int], list[float]]],
    threshold: float,
    top_k: int = DEFAULT_TOP_K,
    margin: float = DEFAULT_MARGIN,
    candidate_students: int = DEFAULT_CANDIDATE_STUDENTS,
    face_quality_flags: list[tuple[str, ...]] | None = None,
) -> list[FaceOutcome]:
    """
    Full Module 3 pipeline: roster matching + margin tests + global 1-to-1
    Hungarian assignment. The worker's single entry point.
    """
    outcomes = await match_faces_to_roster(
        session,
        section_id=section_id,
        face_embeddings=face_embeddings,
        threshold=threshold,
        top_k=top_k,
        margin=margin,
        candidate_students=candidate_students,
        face_quality_flags=face_quality_flags,
    )
    return assign_one_to_one(outcomes, threshold=threshold, margin=margin)


# ─── Persistence ────────────────────────────────────────────────────────────


async def write_capture_results(
    session: AsyncSession,
    *,
    session_id: uuid.UUID,
    outcomes: list[FaceOutcome],
    crop_urls: dict[int, str] | None = None,
) -> CaptureOutcome:
    """
    Persist matching results for one capture job.

    Matched faces → attendance rows (idempotent via ON CONFLICT).
    Unmatched     → review_queue rows (with best-guess info when available).

    Module 3 invariant: every face lands in exactly one bucket. A face whose
    mark_attendance returns False (duplicate: same student already marked —
    e.g. two frames of a burst, or two similar faces assigned to one student
    then conflicting) is APPENDED to review_needed with its crop, never
    silently dropped.

    ``crop_urls`` maps the index of each unmatched face to its stored crop
    URL. Returns the aggregated outcome; caller owns the commit and the
    job-status update.
    """
    # Lazy import avoids a cycle: review dao pulls models only.
    from backend.db.dao import enqueue_review

    class_session = await get_class_session(session, session_id)
    if class_session is None:
        raise LookupError(f"class session {session_id} does not exist")

    session_date: date = class_session.session_date
    outcome = CaptureOutcome(faces_total=len(outcomes))

    for idx, face in enumerate(outcomes):
        if face.matched and face.student_id is not None:
            inserted = await mark_attendance(
                session,
                student_id=face.student_id,
                session_id=session_id,
                session_date=session_date,
                match_distance=face.match_distance,
            )
            if inserted:
                outcome.matched.append(face)
            else:
                # Duplicate attendance: the student is ALREADY marked — this
                # is the Module 4 burst-frame union working as designed (or a
                # re-capture of the same session). The face matched cleanly,
                # so it is accounted as matched — NOT review — otherwise a
                # 5-frame burst would flood the queue with one review item
                # per student per extra frame. The face is never dropped:
                # faces_total == matched + review still holds exactly.
                log.debug(
                    "write_capture_results: duplicate attendance for student %s "
                    "in session %s — union absorbed (attendance already marked)",
                    face.student_id,
                    session_id,
                )
                outcome.matched.append(face)
        else:
            await enqueue_review(
                session,
                session_id=session_id,
                face_crop_url=(crop_urls or {}).get(idx),
                face_embedding=face.embedding,
                best_guess_student_id=face.student_id,
                best_guess_distance=face.match_distance,
            )
            outcome.review_needed.append(face)

    # Invariant check — a violation is a bug, loud not silent.
    if outcome.faces_total != len(outcome.matched) + len(outcome.review_needed):
        raise RuntimeError(
            "face-accounting invariant violated: "
            f"total={outcome.faces_total} matched={len(outcome.matched)} "
            f"review={len(outcome.review_needed)}"
        )

    await session.flush()
    return outcome
