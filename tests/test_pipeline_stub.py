"""
tests/test_pipeline_stub.py
────────────────────────────
Phase 0 smoke tests for the ml.pipeline module stubs.
Confirms the public API exists, accepts correct types, and returns correct
shapes even before Phase 2 installs real InsightFace weights.
"""

from __future__ import annotations

import numpy as np

from ml.pipeline import (
    CandidateDict,
    FaceEmbedding,
    MatchResult,
    align_and_embed,
    batch_process,
    detect_faces,
    match_embedding,
)


def make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_unit_vector(dim: int = 512) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


class TestDetectFacesStub:
    def test_returns_list(self) -> None:
        result = detect_faces(make_dummy_image())
        assert isinstance(result, list)

    def test_empty_on_blank_image(self) -> None:
        # The stub always returns [] — the real implementation may not
        result = detect_faces(make_dummy_image())
        assert result == []


class TestAlignAndEmbedStub:
    def test_returns_512_floats(self) -> None:
        dummy_landmarks = [(0.0, 0.0)] * 5
        emb = align_and_embed(make_dummy_image(), dummy_landmarks)
        assert isinstance(emb, list)
        assert len(emb) == 512
        assert all(isinstance(v, float) for v in emb)


class TestBatchProcessStub:
    def test_returns_list_of_face_embeddings(self) -> None:
        result = batch_process(make_dummy_image())
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, FaceEmbedding)
            assert len(item.embedding) == 512


class TestMatchEmbedding:
    def test_returns_none_on_empty_candidates(self) -> None:
        result = match_embedding(make_unit_vector(), candidates=[], threshold=0.45)
        assert result is None

    def test_returns_none_when_all_above_threshold(self) -> None:
        # Opposite vector will have distance ≈ 2.0
        query = [1.0] + [0.0] * 511
        candidates: list[CandidateDict] = [
            {
                "student_id": "s1",
                "embedding_id": "e1",
                "embedding": [-1.0] + [0.0] * 511,
            }
        ]
        result = match_embedding(query, candidates, threshold=0.45)
        assert result is None

    def test_returns_match_result_for_identical_vector(self) -> None:
        v = make_unit_vector()
        candidates: list[CandidateDict] = [
            {"student_id": "s1", "embedding_id": "e1", "embedding": v}
        ]
        result = match_embedding(v, candidates, threshold=0.45)
        assert result is not None
        assert isinstance(result, MatchResult)
        assert result.student_id == "s1"
        assert result.distance < 1e-4  # cosine dist to itself ≈ 0

    def test_returns_closest_match(self) -> None:
        # Two candidates — one much closer
        query = make_unit_vector()
        close = query[:]  # distance ≈ 0
        far_v = np.random.randn(512).astype(np.float32)
        far_v /= np.linalg.norm(far_v)
        candidates: list[CandidateDict] = [
            {"student_id": "close_student", "embedding_id": "e1", "embedding": close},
            {
                "student_id": "far_student",
                "embedding_id": "e2",
                "embedding": far_v.tolist(),
            },
        ]
        result = match_embedding(query, candidates, threshold=0.45)
        assert result is not None
        assert result.student_id == "close_student"
