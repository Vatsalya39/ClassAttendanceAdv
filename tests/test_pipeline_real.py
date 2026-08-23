"""
tests/test_pipeline_real.py
────────────────────────────
Phase 2 tests for the ml.pipeline module with real InsightFace calls.

These tests skip gracefully when InsightFace is not installed (CI-safe).
On a machine with InsightFace installed, the first run will download the
buffalo_l model pack (~600 MB) to ~/.insightface/models/buffalo_l/.

Run with:
    poetry run pytest tests/test_pipeline_real.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Skip the entire module if insightface is not installed
insightface = pytest.importorskip(
    "insightface",
    reason="insightface not installed — skipping real pipeline tests (install ml/requirements.txt)",
)

from ml.pipeline import (  # noqa: E402 (after importorskip)
    MIN_DETECTION_SCORE,
    FaceEmbedding,
    _get_app,
    align_and_embed,
    batch_process,
    detect_faces,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_blank_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Solid black image — no faces should be detected."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_noise_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Random noise image — no real faces, but valid dtype/shape."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── Model loading ───────────────────────────────────────────────────────────


class TestGetApp:
    def test_returns_non_none(self) -> None:
        app = _get_app()
        assert app is not None, "InsightFace model failed to load"

    def test_singleton_same_object(self) -> None:
        app1 = _get_app()
        app2 = _get_app()
        assert app1 is app2, "_get_app() must return the same cached object"


# ─── Image validation ─────────────────────────────────────────────────────────


class TestImageValidation:
    def test_detect_faces_wrong_dtype_raises(self) -> None:
        bad = np.zeros((480, 640, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            detect_faces(bad)

    def test_detect_faces_wrong_channels_raises(self) -> None:
        bad = np.zeros((480, 640), dtype=np.uint8)  # grayscale, ndim=2
        with pytest.raises(ValueError):
            detect_faces(bad)

    def test_detect_faces_too_small_raises(self) -> None:
        bad = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            detect_faces(bad)

    def test_batch_process_wrong_dtype_raises(self) -> None:
        bad = np.zeros((480, 640, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            batch_process(bad)

    def test_align_and_embed_wrong_landmarks_count_raises(self) -> None:
        image = make_blank_image()
        with pytest.raises(ValueError, match="5"):
            align_and_embed(image, [(0.0, 0.0)] * 3)


# ─── detect_faces ─────────────────────────────────────────────────────────────


class TestDetectFacesReal:
    def test_blank_image_returns_empty_list(self) -> None:
        result = detect_faces(make_blank_image())
        assert isinstance(result, list)
        assert result == [], "No faces expected in a blank image"

    def test_noise_image_returns_list(self) -> None:
        result = detect_faces(make_noise_image())
        assert isinstance(result, list)
        # Noise might occasionally trigger false detections — just check type
        for det in result:
            assert det.detection_score >= MIN_DETECTION_SCORE


# ─── batch_process ────────────────────────────────────────────────────────────


class TestBatchProcessReal:
    def test_blank_image_returns_empty(self) -> None:
        result = batch_process(make_blank_image())
        assert result == []

    def test_returns_list_of_face_embeddings(self) -> None:
        result = batch_process(make_noise_image())
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, FaceEmbedding)

    def test_embeddings_are_unit_normalised(self) -> None:
        """All embeddings from a real image must be unit vectors (norm ≈ 1.0)."""
        result = batch_process(make_noise_image())
        for face in result:
            emb = np.array(face.embedding, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            assert math.isclose(
                norm, 1.0, abs_tol=1e-4
            ), f"Embedding norm {norm:.6f} is not ≈ 1.0 — ArcFace normed_embedding contract violated"

    def test_embedding_dimension_is_512(self) -> None:
        result = batch_process(make_noise_image())
        for face in result:
            assert len(face.embedding) == 512


# ─── align_and_embed (no-face case) ──────────────────────────────────────────


class TestAlignAndEmbedReal:
    def test_blank_image_returns_zero_vector(self) -> None:
        # No face detected → zero vector fallback
        dummy_landmarks = [
            (100.0, 100.0),
            (200.0, 100.0),
            (150.0, 150.0),
            (120.0, 200.0),
            (180.0, 200.0),
        ]
        result = align_and_embed(make_blank_image(), dummy_landmarks)
        assert isinstance(result, list)
        assert len(result) == 512

    def test_returns_512_floats(self) -> None:
        dummy_landmarks = [(0.0, 0.0)] * 5
        result = align_and_embed(make_blank_image(), dummy_landmarks)
        assert all(isinstance(v, float) for v in result)
