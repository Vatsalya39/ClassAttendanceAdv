"""
tests/test_step1_clahe.py
─────────────────────────
Verification suite for Step 1 — LAB-space CLAHE preprocessing.
Validates:
  1. CLAHE applies strictly to the L (luminance) channel in LAB space,
     leaving A and B chromaticity channels untouched before blending.
  2. Output dimensions and dtype (112x112, uint8) are strictly preserved.
  3. Acceptance criterion:
     - Zero regression (> 0.01) on existing frontal / well-lit test cases.
     - Confirmed distance reduction (improvement) on shadow and backlit test cases.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from ml.pipeline import (
    _align_face_crop,
    _configure_pipeline,
    _embed_aligned_crop,
    apply_clahe_lab,
    detect_faces,
)


def _cos_dist(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    va = va / na
    vb = vb / nb
    return float(1.0 - np.dot(va, vb))


def make_directional_shadow(
    img: np.ndarray, side: str = "right", factor: float = 0.35
) -> np.ndarray:
    h, w = img.shape[:2]
    sh = img.copy().astype(np.float32)
    for c in range(w):
        mult = (
            1.0 - (1.0 - factor) * (c / w) if side == "right" else factor + (1.0 - factor) * (c / w)
        )
        sh[:, c] *= mult
    return sh.clip(0, 255).astype(np.uint8)


def make_backlit(img: np.ndarray, contrast: float = 0.5, brightness: float = -10.0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


class TestClaheProperties:
    def test_dimensions_and_dtype_preserved(self) -> None:
        dummy = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        processed = apply_clahe_lab(dummy)
        assert processed.shape == (112, 112, 3)
        assert processed.dtype == np.uint8

    def test_operates_on_l_channel_preserving_ab_channels(self) -> None:
        # Create an image with known, non-trivial A and B channels
        bgr = np.full((112, 112, 3), (40, 100, 200), dtype=np.uint8)
        lab_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        # When blend_weight is 1.0 (pure CLAHE on L), A and B must remain identical
        processed = apply_clahe_lab(bgr, blend_weight=1.0)
        lab_after = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)

        # A and B channels must match bit-for-bit or within rounding precision of 1 LSB
        np.testing.assert_allclose(lab_after[:, :, 1], lab_before[:, :, 1], atol=1)
        np.testing.assert_allclose(lab_after[:, :, 2], lab_before[:, :, 2], atol=1)

    def test_none_or_empty_returns_unchanged(self) -> None:
        assert apply_clahe_lab(None) is None
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert apply_clahe_lab(empty).size == 0


class TestClaheAcceptanceCriteria:
    @pytest.fixture(scope="class")
    def pipeline_context(self):
        app = _configure_pipeline()
        if app is None:
            pytest.skip("InsightFace model not available in this environment")
        rec_model = app.models.get("recognition")
        if rec_model is None:
            pytest.skip("Recognition model not loaded")

        # Load reference enrollment image
        ref_path = (
            "data/uploads/enrollment-images/23288bba75c946e9ab75fb4a36542076_25BLC1105_6c723aee.jpg"
        )
        if not os.path.exists(ref_path):
            pytest.skip("Test image data not found")

        ref_img = cv2.imread(ref_path)
        ref_dets = detect_faces(ref_img)
        assert len(ref_dets) >= 1
        ref_crop = _align_face_crop(ref_img, ref_dets[0].landmarks, rec_model)

        # Load well-lit probe crops
        probe_paths = [
            (
                "Enroll_2",
                "data/uploads/enrollment-images/3994b52d39264de09304ae1156345680_25BLC1105_5779cf67.jpg",
            ),
            (
                "Session_A",
                "data/uploads/session-images/38b839c05ba0484b83307130c9bd458d_capture_9c7d8376a5d6.jpg",
            ),
            (
                "Session_B",
                "data/uploads/session-images/6c21b68ddb1b4003881990759a781ed6_capture_5885c698cbc3.jpg",
            ),
            (
                "Session_C",
                "data/uploads/session-images/f2f3393ce9334fa9b3a4e8e24dda8de1_capture_0aacf183edc0.jpg",
            ),
        ]
        probes = {}
        for name, p in probe_paths:
            if os.path.exists(p):
                img = cv2.imread(p)
                dets = detect_faces(img)
                if dets:
                    probes[name] = _align_face_crop(img, dets[0].landmarks, rec_model)

        return rec_model, ref_crop, probes

    def test_well_lit_frontal_cases_no_regression_exceeding_0_01(self, pipeline_context) -> None:
        rec_model, ref_crop, probes = pipeline_context
        ref_emb_raw = _embed_aligned_crop(ref_crop, rec_model, tta_flip=False, apply_clahe=False)
        ref_emb_clahe = _embed_aligned_crop(ref_crop, rec_model, tta_flip=False, apply_clahe=True)

        deltas = []
        for name, crop in probes.items():
            emb_raw = _embed_aligned_crop(crop, rec_model, tta_flip=False, apply_clahe=False)
            emb_clahe = _embed_aligned_crop(crop, rec_model, tta_flip=False, apply_clahe=True)
            d_raw = _cos_dist(ref_emb_raw, emb_raw)
            d_clahe = _cos_dist(ref_emb_clahe, emb_clahe)
            delta = d_clahe - d_raw
            deltas.append(delta)
            # Strict gating assertion: no single well-lit case may regress > 0.010
            assert (
                delta <= 0.010
            ), f"Regression on {name}: baseline={d_raw:.4f}, clahe={d_clahe:.4f}, delta={delta:+.4f} > +0.010"

        # Mean delta across well-lit cases should be neutral or improving
        assert np.mean(deltas) <= 0.005

    def test_shadow_and_backlit_cases_improve_distance(self, pipeline_context) -> None:
        rec_model, ref_crop, probes = pipeline_context
        ref_emb_raw = _embed_aligned_crop(ref_crop, rec_model, tta_flip=False, apply_clahe=False)
        ref_emb_clahe = _embed_aligned_crop(ref_crop, rec_model, tta_flip=False, apply_clahe=True)

        backlit_deltas = []
        shadow_deltas = []

        for _name, crop in probes.items():
            # Backlit
            bl_crop = make_backlit(crop)
            bl_raw = _embed_aligned_crop(bl_crop, rec_model, tta_flip=False, apply_clahe=False)
            bl_clahe = _embed_aligned_crop(bl_crop, rec_model, tta_flip=False, apply_clahe=True)
            d_bl_raw = _cos_dist(ref_emb_raw, bl_raw)
            d_bl_clahe = _cos_dist(ref_emb_clahe, bl_clahe)
            backlit_deltas.append(d_bl_clahe - d_bl_raw)

            # Directional Shadow
            for side in ["right", "left"]:
                sh_crop = make_directional_shadow(crop, side=side)
                sh_raw = _embed_aligned_crop(sh_crop, rec_model, tta_flip=False, apply_clahe=False)
                sh_clahe = _embed_aligned_crop(sh_crop, rec_model, tta_flip=False, apply_clahe=True)
                d_sh_raw = _cos_dist(ref_emb_raw, sh_raw)
                d_sh_clahe = _cos_dist(ref_emb_clahe, sh_clahe)
                shadow_deltas.append(d_sh_clahe - d_sh_raw)

        # Confirm overall improvement (negative mean delta = lower distance to reference)
        assert (
            np.mean(backlit_deltas) < 0.0
        ), f"Backlit mean delta not negative: {np.mean(backlit_deltas):+.4f}"
        assert (
            np.mean(shadow_deltas) < 0.0
        ), f"Shadow mean delta not negative: {np.mean(shadow_deltas):+.4f}"


class TestAdaptiveClahe:
    def test_well_lit_crop_does_not_trigger_enhancement(self) -> None:
        from ml.pipeline import needs_clahe_enhancement

        # Synthesize well-lit face with balanced luminance (mean ~130, std ~50)
        well_lit = np.full((112, 112, 3), 130, dtype=np.uint8)
        assert not needs_clahe_enhancement(well_lit)

    def test_backlit_crop_triggers_enhancement(self) -> None:
        from ml.pipeline import needs_clahe_enhancement

        # Very dark crop (L < 40)
        dark = np.full((112, 112, 3), 35, dtype=np.uint8)
        assert needs_clahe_enhancement(dark)

    def test_directional_shadow_triggers_enhancement(self) -> None:
        from ml.pipeline import needs_clahe_enhancement

        # Harsh left-right asymmetry
        asym = np.zeros((112, 112, 3), dtype=np.uint8)
        asym[:, :56] = 40
        asym[:, 56:] = 180
        assert needs_clahe_enhancement(asym)
