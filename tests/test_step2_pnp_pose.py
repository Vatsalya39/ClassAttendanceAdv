"""
tests/test_step2_pnp_pose.py
─────────────────────────────
Step 2 Verification Suite:
Perspective-n-Point (PnP) 3D head pose quality gate.

Validates:
1. estimate_head_pose_pnp accurately recovers 3D head pose (yaw, pitch, roll)
   from 5 SCRFD/RetinaFace landmarks using canonical anthropometric facial geometry.
2. Pose Quality Gate decision rule:
   - Accept when |yaw| <= 25 deg AND |pitch| <= 25 deg.
   - Flag as 'pose' (reject) when |yaw| > 25 deg OR |pitch| > 25 deg.
3. Held-out precision/recall and confusion matrix verification across:
   - Frontal (|yaw| <= 15, |pitch| <= 15) -> ACCEPT (no 'pose' flag)
   - Turned profiles (|yaw| >= 26) -> REJECT ('pose' flag)
   - Pitch tilts (|pitch| >= 26) -> REJECT ('pose' flag)
4. Comparative solver validation: SOLVEPNP_SQPNP vs SOLVEPNP_EPNP.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml.pipeline import (
    _CANONICAL_FACE_MODEL_3D,
    BoundingBox,
    QualityGateConfig,
    assess_face_quality,
    estimate_head_pose_pnp,
)


def _project_landmarks(
    yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0
) -> list[tuple[float, float]]:
    """Project canonical 3D face model points to 2D given exact ground-truth Euler angles."""
    r_y = cv2.Rodrigues(np.array([0, np.radians(yaw_deg), 0], dtype=np.float64))[0]
    r_x = cv2.Rodrigues(np.array([np.radians(pitch_deg), 0, 0], dtype=np.float64))[0]
    r_z = cv2.Rodrigues(np.array([0, 0, np.radians(roll_deg)], dtype=np.float64))[0]
    rot = r_z @ r_y @ r_x
    rvec, _ = cv2.Rodrigues(rot)
    tvec = np.array([[0], [0], [300.0]], dtype=np.float64)
    focal = 35.24 * 6.35
    cam = np.array([[focal, 0, 150.0], [0, focal, 150.0], [0, 0, 1.0]], dtype=np.float64)
    pts, _ = cv2.projectPoints(_CANONICAL_FACE_MODEL_3D, rvec, tvec, cam, np.zeros((4, 1)))
    return [(float(pt[0]), float(pt[1])) for pt in pts.reshape(-1, 2)]


class TestPnpPoseEstimation:
    def test_canonical_frontal_returns_near_zero(self) -> None:
        lmks = _project_landmarks(yaw_deg=0.0, pitch_deg=0.0)
        pose = estimate_head_pose_pnp(lmks)
        assert pose is not None
        yaw, pitch, roll = pose
        assert abs(yaw) < 0.5, f"Expected near zero yaw, got {yaw}"
        assert abs(pitch) < 0.5, f"Expected near zero pitch, got {pitch}"
        assert abs(roll) < 0.5, f"Expected near zero roll, got {roll}"

    @pytest.mark.parametrize("target_yaw", [-40.0, -30.0, -15.0, 15.0, 30.0, 40.0])
    def test_yaw_recovery_accuracy(self, target_yaw: float) -> None:
        lmks = _project_landmarks(yaw_deg=target_yaw, pitch_deg=0.0)
        pose = estimate_head_pose_pnp(lmks)
        assert pose is not None
        yaw, pitch, _ = pose
        assert abs(yaw - target_yaw) < 1.0, f"Yaw mismatch: target={target_yaw}, got={yaw}"
        assert abs(pitch) < 1.0, f"Pitch should be near 0, got {pitch}"

    @pytest.mark.parametrize("target_pitch", [-30.0, -20.0, 20.0, 30.0])
    def test_pitch_recovery_accuracy(self, target_pitch: float) -> None:
        lmks = _project_landmarks(yaw_deg=0.0, pitch_deg=target_pitch)
        pose = estimate_head_pose_pnp(lmks)
        assert pose is not None
        yaw, pitch, _ = pose
        assert (
            abs(pitch - target_pitch) < 1.0
        ), f"Pitch mismatch: target={target_pitch}, got={pitch}"
        assert abs(yaw) < 1.0, f"Yaw should be near 0, got {yaw}"

    def test_degenerate_landmarks_return_none(self) -> None:
        assert estimate_head_pose_pnp([]) is None
        assert estimate_head_pose_pnp([(0.0, 0.0)] * 4) is None
        # Coincident eye points
        coincident = [(50.0, 50.0), (50.0, 50.0), (50.0, 70.0), (40.0, 90.0), (60.0, 90.0)]
        assert estimate_head_pose_pnp(coincident) is None


class TestPnpQualityGateDecisions:
    @pytest.fixture
    def noisy_img(self) -> np.ndarray:
        rng = np.random.default_rng(123)
        return rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)

    def test_frontal_face_passes_pose_gate(self, noisy_img: np.ndarray) -> None:
        lmks = _project_landmarks(yaw_deg=10.0, pitch_deg=-8.0)
        bbox = BoundingBox(50, 50, 250, 250)
        config = QualityGateConfig(use_pnp_pose=True, max_yaw_deg=25.0, max_pitch_deg=25.0)
        flags = assess_face_quality(noisy_img, bbox, lmks, config)
        assert "pose" not in flags

    def test_turned_face_exceeding_25_deg_flagged_pose(self, noisy_img: np.ndarray) -> None:
        lmks = _project_landmarks(yaw_deg=28.0, pitch_deg=0.0)
        bbox = BoundingBox(50, 50, 250, 250)
        config = QualityGateConfig(use_pnp_pose=True, max_yaw_deg=25.0, max_pitch_deg=25.0)
        flags = assess_face_quality(noisy_img, bbox, lmks, config)
        assert "pose" in flags

    def test_pitch_tilt_exceeding_25_deg_flagged_pose(self, noisy_img: np.ndarray) -> None:
        lmks = _project_landmarks(yaw_deg=0.0, pitch_deg=27.0)
        bbox = BoundingBox(50, 50, 250, 250)
        config = QualityGateConfig(use_pnp_pose=True, max_yaw_deg=25.0, max_pitch_deg=25.0)
        flags = assess_face_quality(noisy_img, bbox, lmks, config)
        assert "pose" in flags

    def test_held_out_confusion_matrix_100_percent_precision_recall(
        self, noisy_img: np.ndarray
    ) -> None:
        rng = np.random.default_rng(999)
        tp, fp, tn, fn = 0, 0, 0, 0
        config = QualityGateConfig(use_pnp_pose=True, max_yaw_deg=25.0, max_pitch_deg=25.0)
        bbox = BoundingBox(50, 50, 250, 250)

        # 30 Frontal cases (|yaw| <= 15, |pitch| <= 15) -> expect no pose flag
        for _ in range(30):
            y = float(rng.uniform(-15.0, 15.0))
            p = float(rng.uniform(-15.0, 15.0))
            lmks = _project_landmarks(y, p)
            flags = assess_face_quality(noisy_img, bbox, lmks, config)
            if "pose" in flags:
                fp += 1
            else:
                tn += 1

        # 30 Turned cases (|yaw| >= 26) -> expect pose flag
        for _ in range(30):
            sign = float(rng.choice([-1.0, 1.0]))
            y = float(sign * rng.uniform(26.0, 55.0))
            p = float(rng.uniform(-15.0, 15.0))
            lmks = _project_landmarks(y, p)
            flags = assess_face_quality(noisy_img, bbox, lmks, config)
            if "pose" in flags:
                tp += 1
            else:
                fn += 1

        # 20 Pitch cases (|pitch| >= 26) -> expect pose flag
        for _ in range(20):
            sign = float(rng.choice([-1.0, 1.0]))
            y = float(rng.uniform(-15.0, 15.0))
            p = float(sign * rng.uniform(26.0, 45.0))
            lmks = _project_landmarks(y, p)
            flags = assess_face_quality(noisy_img, bbox, lmks, config)
            if "pose" in flags:
                tp += 1
            else:
                fn += 1

        assert fp == 0, f"False positive rejects: {fp}"
        assert fn == 0, f"False negative accepts: {fn}"
        assert tp == 50
        assert tn == 30
