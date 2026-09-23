#!/usr/bin/env python3
"""
scripts/build_calibration_dataset.py
─────────────────────────────────────
Builds the comprehensive calibration and evaluation datasets for Phase B:
1. data/calibration_pairs/:
   - 10 distinct student identities (student_001 to student_010)
   - 5 variations per student (original, shadow_left, shadow_right, backlit, underexposed)
   - Generates 100 genuine pairs and 1,125 impostor pairs
2. data/pose_eval/:
   - 30 pose-labeled frames across 3 ground truth categories:
     * frontal (yaw, pitch in [-15, +15] deg) -> expected: ACCEPT (reject=False)
     * turned (|yaw| >= 25 deg) -> expected: REJECT (reject=True)
     * looking_down (|pitch| >= 25 deg) -> expected: REJECT (reject=True)
   - Accompanied by data/pose_eval/ground_truth.json
"""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.pipeline import batch_process, detect_faces


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


def make_underexposed(img: np.ndarray, alpha: float = 0.6, beta: float = -15.0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def crop_face(img: np.ndarray, bbox: Any, pad_factor: float = 0.45) -> np.ndarray:
    h, w = img.shape[:2]
    bw = bbox.width
    bh = bbox.height
    pad_x = int(bw * pad_factor)
    pad_y = int(bh * pad_factor)
    x1 = max(0, bbox.x1 - pad_x)
    y1 = max(0, bbox.y1 - pad_y)
    x2 = min(w, bbox.x2 + pad_x)
    y2 = min(h, bbox.y2 + pad_y)
    return img[y1:y2, x1:x2].copy()


def warp_pose_3d(img: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Apply synthetic 3D rotation with perspective projection to generate controlled head poses."""
    h, w = img.shape[:2]
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # 3D rotation matrices (camera aligned: Y down, X right, Z forward)
    rot_x = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float64
    )
    r_mat = rot_y @ rot_x

    # Camera intrinsic
    f = float(max(w, h)) * 2.0
    k_mat = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    k_inv = np.linalg.inv(k_mat)

    # Homography for planar face under 3D rotation
    h_mat = k_mat @ r_mat @ k_inv
    warped = cv2.warpPerspective(img, h_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    calib_dir = repo_root / "data" / "calibration_pairs"
    pose_dir = repo_root / "data" / "pose_eval"

    # Reset directories
    if calib_dir.exists():
        shutil.rmtree(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)

    if pose_dir.exists():
        shutil.rmtree(pose_dir)
    pose_dir.mkdir(parents=True, exist_ok=True)
    (pose_dir / "frontal").mkdir(parents=True, exist_ok=True)
    (pose_dir / "turned").mkdir(parents=True, exist_ok=True)
    (pose_dir / "looking_down").mkdir(parents=True, exist_ok=True)

    print("═══════════════════════════════════════════════════════════════")
    print("BUILDING MULTI-IDENTITY CALIBRATION DATASET (10 IDENTITIES)")
    print("═══════════════════════════════════════════════════════════════")

    # ── Identity 1: 25BLC1105 (from enrollment uploads) ──────────────────────
    id1_files = sorted(glob.glob(str(repo_root / "data/uploads/enrollment-images/*.jpg")))
    assert len(id1_files) > 0, "Enrollment images for student 1 must exist"
    id1_base = cv2.imread(id1_files[0])
    assert id1_base is not None, "Failed to load enrollment image"

    identities_raw: list[tuple[str, np.ndarray]] = [("student_001", id1_base)]

    # ── Extract distinct students from classroom photos ───────────────────────
    # Sources: classroomFourHigh.jpg, classroomTwo.jpg, classroomOne.jpg
    sources = [
        (
            "classroomFourHigh.jpg",
            [(0, 0.45), (1, 0.45), (3, 0.45), (4, 0.45), (6, 0.45), (8, 0.45)],
        ),
        ("classroomTwo.jpg", [(0, 0.40), (1, 0.40), (7, 0.40)]),
    ]

    student_counter = 2
    for img_name, face_specs in sources:
        full_img = cv2.imread(str(repo_root / img_name))
        assert full_img is not None, f"Failed to load {img_name}"
        faces = detect_faces(full_img)
        for face_idx, pad in face_specs:
            if student_counter > 10:
                break
            if face_idx < len(faces):
                f = faces[face_idx]
                crop = crop_face(full_img, f.bbox, pad_factor=pad)
                # Verify that batch_process can detect the face in the crop
                sub = batch_process(crop)
                if sub:
                    identities_raw.append((f"student_{student_counter:03d}", crop))
                    print(
                        f"Extracted {identities_raw[-1][0]} from {img_name} face {face_idx} (bbox={f.bbox.width}x{f.bbox.height})"
                    )
                    student_counter += 1

    assert len(identities_raw) >= 10, f"Expected 10 identities, got {len(identities_raw)}"

    # ── Write calibration images with lighting augmentations ─────────────────
    for student_id, base_img in identities_raw[:10]:
        s_dir = calib_dir / student_id
        s_dir.mkdir(parents=True, exist_ok=True)

        # 1. Base / original
        cv2.imwrite(str(s_dir / "01_original.jpg"), base_img)
        # 2. Left shadow
        cv2.imwrite(
            str(s_dir / "02_shadow_left.jpg"),
            make_directional_shadow(base_img, side="left", factor=0.35),
        )
        # 3. Right shadow
        cv2.imwrite(
            str(s_dir / "03_shadow_right.jpg"),
            make_directional_shadow(base_img, side="right", factor=0.35),
        )
        # 4. Backlit
        cv2.imwrite(
            str(s_dir / "04_backlit.jpg"), make_backlit(base_img, contrast=0.5, brightness=-10.0)
        )
        # 5. Underexposed
        cv2.imwrite(
            str(s_dir / "05_underexposed.jpg"), make_underexposed(base_img, alpha=0.6, beta=-15.0)
        )

    print("Successfully generated 10 student directories with 5 images each (50 images total).")

    # ── Pose-labeled dataset generation ──────────────────────────────────────
    print("\n═══════════════════════════════════════════════════════════════")
    print("BUILDING POSE-LABELED EVALUATION DATASET (30 FRAMES)")
    print("═══════════════════════════════════════════════════════════════")

    pose_labels: list[dict[str, Any]] = []

    # Select high-resolution base frontal crops for pose transformations
    base_crops = [
        identities_raw[1][1],
        identities_raw[2][1],
        identities_raw[3][1],
        identities_raw[4][1],
        identities_raw[5][1],
    ]

    # 1. Frontal frames: yaw in [-12, +12], pitch in [-10, +10] -> ACCEPT
    frontal_specs = [
        (0.0, 0.0),
        (+5.0, 0.0),
        (-5.0, 0.0),
        (0.0, +6.0),
        (0.0, -6.0),
        (+10.0, +5.0),
        (-10.0, +5.0),
        (+8.0, -7.0),
        (-8.0, -7.0),
        (+12.0, 0.0),
    ]
    for i, (y, p) in enumerate(frontal_specs):
        crop = base_crops[i % len(base_crops)]
        frame = warp_pose_3d(crop, yaw_deg=y, pitch_deg=p)
        fn = f"frontal_{i+1:02d}.jpg"
        rel_path = f"frontal/{fn}"
        cv2.imwrite(str(pose_dir / rel_path), frame)
        pose_labels.append(
            {
                "image_path": rel_path,
                "category": "frontal",
                "ground_truth_yaw": y,
                "ground_truth_pitch": p,
                "expected_reject": False,
            }
        )

    # 2. Turned frames: |yaw| >= 28 deg (e.g. 28 to 45 deg) -> REJECT
    turned_specs = [
        (+28.0, 0.0),
        (-28.0, 0.0),
        (+32.0, +5.0),
        (-32.0, +5.0),
        (+35.0, -5.0),
        (-35.0, -5.0),
        (+40.0, 0.0),
        (-40.0, 0.0),
        (+45.0, +4.0),
        (-45.0, -4.0),
    ]
    for i, (y, p) in enumerate(turned_specs):
        crop = base_crops[i % len(base_crops)]
        frame = warp_pose_3d(crop, yaw_deg=y, pitch_deg=p)
        fn = f"turned_{i+1:02d}.jpg"
        rel_path = f"turned/{fn}"
        cv2.imwrite(str(pose_dir / rel_path), frame)
        pose_labels.append(
            {
                "image_path": rel_path,
                "category": "turned",
                "ground_truth_yaw": y,
                "ground_truth_pitch": p,
                "expected_reject": True,
            }
        )

    # 3. Looking-down frames: pitch >= 26 deg (e.g. 26 to 40 deg) -> REJECT
    down_specs = [
        (0.0, +26.0),
        (0.0, -26.0),
        (+5.0, +28.0),
        (-5.0, +28.0),
        (0.0, +32.0),
        (+8.0, +32.0),
        (-8.0, +32.0),
        (0.0, +35.0),
        (+5.0, +38.0),
        (0.0, +40.0),
    ]
    for i, (y, p) in enumerate(down_specs):
        crop = base_crops[i % len(base_crops)]
        frame = warp_pose_3d(crop, yaw_deg=y, pitch_deg=p)
        fn = f"looking_down_{i+1:02d}.jpg"
        rel_path = f"looking_down/{fn}"
        cv2.imwrite(str(pose_dir / rel_path), frame)
        pose_labels.append(
            {
                "image_path": rel_path,
                "category": "looking_down",
                "ground_truth_yaw": y,
                "ground_truth_pitch": p,
                "expected_reject": True,
            }
        )

    manifest_path = pose_dir / "ground_truth.json"
    with open(manifest_path, "w", encoding="utf-8") as out_f:
        json.dump(pose_labels, out_f, indent=2)

    print(f"Generated 30 pose-labeled frames with manifest at {manifest_path}:")
    print(f"  Frontal (accept): {len(frontal_specs)}")
    print(f"  Turned (reject): {len(turned_specs)}")
    print(f"  Looking-down (reject): {len(down_specs)}")
    print("Dataset build complete!")


if __name__ == "__main__":
    main()
