#!/usr/bin/env python3
"""
scripts/calibrate_threshold.py
───────────────────────────────
Compute false-accept and false-reject rates at a range of cosine distance
thresholds, and recommend a threshold prioritising precision (low FAR).

This script is used in Phase 8 with real classroom-condition photos.

Input directory structure
─────────────────────────
The ``pairs_dir`` must contain subdirectories named by student ID:

    pairs_dir/
        student_001/        ← same-person pairs (intra-class)
            img_a.jpg
            img_b.jpg
            img_c.jpg
        student_002/
            ...

Known-different pairs are sampled by comparing images from different
student subdirectories.

Usage
─────
    python scripts/calibrate_threshold.py \\
        --pairs-dir data/calibration_pairs \\
        --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \\
        --output results/threshold_calibration.png
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold calibration script")
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    )
    parser.add_argument(
        "--clahe-mode",
        type=str,
        choices=["none", "global", "adaptive"],
        default="adaptive",
        help="CLAHE mode: none, global, or adaptive",
    )
    parser.add_argument("--no-clahe", action="store_true", help="Shortcut for --clahe-mode none")
    parser.add_argument(
        "--model-pack", type=str, default="buffalo_l", help="InsightFace model pack"
    )
    parser.add_argument("--output", type=Path, default=Path("threshold_calibration.png"))
    args = parser.parse_args()

    if args.no_clahe:
        args.clahe_mode = "none"

    if not args.pairs_dir.exists():
        print(f"ERROR: pairs directory not found: {args.pairs_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERROR: run `pip install -r ml/requirements.txt`", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.pipeline import batch_process

    # ── Collect embeddings per student ────────────────────────────────────────
    student_embeddings: dict[str, list[np.ndarray]] = {}
    student_dirs = sorted(d for d in args.pairs_dir.iterdir() if d.is_dir())

    print(
        f"Found {len(student_dirs)} student directories. "
        f"CLAHE_MODE={args.clahe_mode.upper()}, Model={args.model_pack}"
    )
    for student_dir in student_dirs:
        embeds: list[np.ndarray] = []
        for img_path in sorted(student_dir.glob("*.jpg")) + sorted(student_dir.glob("*.png")):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            faces = batch_process(
                image,
                apply_clahe=(args.clahe_mode != "none"),
                clahe_mode=args.clahe_mode,
                model_pack=args.model_pack,
            )
            if faces:
                # Module 4: pick the LARGEST face (main subject), not blindly
                # faces[0] — detection order is arbitrary and a background
                # bystander's embedding would poison the pair distributions.
                main_face = max(faces, key=lambda f: f.bbox.area)
                embeds.append(np.array(main_face.embedding, dtype=np.float32))
        if embeds:
            student_embeddings[student_dir.name] = embeds
            print(f"  {student_dir.name}: {len(embeds)} embeddings")

    # ── Build same/different pairs ────────────────────────────────────────────
    same_distances: list[float] = []
    diff_distances: list[float] = []

    for _, embeds in student_embeddings.items():
        for a, b in itertools.combinations(embeds, 2):
            dist = float(1.0 - float(np.dot(a, b)))
            same_distances.append(dist)

    student_ids = list(student_embeddings.keys())
    for i, s1 in enumerate(student_ids):
        for s2 in student_ids[i + 1 :]:
            for e1 in student_embeddings[s1]:
                for e2 in student_embeddings[s2]:
                    dist = float(1.0 - float(np.dot(e1, e2)))
                    diff_distances.append(dist)

    same_arr = np.array(same_distances, dtype=np.float64)
    diff_arr = np.array(diff_distances, dtype=np.float64)

    mean_same, std_same = float(np.mean(same_arr)), float(np.std(same_arr))
    mean_diff, std_diff = float(np.mean(diff_arr)), float(np.std(diff_arr))
    margin = mean_diff - mean_same

    print(
        f"\nSame-person (genuine) pairs: {len(same_distances)} | mean={mean_same:.4f}, std={std_same:.4f}"
    )
    print(
        f"Different-person (impostor) pairs: {len(diff_distances)} | mean={mean_diff:.4f}, std={std_diff:.4f}"
    )
    print(f"Decision Margin (mean_diff - mean_same): {margin:+.4f}")

    # ── Compute FAR / FRR at each threshold ───────────────────────────────────
    print(f"\n{'Threshold':>10} {'FAR':>8} {'FRR':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)

    far_list, frr_list, thresholds = [], [], []
    for t in args.thresholds:
        fa = sum(1 for d in diff_distances if d <= t)
        fr = sum(1 for d in same_distances if d > t)
        far = fa / max(len(diff_distances), 1)
        frr = fr / max(len(same_distances), 1)
        precision = (len(same_distances) - fr) / max((len(same_distances) - fr) + fa, 1)
        recall = 1.0 - frr
        far_list.append(far)
        frr_list.append(frr)
        thresholds.append(t)
        print(f"{t:>10.2f} {far:>8.4f} {frr:>8.4f} {precision:>10.4f} {recall:>8.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(same_distances, bins=50, alpha=0.6, label="Same person", color="green")
        axes[0].hist(diff_distances, bins=50, alpha=0.6, label="Different person", color="red")
        axes[0].set_xlabel("Cosine distance")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distance distribution")
        axes[0].legend()

        axes[1].plot(thresholds, far_list, label="FAR (false accept)", marker="o")
        axes[1].plot(
            thresholds, frr_list, label="FRR (false reject / review queue rate)", marker="s"
        )
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Rate")
        axes[1].set_title("FAR / FRR vs Threshold")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(str(args.output), dpi=150)
        print(f"\nPlot saved to {args.output}")
    else:
        print("\nNote: matplotlib not installed; histogram plot skipped.")

    # ── Suggest threshold (FAR < 1%) ─────────────────────────────────────────
    candidates = [
        (t, far, frr)
        for t, far, frr in zip(thresholds, far_list, frr_list, strict=False)
        if far < 0.01
    ]
    if candidates:
        best_t, best_far, best_frr = min(candidates, key=lambda x: x[2])
        print(
            f"\nRecommended threshold (FAR < 1%): {best_t} "
            f"(FAR={best_far:.4f}, FRR={best_frr:.4f})"
        )
        print(f"Review queue rate ≈ {best_frr * 100:.1f}% of genuine faces will go to review.")
    else:
        print(
            "\nNo threshold achieves FAR < 1% in this dataset. Lower thresholds or collect more data."
        )


if __name__ == "__main__":
    main()
