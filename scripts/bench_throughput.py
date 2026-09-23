#!/usr/bin/env python3
"""
scripts/bench_throughput.py
────────────────────────────
Measure face pipeline throughput (faces/second) at realistic classroom
batch sizes, and estimate how many concurrent worker replicas are needed
to keep queue wait time under a target latency when N sections all submit
captures within the same 60-second window.

This output feeds directly into the KEDA scale-rule max-replica numbers
for the ml-inference Container Apps workload profile in Phase 6.

Usage
─────
    python scripts/bench_throughput.py \\
        --image-dir data/bench_images \\
        --batch-sizes 1 5 10 20 \\
        --target-latency-s 30 \\
        --concurrent-sections 50 200

Output (example):
    Batch size 1:  12.3 faces/s   (0.081 s/face)
    Batch size 5:  47.8 faces/s   (0.021 s/face)
    ...

    Estimate for 50 concurrent sections (avg 35 faces/photo):
      Total faces in burst: 1750
      At 47.8 faces/s per replica, need ≥ 2 replicas to finish in 30s

    Estimate for 200 concurrent sections (avg 35 faces/photo):
      Total faces in burst: 7000
      At 47.8 faces/s per replica, need ≥ 5 replicas to finish in 30s
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Face pipeline throughput benchmark")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory of benchmark images (JPEG/PNG)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Number of images per batch to benchmark",
    )
    parser.add_argument(
        "--target-latency-s",
        type=float,
        default=30.0,
        help="Target max job processing time in seconds",
    )
    parser.add_argument(
        "--concurrent-sections",
        type=int,
        nargs="+",
        default=[50, 200],
        help="Simulated concurrent section captures for replica estimation",
    )
    parser.add_argument(
        "--avg-faces-per-photo",
        type=int,
        default=35,
        help="Average number of students visible per classroom photo",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Warm-up iterations before timing (to prime CUDA kernels)",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=10,
        help="Timed iterations per batch size",
    )
    parser.add_argument(
        "--model-pack",
        type=str,
        default="buffalo_l",
        help="InsightFace model pack (e.g. buffalo_l or antelopev2)",
    )
    args = parser.parse_args()

    if not args.image_dir.exists():
        print(f"ERROR: image directory not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        print("ERROR: run `pip install -r ml/requirements.txt`", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

    from ml.pipeline import _get_app, batch_process  # noqa: E402

    # Pre-load model
    _get_app(model_pack=args.model_pack)

    all_images = [
        cv2.imread(str(p))
        for p in sorted(args.image_dir.glob("*.jpg"))
        if cv2.imread(str(p)) is not None
    ]
    if not all_images:
        print("ERROR: no readable images found in --image-dir", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_images)} benchmark images. Model pack: {args.model_pack}")
    print(f"Warm-up: {args.warmup_iters} iterations")
    print(f"Timed:   {args.bench_iters} iterations\n")

    best_fps = 0.0
    for batch_size in args.batch_sizes:
        images = (all_images * math.ceil(batch_size / max(len(all_images), 1)))[:batch_size]

        # Warm-up
        for _ in range(args.warmup_iters):
            for img in images:
                batch_process(img, model_pack=args.model_pack)

        # Timed
        total_faces = 0
        start = time.perf_counter()
        for _ in range(args.bench_iters):
            for img in images:
                faces = batch_process(img, model_pack=args.model_pack)
                total_faces += len(faces)
        elapsed = time.perf_counter() - start

        total_images = batch_size * args.bench_iters
        fps_images = total_images / elapsed
        fps_faces = total_faces / elapsed if total_faces else 0
        ms_per_image = (elapsed / total_images) * 1000

        print(
            f"Batch size {batch_size:>3}: "
            f"{fps_images:>7.1f} images/s  "
            f"{fps_faces:>7.1f} faces/s  "
            f"({ms_per_image:.1f} ms/image)"
        )
        if fps_faces > best_fps:
            best_fps = fps_faces

    # ── Replica estimate ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Target latency: {args.target_latency_s}s")
    print(f"Avg faces/photo: {args.avg_faces_per_photo}")
    print(f"Best throughput: {best_fps:.1f} faces/s per replica")
    print()

    for n_sections in args.concurrent_sections:
        total_faces = n_sections * args.avg_faces_per_photo
        faces_per_second_needed = total_faces / args.target_latency_s
        replicas_needed = math.ceil(faces_per_second_needed / max(best_fps, 0.001))
        print(
            f"  {n_sections:>4} concurrent sections → "
            f"{total_faces:>7} total faces → "
            f"≥ {replicas_needed} replica(s) needed"
        )

    print(
        "\nSet KEDA maxReplicaCount to the largest number above, "
        "plus ~20% headroom for detection overhead."
    )


if __name__ == "__main__":
    main()
