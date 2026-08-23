#!/usr/bin/env python3
"""
scripts/test_pipeline.py
─────────────────────────
CLI smoke-test for the face pipeline on a single image.

Usage:
    python scripts/test_pipeline.py path/to/image.jpg [--threshold 0.45]

Output:
    - Number of faces detected
    - Bounding boxes and detection scores
    - Embedding norm (should be ~1.0 for a properly normalised ArcFace output)
    - Similarity matrix if multiple faces are found (useful sanity check)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Face pipeline smoke test")
    parser.add_argument("image", type=Path, help="Path to image file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Cosine distance threshold to report (default: 0.45)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print(
            "ERROR: cv2 / numpy not installed. " "Run: pip install -r ml/requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    # Add repo root to path so ml.pipeline is importable
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.pipeline import batch_process

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Image shape: {image.shape}")
    faces = batch_process(image)
    print(f"Faces detected: {len(faces)}")

    for i, face in enumerate(faces):
        emb = np.array(face.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        print(
            f"  Face {i}: bbox=({face.bbox.x1},{face.bbox.y1},"
            f"{face.bbox.x2},{face.bbox.y2})  emb_norm={norm:.4f}"
        )

    if len(faces) > 1:
        print("\nPairwise cosine distances:")
        embeddings = [np.array(f.embedding, dtype=np.float32) for f in faces]
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = 1.0 - float(np.dot(embeddings[i], embeddings[j]))
                match = "MATCH" if dist <= args.threshold else "no-match"
                print(f"  Face {i} vs {j}: dist={dist:.4f}  [{match}]")


if __name__ == "__main__":
    main()
