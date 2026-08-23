"""
ml/utils.py
───────────
Image loading and preprocessing utilities for the face pipeline.

Used by:
  - The capture worker (reads JPEG/PNG bytes from blob storage).
  - CLI scripts (test_pipeline.py, calibrate_threshold.py, bench_throughput.py).

Must NOT import from backend.* or depend on HTTP/queue/DB.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# Maximum image side length before downscaling for detection.
# RetinaFace internally uses det_size=(640, 640), so sending a 4K image gives
# no accuracy benefit and wastes memory / CPU decode time.
DEFAULT_MAX_SIDE: int = 1280


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Decode JPEG/PNG image bytes into a BGR uint8 numpy array.

    This is the primary entry point for the capture worker, which reads raw
    image bytes from blob storage.

    Parameters
    ----------
    data : Raw image bytes (JPEG, PNG, BMP, or any format supported by OpenCV).

    Returns
    -------
    (H, W, 3) uint8 numpy array in BGR colour order.

    Raises
    ------
    ValueError
        If ``data`` is empty or OpenCV cannot decode the bytes as an image.
    """
    if not data:
        raise ValueError("image bytes must not be empty")

    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image bytes — unsupported format or corrupted data")

    log.debug("load_image_from_bytes: decoded image shape=%s", image.shape)
    return image


def load_image_from_path(path: Path) -> np.ndarray:
    """
    Load an image from disk into a BGR uint8 numpy array.

    Parameters
    ----------
    path : Path to a JPEG, PNG, or any OpenCV-supported image file.

    Returns
    -------
    (H, W, 3) uint8 numpy array in BGR colour order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV cannot read/decode the file (e.g. wrong format, corrupted).
    """
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image at {path} — unsupported format or corrupted file")

    log.debug("load_image_from_path: loaded %s  shape=%s", path.name, image.shape)
    return image


def resize_for_detection(
    image: np.ndarray,
    max_side: int = DEFAULT_MAX_SIDE,
) -> np.ndarray:
    """
    Downscale ``image`` so its longest side is at most ``max_side`` pixels.

    Small images are returned unchanged (no upscaling). This caps latency
    and memory for high-resolution classroom cameras without degrading
    RetinaFace accuracy (which re-downscales internally to det_size=640×640).

    Parameters
    ----------
    image    : BGR uint8 numpy array (H, W, 3).
    max_side : Maximum allowed length of the longest side. Default 1280.

    Returns
    -------
    Original array if already within bounds, otherwise a downscaled copy.
    Aspect ratio is preserved. Interpolation: INTER_AREA (best for shrinking).
    """
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return image  # no copy needed

    scale = max_side / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    log.debug(
        "resize_for_detection: %dx%d → %dx%d (scale=%.3f)",
        w,
        h,
        new_w,
        new_h,
        scale,
    )
    return resized
