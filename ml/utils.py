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

# Maximum image side length before downscaling for SINGLE-STUDENT ENROLLMENT
# SELFIES only. RetinaFace internally re-downscales to its det_size anyway,
# so a 4K selfie gains nothing and wastes decode time.
#
# IMPORTANT (Module 2): classroom session/group photos are NOT downscaled —
# they keep their native resolution so tiled detection can resolve small
# back-row faces. See ml.pipeline.batch_process(tiled_config=...).
DEFAULT_MAX_SIDE: int = 1280


def validate_image_resolution(
    image: np.ndarray,
    min_side: int,
) -> tuple[bool, tuple[int, int]]:
    """
    Check that ``image`` meets a minimum side-length floor (session photos).

    Back-row faces in a full classroom shot need roughly 1024px on the
    shorter side to embed reliably — the capture endpoint uses this to
    reject low-resolution uploads early (422) instead of letting the
    worker flood the review queue with unrecognizable faces.

    Parameters
    ----------
    image    : Decoded BGR uint8 numpy array (H, W, 3).
    min_side : Minimum acceptable length of the SHORTER side in pixels.
               Values <= 0 disable the check (always pass).

    Returns
    -------
    (ok, (width, height)) — ok=True when the floor is met or disabled.
    """
    h, w = image.shape[:2]
    if min_side <= 0:
        return True, (w, h)
    return min(h, w) >= min_side, (w, h)


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

    ENROLLMENT PHOTOS ONLY (Module 2): a close-up selfie needs no more than
    ~1280 px for reliable detection, and RetinaFace re-downscales internally
    to its det_size anyway.

    Session/classroom photos must NOT be passed through this function —
    the worker hands them to batch_process() at NATIVE resolution (with
    optional tiled detection for wide shots).

    Small images are returned unchanged (no upscaling). This caps latency
    and memory for high-resolution enrollment cameras without degrading
    RetinaFace accuracy.

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
