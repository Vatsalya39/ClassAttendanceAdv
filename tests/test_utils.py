"""
tests/test_utils.py
────────────────────
Tests for ml/utils.py — image loading and preprocessing utilities.

These tests use only numpy and opencv-python-headless (no insightface),
so they always run in CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ml.utils import (
    DEFAULT_MAX_SIDE,
    load_image_from_bytes,
    load_image_from_path,
    resize_for_detection,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_bgr_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a valid (H, W, 3) uint8 BGR array."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def encode_to_jpeg(image: np.ndarray) -> bytes:
    """Encode a numpy array to JPEG bytes using OpenCV."""
    ok, buf = cv2.imencode(".jpg", image)
    assert ok, "cv2.imencode failed in test helper"
    return buf.tobytes()


def encode_to_png(image: np.ndarray) -> bytes:
    """Encode a numpy array to PNG bytes using OpenCV."""
    ok, buf = cv2.imencode(".png", image)
    assert ok, "cv2.imencode failed in test helper"
    return buf.tobytes()


# ─── load_image_from_bytes ────────────────────────────────────────────────────


class TestLoadImageFromBytes:
    def test_jpeg_roundtrip(self) -> None:
        original = make_bgr_image(100, 120)
        original[:] = 128  # mid-grey, survives JPEG compression
        decoded = load_image_from_bytes(encode_to_jpeg(original))
        assert decoded.dtype == np.uint8
        assert decoded.ndim == 3
        assert decoded.shape[2] == 3
        assert decoded.shape[0] == 100
        assert decoded.shape[1] == 120

    def test_png_roundtrip_exact(self) -> None:
        original = make_bgr_image(80, 90)
        original[10:20, 30:40] = [255, 0, 0]  # blue patch
        decoded = load_image_from_bytes(encode_to_png(original))
        assert decoded.shape == original.shape
        np.testing.assert_array_equal(decoded, original)

    def test_empty_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            load_image_from_bytes(b"")

    def test_garbage_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="decode"):
            load_image_from_bytes(b"\x00\x01\x02\x03garbage data")

    def test_returns_bgr_uint8(self) -> None:
        decoded = load_image_from_bytes(encode_to_jpeg(make_bgr_image()))
        assert decoded.dtype == np.uint8
        assert decoded.ndim == 3


# ─── load_image_from_path ─────────────────────────────────────────────────────


class TestLoadImageFromPath:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.jpg"
        with pytest.raises(FileNotFoundError):
            load_image_from_path(missing)

    def test_valid_jpeg_roundtrip(self, tmp_path: Path) -> None:
        image = make_bgr_image(60, 80)
        image[:] = 200
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), image)
        loaded = load_image_from_path(path)
        assert loaded.dtype == np.uint8
        assert loaded.shape == (60, 80, 3)

    def test_corrupted_file_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupted.jpg"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Could not read"):
            load_image_from_path(bad)


# ─── resize_for_detection ─────────────────────────────────────────────────────


class TestResizeForDetection:
    def test_small_image_not_upscaled(self) -> None:
        image = make_bgr_image(100, 200)
        result = resize_for_detection(image, max_side=1280)
        # Small image should be returned unchanged (same object)
        assert result is image

    def test_exact_max_side_not_resized(self) -> None:
        image = make_bgr_image(640, 1280)
        result = resize_for_detection(image, max_side=1280)
        assert result is image

    def test_large_landscape_image_downscaled(self) -> None:
        image = make_bgr_image(720, 2560)  # longest side = 2560
        result = resize_for_detection(image, max_side=1280)
        assert result is not image
        h, w = result.shape[:2]
        assert max(h, w) == 1280
        # Aspect ratio preserved: w/h ≈ 2560/720
        assert abs(w / h - 2560 / 720) < 0.02

    def test_large_portrait_image_downscaled(self) -> None:
        image = make_bgr_image(2560, 720)  # longest side = 2560 (height)
        result = resize_for_detection(image, max_side=1280)
        h, w = result.shape[:2]
        assert max(h, w) == 1280

    def test_square_image_downscaled_correctly(self) -> None:
        image = make_bgr_image(2000, 2000)
        result = resize_for_detection(image, max_side=1000)
        assert result.shape == (1000, 1000, 3)

    def test_default_max_side_constant(self) -> None:
        assert DEFAULT_MAX_SIDE == 1280

    def test_output_is_uint8(self) -> None:
        image = make_bgr_image(1500, 2000)
        result = resize_for_detection(image, max_side=1280)
        assert result.dtype == np.uint8
