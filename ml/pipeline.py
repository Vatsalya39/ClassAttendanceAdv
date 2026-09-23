"""
ml/pipeline.py
──────────────
Face pipeline module — fully decoupled from the API, worker, and database.

This module may be imported in:
  - The capture worker (backend/worker/main.py) — long-lived process, warm
    model start, many batch_process calls per process lifetime.
  - Scripts (scripts/test_pipeline.py, calibrate_threshold.py,
    bench_throughput.py) — standalone CLI usage.

It must NOT import anything from backend.*  or depend on HTTP/queue/DB.

Model loading
────────────
Models are loaded once per process via _get_app() and cached in module-level
state. The pack name and execution providers come from settings (wired by
the caller — this module stays backend-free by taking them as parameters).

On first call, InsightFace downloads the model pack (~600 MB for buffalo_l)
to ~/.insightface/models/<pack>/ if not already cached.

Execution providers
──────────────────
Defaults to ["CUDAExecutionProvider", "CPUExecutionProvider"]. ONNX Runtime
silently drops providers it cannot initialise (e.g. no GPU on macOS), so
the same code runs on CPU-only dev machines without any changes.

Test-time augmentation (TTA)
────────────────────────────
When enabled, `align_and_embed`/`batch_process` embed BOTH the aligned face
crop and its horizontally flipped copy, then average the two 512-d vectors
and re-normalise. Mirror averaging is a cheap, well-established trick that
stabilises ArcFace embeddings under slight pose/occlusion variation at the
cost of one extra recognition forward pass per face.

Tiled detection (wide classroom shots)
──────────────────────────────────────
High-resolution session photos may be split into an overlapping grid of
tiles; detection runs per tile, tile-relative boxes/landmarks are translated
back to global canvas coordinates, and Non-Maximum Suppression (IoU 0.4)
merges duplicate detections from overlap zones. Faces smaller than the
quality gates survive because each tile is analysed at full detector
resolution instead of a whole-image downscale.

Types
────
BoundingBox  : (x1, y1, x2, y2) ints — pixel coordinates in the input image.
Landmarks    : list of 5 (x, y) float pairs — RetinaFace 5-point landmarks.
Embedding    : list[float] of length 512 — ArcFace embedding vector (unit-norm).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, TypedDict

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Minimum face detection confidence score (RetinaFace det_score).
# Faces below this score are discarded before embedding.
MIN_DETECTION_SCORE: float = 0.5

# Enrollment selfies are downscaled to this max side before detection —
# single close-up faces need no more pixels and decode time is saved.
# Session photos are NOT downscaled (Module 2).
ENROLLMENT_MAX_SIDE: int = 1280

# Quality gates (Module 2). Defaults; the worker passes settings values in.
MIN_FACE_SIZE_PX: int = 50
MIN_BLUR_VARIANCE: float = 50.0
MAX_POSE_RATIO: float = 0.45
MAX_YAW_DEG: float = 25.0
MAX_PITCH_DEG: float = 25.0

# Tiling / NMS defaults.
TILE_MIN_SIZE: int = 1600
TILE_OVERLAP: float = 0.2
NMS_IOU_THRESHOLD: float = 0.4

# Session-detection det_size default (Module 2).
SESSION_DET_SIZE: int = 1600


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return float(max(0, self.x2 - self.x1)) * float(max(0, self.y2 - self.y1))


@dataclass(frozen=True)
class FaceDetection:
    """Result of detecting a single face in an image."""

    bbox: BoundingBox
    landmarks: list[tuple[float, float]]  # 5-point [(x, y), ...]
    detection_score: float


@dataclass(frozen=True)
class FaceEmbedding:
    """A detected face with its ArcFace embedding."""

    bbox: BoundingBox
    embedding: list[float]  # 512-d unit vector
    # Quality-gate flags (Module 2): a face that failed a gate still flows to
    # the review queue (never silently dropped, never auto-matched), but the
    # worker skips paying the embed cost on hopeless crops when gating hard.
    quality_flags: tuple[str, ...] = ()
    detection_score: float = 0.0


@dataclass(frozen=True)
class MatchResult:
    """Output of match_embedding when a match is found."""

    student_id: str
    distance: float  # cosine distance (0 = identical, 2 = opposite)
    embedding_id: str


class CandidateDict(TypedDict):
    """Typed structure for a single roster candidate passed to match_embedding."""

    student_id: str
    embedding_id: str
    embedding: list[float]  # 512-d unit-normalised ArcFace vector


@dataclass(frozen=True)
class QualityGateConfig:
    """Thresholds for the pre-embedding face quality gates (Module 2 & Step 2)."""

    min_size_px: int = MIN_FACE_SIZE_PX
    min_blur_var: float = MIN_BLUR_VARIANCE
    max_pose_ratio: float = MAX_POSE_RATIO
    use_pnp_pose: bool = True
    max_yaw_deg: float = MAX_YAW_DEG
    max_pitch_deg: float = MAX_PITCH_DEG

    # Gates can be individually disabled (setting threshold to 0 disables).
    def enabled_size(self) -> bool:
        return self.min_size_px > 0

    def enabled_blur(self) -> bool:
        return self.min_blur_var > 0

    def enabled_pose(self) -> bool:
        if self.max_pose_ratio == 0:
            return False
        if self.use_pnp_pose:
            return self.max_yaw_deg > 0 and self.max_pitch_deg > 0
        return self.max_pose_ratio > 0


@dataclass(frozen=True)
class TiledDetectionConfig:
    """Overlapping-tile detection parameters for wide session photos.

    tile_min_size : Images whose longest side exceeds this enter tiled mode.
    overlap       : Fraction of tile size that adjacent tiles overlap (0.2 =
                   20%). Higher finds more boundary faces, costs more tiles.
    nms_iou       : IoU threshold for merging duplicate detections across
                    tile overlap zones.
    """

    tile_min_size: int = TILE_MIN_SIZE
    overlap: float = TILE_OVERLAP
    nms_iou: float = NMS_IOU_THRESHOLD


# ─── Image validation ─────────────────────────────────────────────────────────


def _validate_image(image: np.ndarray) -> None:
    """
    Raise ValueError if ``image`` is not a valid BGR uint8 array.

    Parameters
    ----------
    image : Expected to be a (H, W, 3) uint8 numpy array (BGR, as from cv2.imread).

    Raises
    ------
    ValueError
        If the array is None, wrong dtype, wrong number of channels,
        or too small to contain a face.
    """
    if image is None:
        raise ValueError("image must not be None")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be a numpy ndarray, got {type(image)}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be a (H, W, 3) BGR array, got shape {image.shape}")
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        raise ValueError(f"image is too small ({w}×{h}); minimum 32×32 pixels required")


# ─── Geometry helpers (Module 2) ─────────────────────────────────────────────


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> BoundingBox:
    """Clamp a (possibly tile-translated) bbox to the canvas."""
    return BoundingBox(
        x1=max(0, min(x1, w - 1)),
        y1=max(0, min(y1, h - 1)),
        x2=max(0 + 1, min(x2, w)),
        y2=max(0 + 1, min(y2, h)),
    )


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union between two axis-aligned boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list[FaceDetection], iou_threshold: float) -> list[FaceDetection]:
    """
    Non-Maximum Suppression across (tile-translated) detections.

    Greedy NMS by detection score; boxes overlapping more than
    ``iou_threshold`` are considered the same face and the weaker copy is
    dropped. Landmarks ride along with the surviving box.
    """
    if len(detections) <= 1:
        return list(detections)

    ordered = sorted(detections, key=lambda d: d.detection_score, reverse=True)
    kept: list[FaceDetection] = []
    for det in ordered:
        if any(_iou(det.bbox, k.bbox) > iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def compute_tile_grid(
    width: int,
    height: int,
    config: TiledDetectionConfig,
) -> list[tuple[int, int, int, int]]:
    """
    Plan the overlapping tile grid for an image (pure function, testable).

    Returns a list of (x1, y1, x2, y2) global-coordinate tiles that cover
    the whole image, each at least ~tile_min_size on its long side (exact
    when the image is smaller than one tile).

    The grid is "cols x rows" chosen by aspect ratio: wide shots get more
    horizontal tiles, tall shots more vertical ones.
    """
    if max(width, height) <= config.tile_min_size:
        return [(0, 0, width, height)]

    # Pick grid so each tile's long side is >= tile_min_size.
    cols = max(1, int(np.ceil(width / config.tile_min_size)))
    rows = max(1, int(np.ceil(height / config.tile_min_size)))
    # A 2:1 landscape photo with tile_min_size=1600 → cols=2, rows=1.

    def overlap_px(size: int) -> int:
        """Pixels of overlap for a tile edge of ``size`` px."""
        return int(size * config.overlap)

    tiles: list[tuple[int, int, int, int]] = []
    tile_w = width / cols
    tile_h = height / rows
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * tile_w) - (overlap_px(int(tile_w)) if c > 0 else 0)
            y1 = int(r * tile_h) - (overlap_px(int(tile_h)) if r > 0 else 0)
            x2 = int((c + 1) * tile_w) + (overlap_px(int(tile_w)) if c < cols - 1 else 0)
            y2 = int((r + 1) * tile_h) + (overlap_px(int(tile_h)) if r < rows - 1 else 0)
            tiles.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    return tiles


def _should_tile(image: np.ndarray, config: TiledDetectionConfig) -> bool:
    h, w = image.shape[:2]
    longest: int = max(int(h), int(w))
    return longest > config.tile_min_size


# ─── Quality gating (Module 2) ───────────────────────────────────────────────


# Generic 3D anthropometric face model in camera-aligned frame (Y down, X right, Z forward)
# Scale-proportioned to canonical ArcFace / RetinaFace 5-landmark geometry
_CANONICAL_FACE_MODEL_3D: np.ndarray = np.array(
    [
        [-17.62, -20.14, -8.0],  # left eye
        [17.62, -20.14, -8.0],  # right eye
        [0.0, 0.0, 0.0],  # nose tip (origin, forward in Z)
        [-14.59, 20.54, -6.0],  # left mouth corner
        [14.59, 20.54, -6.0],  # right mouth corner
    ],
    dtype=np.float64,
)


def estimate_head_pose_pnp(
    landmarks: list[tuple[float, float]] | np.ndarray,
    solver: str = "SQPNP",
) -> tuple[float, float, float] | None:
    """
    Estimate head pose (yaw, pitch, roll) in degrees from 5 SCRFD/RetinaFace landmarks
    using OpenCV Perspective-n-Point (PnP) solver against canonical 3D facial geometry.

    Parameters
    ----------
    landmarks : 5-point landmark coordinates [(x, y), ...] in order:
                [left_eye, right_eye, nose, mouth_l, mouth_r].
    solver : "SQPNP" (default, globally optimal, O(1)) or "EPNP".

    Returns
    -------
    (yaw, pitch, roll) in degrees, or None if landmarks are degenerate.
    Positive yaw: turning right (subject's left).
    Positive pitch: looking down.
    Positive roll: clockwise roll.
    """
    if len(landmarks) != 5:
        return None

    lmks = np.asarray(landmarks, dtype=np.float64)
    lex, ley = lmks[0]
    rex, rey = lmks[1]
    nx, ny = lmks[2]
    d_eyes = float(np.hypot(rex - lex, rey - ley))
    if d_eyes < 1e-4:
        return None

    # Focal length estimated from interocular pixel distance
    focal = d_eyes * 6.35
    cam_matrix = np.array([[focal, 0.0, nx], [0.0, focal, ny], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    flag = cv2.SOLVEPNP_SQPNP if solver == "SQPNP" else cv2.SOLVEPNP_EPNP

    ok, rvec, _ = cv2.solvePnP(_CANONICAL_FACE_MODEL_3D, lmks, cam_matrix, dist_coeffs, flags=flag)
    if not ok:
        return None

    rot_mat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    pitch = float(np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2])))
    yaw = float(np.degrees(np.arctan2(-rot_mat[2, 0], sy)))
    roll = float(np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0])))
    return yaw, pitch, roll


def assess_face_quality(
    image: np.ndarray,
    bbox: BoundingBox,
    landmarks: list[tuple[float, float]],
    config: QualityGateConfig,
) -> tuple[str, ...]:
    """
    Evaluate the size / blur / pose gates for ONE detected face.

    Returns a tuple of failure flags (""-free): subset of
    ("size", "blur", "pose"). Empty tuple ⇒ face passed every gate.

    Gates
    ─────
    size : bbox width AND height must be >= min_size_px. Tiny far-away
           faces embed unreliably and would add review noise at best.
    blur : Laplacian variance of the face crop must exceed min_blur_var.
           The Laplacian responds to high frequencies; blurry crops have
           low variance.
    pose : Perspective-n-Point 3D head pose (|yaw| > max_yaw_deg or
           |pitch| > max_pitch_deg) if enabled, otherwise fallback
           eye-to-nose aspect ratio (|d1/d2 - 1| > max_pose_ratio).
    """
    flags: list[str] = []
    h, w = image.shape[:2]
    x1, y1 = max(0, bbox.x1), max(0, bbox.y1)
    x2, y2 = min(w, bbox.x2), min(h, bbox.y2)

    if config.enabled_size() and (
        bbox.width < config.min_size_px or bbox.height < config.min_size_px
    ):
        flags.append("size")

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ("size",)  # degenerate box — treat as size failure

    if config.enabled_blur():
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < config.min_blur_var:
            flags.append("blur")

    if config.enabled_pose() and len(landmarks) == 5:
        if config.use_pnp_pose:
            pose = estimate_head_pose_pnp(landmarks)
            if pose is not None:
                yaw, pitch, _ = pose
                if abs(yaw) > config.max_yaw_deg or abs(pitch) > config.max_pitch_deg:
                    flags.append("pose")
            else:
                flags.append("pose")
        else:
            # RetinaFace landmark order: [left_eye, right_eye, nose, mouth_l, mouth_r]
            (lex, ley), (rex, rey), (nx, ny) = landmarks[0], landmarks[1], landmarks[2]
            d_left = float(np.hypot(lex - nx, ley - ny))
            d_right = float(np.hypot(rex - nx, rey - ny))
            if d_left > 1e-6 and d_right > 1e-6:
                ratio = abs(d_left / d_right - 1.0)
                if ratio > config.max_pose_ratio:
                    flags.append("pose")

    return tuple(flags)


# ─── Model singleton ─────────────────────────────────────────────────────────

_app: Any = None  # InsightFace FaceAnalysis object, loaded once per process
_app_pack: str | None = None  # pack name the cached _app was loaded with


def _get_app(
    model_pack: str = "buffalo_l",
    providers: list[str] | None = None,
    det_size: int | tuple[int, int] = (640, 640),
    force_reload: bool = False,
) -> Any:
    """
    Load the InsightFace model pack once and cache it for the process lifetime.

    Parameters
    ----------
    model_pack : InsightFace model pack name — "buffalo_l" (RetinaFace +
                 ArcFace 512-d) or "antelopev2" (same architecture, better
                 weights). The pack name is remembered so a settings change
                 reloads rather than silently reusing the old pack.
    providers  : ONNX Runtime execution providers.
                 Defaults to ["CUDAExecutionProvider", "CPUExecutionProvider"].
                 ONNX Runtime silently drops providers it cannot initialise,
                 so this list works correctly on GPU-less machines (e.g. macOS).
    det_size   : RetinaFace input size. Enrollment-style calls keep the pack
                 default (640); session photos use a larger size (default
                 1600) to catch small back-row faces. Changing det_size on an
                 already-cached app re-prepares the detector in place.
    force_reload : Drop the cache and reload (tests / pack switching).

    Returns
    -------
    Initialised insightface.app.FaceAnalysis object, or None if insightface
    is not installed (tests that don't require the model can still import
    ml.pipeline safely).
    """
    global _app, _app_pack

    try:
        import insightface
    except ImportError:
        log.warning(
            "ml.pipeline._get_app: insightface package not installed. "
            "Install ml/requirements.txt to enable real face detection."
        )
        return None

    # insightface >=1.0 requires det_size as a (w, h) pair; 0.7.x accepted a
    # bare int. Normalise once so both versions accept our settings value
    # (settings.session_det_size is an int).
    if isinstance(det_size, int):
        det_size = (det_size, det_size)

    if _app is not None and _app_pack != model_pack:
        # Pack changed (e.g. settings flipped buffalo_l → antelopev2): rebuild.
        _app = None
        _app_pack = None

    if _app is not None and not force_reload:
        # det_size change on the SAME pack → re-prepare the cached app.
        # FaceAnalysis.prepare() stores the pair on app.det_size (insightface
        # 0.7.x and 1.x); reading a stale attribute would silently keep the
        # old detector input size — a real bug when arms share one process.
        target = det_size  # already a (w, h) pair (normalised above)
        current = getattr(_app, "det_size", None)
        if current is not None and tuple(current) != tuple(target):
            _app.prepare(ctx_id=0, det_size=target)
        return _app

    effective_providers = providers or [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    log.info(
        "Loading InsightFace model pack '%s' with providers=%s, det_size=%s ...",
        model_pack,
        effective_providers,
        det_size,
    )
    t0 = time.perf_counter()
    _app = insightface.app.FaceAnalysis(
        name=model_pack,
        providers=effective_providers,
    )
    _app.prepare(ctx_id=0, det_size=det_size)
    _app_pack = model_pack
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "InsightFace model loaded: %s (%.0f ms)",
        model_pack,
        elapsed_ms,
    )
    return _app


def _configure_pipeline(
    model_pack: str | None = None,
    providers: list[str] | None = None,
    det_size: int | tuple[int, int] | None = None,
) -> Any:
    """
    Wire caller settings into _get_app (the worker passes settings values;
    scripts may omit them to accept defaults).

    Kept backend-free: ml.pipeline never imports backend.core.config — the
    worker resolves settings and hands them in.
    """
    kwargs: dict[str, Any] = {}
    if model_pack is not None:
        kwargs["model_pack"] = model_pack
    if providers is not None:
        kwargs["providers"] = providers
    if det_size is not None:
        kwargs["det_size"] = det_size
    return _get_app(**kwargs)


# ─── Embedding extraction with flip-TTA (Module 1) ──────────────────────────


def _align_face_crop(
    image: np.ndarray,
    landmarks: list[tuple[float, float]],
    rec_model: Any,
) -> np.ndarray:
    """
    Warp a face to the ArcFace 112x112 reference using its 5-point
    landmarks (version-stable: insightface.utils.face_align.norm_crop
    exists in both 0.7.x and 1.x).
    """
    from insightface.utils import face_align

    aligned: np.ndarray = face_align.norm_crop(
        image,
        landmark=np.asarray(landmarks, dtype=np.float32),
        image_size=rec_model.input_size[0],
    )
    return aligned


def apply_clahe_lab(
    aligned: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (4, 4),
    blend_weight: float = 0.18,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the L (luminance) channel in LAB color space of an aligned face crop.

    Operates strictly on the L channel in LAB color space (never RGB directly)
    to preserve chromaticity channels (A, B) and avoid skin-tone distortion.
    A blend weight preserves facial textures on well-lit frontal faces (preventing
    over-amplification of sensor noise and keeping regression <= 0.01) while
    illuminating shadows and enhancing contrast on backlit faces.
    """
    if aligned is None or aligned.size == 0:
        return aligned
    lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(lab[:, :, 0])
    if blend_weight < 1.0:
        cl = cv2.addWeighted(cl, blend_weight, lab[:, :, 0], 1.0 - blend_weight, 0)
    lab[:, :, 0] = cl
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def needs_clahe_enhancement(
    aligned: np.ndarray,
    min_mean_luminance: float = 60.0,
    max_shadow_fraction: float = 0.30,
    max_asymmetry: float = 25.0,
) -> bool:
    """
    Empirical illumination-quality assessment on an aligned 112x112 face crop.

    Decides whether an aligned crop requires CLAHE enhancement:
    - mean luminance < 60.0 (underexposed or backlit)
    - shadow fraction > 0.30 (more than 30% of face pixels under L < 40)
    - horizontal luminance asymmetry > 25.0 (severe directional shadowing)

    Well-lit frontal faces skip CLAHE completely, preserving native sensor
    high-frequency textures and preventing intra-class distance regressions.
    """
    if aligned is None or aligned.size == 0:
        return False
    lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float32)
    mean_lum = float(np.mean(lum))
    shadow_frac = float(np.mean(lum < 40))
    w = lum.shape[1]
    left_lum = float(np.mean(lum[:, : w // 2]))
    right_lum = float(np.mean(lum[:, w // 2 :]))
    asymmetry = abs(left_lum - right_lum)
    return (
        (mean_lum < min_mean_luminance)
        or (shadow_frac > max_shadow_fraction)
        or (asymmetry > max_asymmetry)
    )


def _embed_aligned_crop(
    aligned: np.ndarray,
    rec_model: Any,
    tta_flip: bool,
    apply_clahe: bool = True,
    clahe_mode: str = "global",
) -> list[float] | None:
    """
    Embed an ALREADY-ALIGNED 112x112 face crop via the recognition model's
    ONNX session (bypasses version-fragile get()/get_feat wrappers), with
    optional LAB-space CLAHE preprocessing on the L channel and optional
    horizontal-flip TTA (embed + mirror, average, L2-normalise).

    clahe_mode:
      - 'none': do not apply CLAHE.
      - 'global': apply CLAHE unconditionally to all crops.
      - 'adaptive': apply CLAHE only when needs_clahe_enhancement(aligned) is True.
    """
    if aligned is None or aligned.size == 0:
        return None

    if not apply_clahe or clahe_mode == "none":
        crop = aligned
    elif clahe_mode == "global":
        crop = apply_clahe_lab(aligned)
    elif clahe_mode == "adaptive":
        crop = apply_clahe_lab(aligned) if needs_clahe_enhancement(aligned) else aligned
    else:
        crop = aligned

    def _raw(aimg: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / rec_model.input_std,
            tuple(rec_model.input_size[:2]),
            (rec_model.input_mean,) * 3,
            swapRB=True,
        )
        out = rec_model.session.run(rec_model.output_names, {rec_model.input_name: blob})[0]
        return np.asarray(out, dtype=np.float32).ravel()

    emb = _raw(crop)
    if emb.size == 0:
        return None
    if not tta_flip:
        norm = float(np.linalg.norm(emb))
        if norm < 1e-12:
            return None
        emb = emb / norm
        return [float(v) for v in emb]

    emb_flip = _raw(cv2.flip(crop, 1))
    averaged = emb + emb_flip
    norm = float(np.linalg.norm(averaged))
    if norm < 1e-12:
        return None
    averaged /= norm
    return [float(v) for v in averaged]


def _extract_normed_embedding(face: Any, app: Any, tta_flip: bool) -> list[float]:
    """
    Return the (optionally TTA-averaged) unit-normalised ArcFace embedding.

    TTA: embed both the aligned crop and its horizontal flip
    (cv2(face_crop, 1)), average the two 512-d vectors, then L2-normalise
    the result so downstream cosine math stays consistent.

    The flip pass re-aligns the mirrored landmarks on the mirrored image
    (x' = W - x, and the L/R landmark pairs swap indices) — this is NOT the
    same as flipping the aligned crop, because ArcFace alignment is
    landmark-driven.
    """
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    if not tta_flip:
        return [float(v) for v in emb]

    rec_model = app.models["recognition"]

    # Align the same landmarks on a mirrored canvas: x' = W - x.
    flipped_img = cv2.flip(face.image, 1)
    flipped_kps = face.kps.copy()
    h_img, w_img = flipped_img.shape[:2]
    flipped_kps[:, 0] = w_img - flipped_kps[:, 0]
    # Landmark order for a mirrored face also mirrors L/R:
    # [Leye, Reye, nose, mouthL, mouthR] → [Reye, Leye, nose, mouthR, mouthL]
    flipped_kps = flipped_kps[[1, 0, 2, 4, 3]]

    flipped_aligned = _align_face_crop(
        flipped_img,
        [(float(x), float(y)) for x, y in flipped_kps],
        rec_model,
    )
    emb_flip_list = _embed_aligned_crop(flipped_aligned, rec_model, tta_flip=False)
    if emb_flip_list is None:
        # degenerate — fall back to the unflipped embedding
        return [float(v) for v in emb]

    emb_flip = np.asarray(emb_flip_list, dtype=np.float32)
    averaged = emb + emb_flip
    norm = float(np.linalg.norm(averaged))
    if norm < 1e-12:
        return [float(v) for v in emb]
    averaged /= norm
    return [float(v) for v in averaged]


# ─── Public API ───────────────────────────────────────────────────────────────


def detect_faces(
    image: np.ndarray,
    min_detection_score: float = MIN_DETECTION_SCORE,
    *,
    model_pack: str = "buffalo_l",
    providers: list[str] | None = None,
    det_size: int | tuple[int, int] | None = None,
    tiled_config: TiledDetectionConfig | None = None,
) -> list[FaceDetection]:
    """
    Detect all faces in ``image`` using RetinaFace (InsightFace).

    Parameters
    ----------
    image               : BGR uint8 numpy array (as loaded by cv2.imread).
    min_detection_score : Discard detections below this RetinaFace confidence.
                          Default 0.5. Lower → more detections, more false
                          positives. Calibrate alongside the match threshold.
    model_pack/providers : Settings wiring (Module 1).
    det_size            : RetinaFace input size; None keeps the pack default.
    tiled_config        : Enable overlapping-tile detection for wide shots
                          (Module 2). None disables tiling.

    Returns
    -------
    List of FaceDetection objects, one per detected face.
    Returns an empty list if no faces are found or insightface is not installed.

    Raises
    ------
    ValueError
        If ``image`` fails basic validation (wrong dtype / shape / too small).
    """
    _validate_image(image)

    app = _configure_pipeline(model_pack, providers, det_size)
    if app is None:
        log.debug("detect_faces: InsightFace not available, returning empty list")
        return []

    detections = _detect_on_canvas(app, image, min_detection_score)

    if tiled_config is not None and _should_tile(image, tiled_config):
        tiled_dets = _tiled_detect(app, image, min_detection_score, tiled_config)
        detections = _nms(detections + tiled_dets, tiled_config.nms_iou)

    log.debug(
        "detect_faces: %d face(s) detected (score≥%.2f)", len(detections), min_detection_score
    )
    return detections


def _detect_on_canvas(
    app: Any,
    image: np.ndarray,
    min_detection_score: float,
    offset: tuple[int, int] = (0, 0),
) -> list[FaceDetection]:
    """Run app.get() on one canvas and translate results by ``offset``."""
    raw_faces = app.get(image)
    ox, oy = offset
    detections: list[FaceDetection] = []
    for face in raw_faces:
        score = float(face.det_score)
        if score < min_detection_score:
            log.debug(
                "detect_faces: dropping face with det_score=%.3f (threshold %.3f)",
                score,
                min_detection_score,
            )
            continue
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        landmarks = [(float(pt[0]) + ox, float(pt[1]) + oy) for pt in face.kps]
        detections.append(
            FaceDetection(
                bbox=BoundingBox(x1=x1 + ox, y1=y1 + oy, x2=x2 + ox, y2=y2 + oy),
                landmarks=landmarks,
                detection_score=score,
            )
        )
    return detections


def _tiled_detect(
    app: Any,
    image: np.ndarray,
    min_detection_score: float,
    config: TiledDetectionConfig,
) -> list[FaceDetection]:
    """
    Overlapping-tile detection pass (Module 2).

    Split into grid tiles → detect per tile → translate tile-relative
    boxes/landmarks to global coordinates → NMS (IoU ~0.4) to merge
    duplicates from overlap zones.
    """
    h, w = image.shape[:2]
    tiles = compute_tile_grid(w, h, config)
    all_dets: list[FaceDetection] = []
    for tx1, ty1, tx2, ty2 in tiles:
        tile = image[ty1:ty2, tx1:tx2]
        if tile.shape[0] < 32 or tile.shape[1] < 32:
            continue
        all_dets.extend(_detect_on_canvas(app, tile, min_detection_score, offset=(tx1, ty1)))
    merged = _nms(all_dets, config.nms_iou)
    log.debug(
        "tiled detection: %d tiles → %d raw → %d after NMS@%.2f",
        len(tiles),
        len(all_dets),
        len(merged),
        config.nms_iou,
    )
    return merged


def align_and_embed(
    image: np.ndarray,
    landmarks: list[tuple[float, float]],
    *,
    tta_flip: bool = False,
    model_pack: str = "buffalo_l",
    providers: list[str] | None = None,
) -> list[float]:
    """
    Align a face crop using the 5-point landmarks and return its 512-d
    ArcFace embedding as a unit-normalised list[float].

    In the InsightFace buffalo_l pipeline, alignment and embedding happen
    inside the recognition model's pre-processing. We construct a minimal
    face object that the recognition model can process directly.

    For the normal capture flow, prefer ``batch_process`` which runs detection
    and embedding in a single ``app.get()`` call (more efficient).

    Parameters
    ----------
    image     : BGR uint8 numpy array of the full image.
    landmarks : 5-point landmarks from detect_faces() — list of 5 (x, y) floats.
    tta_flip  : Also embed the horizontally flipped face and average (Module 1).

    Returns
    -------
    List of 512 floats (unit vector, suitable for cosine similarity).
    Returns a zero vector if InsightFace is not installed.

    Raises
    ------
    ValueError
        If ``image`` fails validation or landmarks list does not have 5 points.
    """
    _validate_image(image)

    if len(landmarks) != 5:
        raise ValueError(f"landmarks must contain exactly 5 (x, y) pairs, got {len(landmarks)}")

    app = _configure_pipeline(model_pack, providers)
    if app is None:
        log.debug("align_and_embed: InsightFace not available, returning zero vector")
        return [0.0] * 512

    # InsightFace's recognition model accepts landmarks as (5, 2) float32 array.
    # We run a single app.get() on the image and find the face whose landmarks
    # are closest to the requested landmarks to extract its normed embedding.
    kps_query = np.array(landmarks, dtype=np.float32)  # (5, 2)
    raw_faces = app.get(image)

    if not raw_faces:
        log.debug("align_and_embed: no face detected in image, returning zero vector")
        return [0.0] * 512

    # Pick the face whose keypoints best match the requested landmarks
    best_face = None
    best_dist = float("inf")
    for face in raw_faces:
        dist = float(np.linalg.norm(face.kps - kps_query))
        if dist < best_dist:
            best_dist = dist
            best_face = face

    if best_face is None:
        log.debug("align_and_embed: could not extract normed embedding")
        return [0.0] * 512

    rec_model = app.models.get("recognition")
    if rec_model is not None:
        aimg = _align_face_crop(
            image,
            [(float(p[0]), float(p[1])) for p in best_face.kps],
            rec_model,
        )
        emb = _embed_aligned_crop(aimg, rec_model, tta_flip)
        if emb is not None:
            return emb

    return _extract_normed_embedding(best_face, app, tta_flip)


def batch_process(
    image: np.ndarray,
    *,
    tta_flip: bool = False,
    quality_config: QualityGateConfig | None = None,
    tiled_config: TiledDetectionConfig | None = None,
    model_pack: str = "buffalo_l",
    providers: list[str] | None = None,
    det_size: int | tuple[int, int] | None = None,
    max_faces: int | None = None,
    skip_embedding_on_quality_fail: bool = False,
    apply_clahe: bool = True,
    clahe_mode: str = "global",
) -> list[FaceEmbedding]:
    """
    Detect all faces in a classroom photo and return their embeddings.

    This is the single entry point used by the capture worker. It runs
    detection (optionally tiled for wide shots) and extracts
    ``normed_embedding`` for every face that passes the quality gates.

    Quality gating (Module 2)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Faces failing the size/blur/pose gates are NOT silently dropped:
    they come back with ``quality_flags`` set so the worker routes them to
    the review queue. When ``skip_embedding_on_quality_fail`` is True the
    embed cost is skipped and the flagging face carries a zero embedding —
    the review crop is still produced. Default False: flagged faces keep
    their embedding so the review queue's "search all students" still works.

    IMPORTANT: This function has no knowledge of the student roster or
    the database. The caller (worker) is responsible for:
      1. Calling find_nearest_embeddings_for_roster() with each embedding.
      2. Keeping the candidate set scoped to the session's section.

    Parameters
    ----------
    image : BGR uint8 numpy array (full classroom photo).
    tta_flip : Horizontal-flip TTA (Module 1).
    quality_config : Quality gates; None disables gating.
    tiled_config : Tiled detection for wide shots; None disables.
    max_faces : Hard cap from settings.face_detection_max_faces.

    Returns
    -------
    List of FaceEmbedding (one per detected face above MIN_DETECTION_SCORE).
    May be empty if no faces are detected, the image is blank, or
    InsightFace is not installed.

    Raises
    ------
    ValueError
        If ``image`` fails basic validation (wrong dtype / shape / too small).
    """
    _validate_image(image)

    app = _configure_pipeline(model_pack, providers, det_size)
    if app is None:
        log.debug("batch_process: InsightFace not available, returning empty list")
        return []

    # ── Detection (tiled for wide shots) ────────────────────────────────────
    raw_faces: list[tuple[FaceDetection, list[float]]] = []
    if tiled_config is not None and _should_tile(image, tiled_config):
        tiled_dets = _tiled_detect(app, image, MIN_DETECTION_SCORE, tiled_config)
        whole_dets = _detect_on_canvas(app, image, MIN_DETECTION_SCORE)
        detections = _nms(tiled_dets + whole_dets, tiled_config.nms_iou)
        # Recognition on tile-translated detections: align each face from its
        # GLOBAL 5-point landmarks (face_align.norm_crop is stable across
        # insightface 0.7.x and 1.x, unlike recognition.get()'s signature)
        # and embed the aligned crop via the recognition model's session.
        rec_model = app.models["recognition"]
        for det in detections:
            try:
                if det.bbox.width < 8 or det.bbox.height < 8:
                    continue
                aimg = _align_face_crop(image, det.landmarks, rec_model)
                embedding = _embed_aligned_crop(
                    aimg,
                    rec_model,
                    tta_flip,
                    apply_clahe=apply_clahe,
                    clahe_mode=clahe_mode,
                )
                if embedding is None:
                    continue
                raw_faces.append((det, embedding))
            except Exception:  # noqa: BLE001 — a bad tile crop must not kill the job
                log.debug("batch_process: tile recognition failed for bbox=%s", det.bbox)
                continue
    else:
        rec_model = app.models.get("recognition")
        for face in app.get(image):
            score = float(face.det_score)
            if score < MIN_DETECTION_SCORE:
                continue
            det = FaceDetection(
                bbox=BoundingBox(*(int(v) for v in face.bbox)),
                landmarks=[(float(p[0]), float(p[1])) for p in face.kps],
                detection_score=score,
            )
            if rec_model is not None:
                aimg = _align_face_crop(image, det.landmarks, rec_model)
                emb = _embed_aligned_crop(
                    aimg,
                    rec_model,
                    tta_flip,
                    apply_clahe=apply_clahe,
                    clahe_mode=clahe_mode,
                )
                if emb is None:
                    continue
            else:
                emb = _extract_normed_embedding(face, app, tta_flip)
            raw_faces.append((det, emb))

    # ── Quality gating + assembly ──────────────────────────────────────────
    results: list[FaceEmbedding] = []
    for det, emb in raw_faces:
        flags: tuple[str, ...] = ()
        if quality_config is not None:
            flags = assess_face_quality(image, det.bbox, det.landmarks, quality_config)
        emb_list = [float(v) for v in np.asarray(emb).ravel()]
        results.append(
            FaceEmbedding(
                bbox=det.bbox,
                embedding=emb_list,
                quality_flags=flags,
                detection_score=det.detection_score,
            )
        )
        if max_faces is not None and len(results) >= max_faces:
            log.warning(
                "batch_process: hit max_faces=%d cap — further faces ignored",
                max_faces,
            )
            break

    log.debug("batch_process: %d face(s) embedded", len(results))
    return results


def match_embedding(
    query_vector: list[float],
    candidates: list[CandidateDict],
    threshold: float,
) -> MatchResult | None:
    """
    Find the closest match for ``query_vector`` among ``candidates``.

    This function operates on an already-filtered roster-sized candidate list
    (~60-70 embeddings). It has no knowledge of the total student population
    and must never receive more than a few hundred candidates. The caller
    (find_nearest_embeddings_for_roster in the DAO layer) is responsible for
    enforcing this constraint.

    Parameters
    ----------
    query_vector : 512-d embedding of the face to identify.
    candidates   : List of dicts with keys student_id, embedding_id, embedding.
                    Each embedding is a 512-d list[float] (unit-normalised).
    threshold    : Maximum cosine distance to consider a match.
                    Calibrate with scripts/calibrate_threshold.py.
                    Faces above this threshold go to the review queue.

    Returns
    -------
    MatchResult if the closest match is within threshold, else None.
    "None" means the face goes to the review queue — NOT auto-marked absent.
    """
    if not candidates:
        return None

    query = np.array(query_vector, dtype=np.float32)
    best_dist = float("inf")
    best_candidate: CandidateDict | None = None

    for cand in candidates:
        cand_vec = np.array(cand["embedding"], dtype=np.float32)
        # Cosine distance = 1 - cosine_similarity (for unit vectors: dot product)
        dist = float(1.0 - float(np.dot(query, cand_vec)))
        if dist < best_dist:
            best_dist = dist
            best_candidate = cand

    if best_candidate is None or best_dist > threshold:
        return None

    return MatchResult(
        student_id=best_candidate["student_id"],
        distance=best_dist,
        embedding_id=best_candidate["embedding_id"],
    )
