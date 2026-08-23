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
─────────────
Models are loaded once per process via _get_app() and cached in module-level
state. This is critical for the worker use-case: InsightFace buffalo_l loads
~600 MB of ONNX models; reloading per request would destroy throughput.

On first call, InsightFace will download the buffalo_l model pack (~600 MB)
to ~/.insightface/models/buffalo_l/ if not already cached.

Execution providers
───────────────────
Defaults to ["CUDAExecutionProvider", "CPUExecutionProvider"]. ONNX Runtime
silently drops providers it cannot initialise (e.g. no GPU on macOS), so
the same code runs on CPU-only dev machines without any changes.

Types
─────
BoundingBox  : (x1, y1, x2, y2) ints — pixel coordinates in the input image.
Landmarks    : list of 5 (x, y) float pairs — RetinaFace 5-point landmarks.
Embedding    : list[float] of length 512 — ArcFace embedding vector (unit-norm).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

log = logging.getLogger(__name__)

# Minimum face detection confidence score (RetinaFace det_score).
# Faces below this score are discarded before embedding.
MIN_DETECTION_SCORE: float = 0.5


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


# ─── Model singleton ─────────────────────────────────────────────────────────

_app: Any = None  # InsightFace FaceAnalysis object, loaded once per process


def _get_app(
    model_pack: str = "buffalo_l",
    providers: list[str] | None = None,
) -> Any:
    """
    Load the InsightFace model pack once and cache it for the process lifetime.

    Parameters
    ----------
    model_pack : InsightFace model pack name (e.g. "buffalo_l").
                 buffalo_l = RetinaFace detection + ArcFace 512-d recognition.
    providers  : ONNX Runtime execution providers.
                 Defaults to ["CUDAExecutionProvider", "CPUExecutionProvider"].
                 ONNX Runtime silently drops providers it cannot initialise,
                 so this list works correctly on GPU-less machines (e.g. macOS).

    Returns
    -------
    Initialised insightface.app.FaceAnalysis object, or None if insightface
    is not installed (tests that don't require the model can still import
    ml.pipeline safely).
    """
    global _app
    if _app is not None:
        return _app

    try:
        import insightface
    except ImportError:
        log.warning(
            "ml.pipeline._get_app: insightface package not installed. "
            "Install ml/requirements.txt to enable real face detection."
        )
        return None

    effective_providers = providers or [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    log.info(
        "Loading InsightFace model pack '%s' with providers=%s ...",
        model_pack,
        effective_providers,
    )
    t0 = time.perf_counter()
    _app = insightface.app.FaceAnalysis(
        name=model_pack,
        providers=effective_providers,
    )
    _app.prepare(ctx_id=0, det_size=(640, 640))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "InsightFace model loaded: %s  (%.0f ms)",
        model_pack,
        elapsed_ms,
    )
    return _app


# ─── Public API ───────────────────────────────────────────────────────────────


def detect_faces(
    image: np.ndarray,
    min_detection_score: float = MIN_DETECTION_SCORE,
) -> list[FaceDetection]:
    """
    Detect all faces in ``image`` using RetinaFace (InsightFace buffalo_l).

    Parameters
    ----------
    image               : BGR uint8 numpy array (as loaded by cv2.imread).
    min_detection_score : Discard detections below this RetinaFace confidence.
                          Default 0.5. Lower → more detections, more false
                          positives. Calibrate alongside the match threshold.

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

    app = _get_app()
    if app is None:
        log.debug("detect_faces: InsightFace not available, returning empty list")
        return []

    raw_faces = app.get(image)
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
        landmarks: list[tuple[float, float]] = [(float(pt[0]), float(pt[1])) for pt in face.kps]
        detections.append(
            FaceDetection(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                landmarks=landmarks,
                detection_score=score,
            )
        )

    log.debug(
        "detect_faces: %d face(s) detected (score≥%.2f)", len(detections), min_detection_score
    )
    return detections


def align_and_embed(image: np.ndarray, landmarks: list[tuple[float, float]]) -> list[float]:
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

    app = _get_app()
    if app is None:
        log.debug("align_and_embed: InsightFace not available, returning zero vector")
        return [0.0] * 512

    # InsightFace's recognition model accepts landmarks as (5, 2) float32 array.
    # We run a single app.get() on the image and find the face whose landmarks
    # are closest to the requested landmarks to extract its normed embedding.
    #
    # This function is primarily used by calibration scripts which already have
    # detection results. The worker uses batch_process (single-pass).
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

    if best_face is None or best_face.normed_embedding is None:
        log.debug("align_and_embed: could not extract normed embedding")
        return [0.0] * 512

    embedding: list[float] = [float(v) for v in best_face.normed_embedding]
    return embedding


def batch_process(image: np.ndarray) -> list[FaceEmbedding]:
    """
    Detect all faces in a classroom photo and return their embeddings.

    This is the single entry point used by the capture worker. It runs
    a single ``app.get()`` pass and extracts both bounding boxes and
    ``normed_embedding`` for every face — no redundant detection passes.

    IMPORTANT: This function has no knowledge of the student roster or
    the database. The caller (worker) is responsible for:
      1. Calling find_nearest_embeddings_for_roster() with each embedding.
      2. Keeping the candidate set scoped to the session's section.

    Parameters
    ----------
    image : BGR uint8 numpy array (full classroom photo).

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

    app = _get_app()
    if app is None:
        log.debug("batch_process: InsightFace not available, returning empty list")
        return []

    raw_faces = app.get(image)
    results: list[FaceEmbedding] = []

    for face in raw_faces:
        score = float(face.det_score)
        if score < MIN_DETECTION_SCORE:
            continue

        if face.normed_embedding is None:
            log.warning(
                "batch_process: face at bbox=%s has no normed_embedding (score=%.3f), skipping",
                face.bbox,
                score,
            )
            continue

        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        results.append(
            FaceEmbedding(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                embedding=face.normed_embedding.tolist(),
            )
        )

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
