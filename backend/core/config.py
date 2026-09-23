"""
backend/core/config.py
─────────────────────
Centralised settings, loaded once at import time via Pydantic Settings.
All configuration is read from environment variables (or a .env file in
local dev). In production, secrets are injected by Azure Container Apps from
Key Vault references — no code change required.

Usage
-----
    from backend.core.config import settings

    db_url = settings.database_url
"""

from __future__ import annotations

import json as _json
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_cors(v: object) -> list[str]:
    """Accept JSON array, comma-separated string, or already-a-list."""
    if isinstance(v, list):
        return [str(o).strip() for o in v]
    if isinstance(v, str):
        raw = v.strip()
        if raw.startswith("["):
            return [str(o).strip() for o in _json.loads(raw)]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    raise TypeError(f"Cannot parse cors_origins from {type(v)}: {v!r}")


class QueueBackendType(str, Enum):
    redis = "redis"
    servicebus = "servicebus"


class StorageBackendType(str, Enum):
    local = "local"
    azure_blob = "azure_blob"


class Environment(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_parse_none_str="",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        ...,
        description="Async SQLAlchemy DSN (postgresql+asyncpg://...)",
    )
    database_url_sync: str = Field(
        ...,
        description="Sync DSN for Alembic (postgresql+psycopg2://...)",
    )
    # asyncpg does NOT understand ?sslmode= in the URL (psycopg-only) — TLS
    # for Azure Flexible Server is enabled via this flag instead (session.py).
    db_ssl: bool = False
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=20, ge=0)

    # ── Uploads ───────────────────────────────────────────────────────────────
    upload_max_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    enrollment_min_images: int = Field(default=3, ge=1)
    enrollment_max_images: int = Field(default=5, ge=1)
    require_registration_token: bool = Field(
        default=True,
        description="Enforce admin-issued registration tokens on /register.",
    )

    # ── Job status (Redis) ────────────────────────────────────────────────────
    job_status_ttl_seconds: int = Field(default=3600, ge=60)

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Queue ─────────────────────────────────────────────────────────────────
    queue_backend: QueueBackendType = QueueBackendType.redis
    queue_name: str = "capture-jobs"
    service_bus_connection_string: str | None = None
    service_bus_namespace: str | None = None

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_backend: StorageBackendType = StorageBackendType.local
    storage_local_root: str = "/app/data/uploads"
    azure_storage_connection_string: str | None = None
    azure_blob_container_enrollment: str = "enrollment-images"
    azure_blob_container_sessions: str = "session-images"

    # ── Azure Key Vault ───────────────────────────────────────────────────────
    azure_keyvault_uri: str | None = None

    # ── Face pipeline ─────────────────────────────────────────────────────────
    face_match_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=2.0,
        description=(
            "Cosine distance threshold (0-2 scale). "
            "Calibrate with scripts/calibrate_threshold.py."
        ),
    )
    # Minimum gap between the best and second-best DISTINCT-student distance.
    # Ambiguous pairs (twins, lookalikes) fall below it and go to review.
    face_match_margin: float = Field(
        default=0.08,
        ge=0.0,
        le=2.0,
        description=(
            "Required distance gap between best and runner-up student. "
            "Smaller margins = fewer review items but more lookalike confusion."
        ),
    )
    # How many nearest embeddings to fetch per face (multi-embedding scoring
    # groups these by student and averages their top-2 distances).
    face_match_top_k: int = Field(default=10, ge=1, le=50)
    # Roster-search scope per face — the 1-to-1 assignment builds its cost
    # matrix from these candidate students.
    face_candidate_students: int = Field(default=5, ge=1, le=25)
    face_detection_max_faces: int = Field(default=100, ge=1)
    insightface_model_pack: str = Field(
        default="buffalo_l",
        description="InsightFace pack: buffalo_l or antelopev2 (better accuracy, larger download).",
    )
    dual_enrollment_write: bool = Field(
        default=True,
        description=(
            "During model evaluation/migration, write embeddings for both buffalo_l "
            "and antelopev2 so rollbacks leave all enrolled students matchable."
        ),
    )
    # ONNX Runtime execution providers, comma-separated. Unavailable
    # providers are silently dropped by ORT, so the GPU-first default also
    # works on CPU-only machines.
    onnx_providers: str = "CUDAExecutionProvider,CPUExecutionProvider"
    # Horizontal-flip test-time augmentation: embed face + its mirror and
    # average — free accuracy gain on blur/occlusion at 2x embed cost.
    face_tta_flip: bool = Field(default=True)
    # ── Session detection tuning (Module 2) ──────────────────────────────────
    # RetinaFace input size for session photos — 1600 catches back-row faces
    # that 640 misses (enrollment selfies stay at the pack default 640).
    session_det_size: int = Field(default=1600, ge=320, le=4096)
    # Enable overlapping-tile detection for wide classroom shots.
    session_tiled_detection: bool = Field(default=True)
    # Tile grid as "cols x rows" auto-chosen by aspect ratio at this min size.
    session_tile_min_size: int = Field(default=1600, ge=640, le=8192)
    # Overlap fraction between adjacent tiles (merges detections cut at
    # tile boundaries; NMS below dedupes the overlap zones).
    session_tile_overlap: float = Field(default=0.2, ge=0.0, le=0.5)
    # IoU threshold for cross-tile non-maximum suppression.
    face_nms_iou: float = Field(default=0.4, ge=0.0, le=1.0)
    # ── Face quality gates (Module 2) ────────────────────────────────────────
    face_min_size_px: int = Field(
        default=50, ge=8, le=512, description="Min face bbox width AND height in pixels."
    )
    face_min_blur_var: float = Field(
        default=10.0,
        ge=0.0,
        description="Laplacian-variance blur gate; lower = blurrier.",
    )
    # Max |eye-distance ratio - 1| — legacy 2D heuristic fallback.
    face_max_pose_ratio: float = Field(
        default=0.45,
        ge=0.0,
        le=2.0,
        description="Pose gate: max asymmetry between eye-to-nose distances.",
    )
    # Step 2: Perspective-n-Point 3D head pose quality gate.
    face_use_pnp_pose: bool = Field(
        default=True,
        description="Enable Perspective-n-Point 3D head pose quality gate.",
    )
    face_max_yaw_deg: float = Field(
        default=25.0,
        ge=0.0,
        le=90.0,
        description="Max head yaw in degrees before quality gate flags as 'pose'.",
    )
    face_max_pitch_deg: float = Field(
        default=25.0,
        ge=0.0,
        le=90.0,
        description="Max head pitch in degrees before quality gate flags as 'pose'.",
    )
    # ── Session capture ────────────────────────────────────────────────────────
    # Multi-frame burst capture: how many frames one capture may include.
    capture_max_frames: int = Field(default=5, ge=1, le=10)
    capture_min_frames: int = Field(default=1, ge=1, le=10)
    # Step 4: Temporal fusion across burst frames using IoU tracking
    burst_temporal_fusion: bool = Field(
        default=True,
        description="Enable Step 4 IoU-based tracking and embedding fusion across burst frames.",
    )
    burst_iou_threshold: float = Field(
        default=0.50,
        ge=0.1,
        le=1.0,
        description="IoU threshold for associating face detections across burst frames.",
    )
    # Minimum accepted image side (pixels) for SESSION capture photos.
    # Back-row faces in a 60-70-seat classroom are ~50-100px at 1080p+;
    # below ~1024px they embed unreliably and flood the review queue (the
    # classroomFour vs classroomFourHigh experiment: 54 flagged @1024px vs
    # 0 flagged @2814px for the same scene). Set 0 to disable the floor.
    capture_min_image_side: int = Field(
        default=1024,
        ge=0,
        le=16384,
        description="Reject classroom photos whose shorter side is below this many pixels.",
    )

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Comma-separated origins (JSON array also accepted). Kept as a plain str:
    # pydantic-settings JSON-decodes list-typed env values BEFORE BeforeValidator
    # runs, which crashed on plain "a,b" values from Container Apps.
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return _parse_cors(self.cors_origins)

    # ── Worker ────────────────────────────────────────────────────────────────
    worker_concurrency: int = Field(default=4, ge=1)

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_auth_rpm: int = 10
    rate_limit_capture_rpm: int = 5
    rate_limit_register_rpm: int = 5

    # ── General ───────────────────────────────────────────────────────────────
    environment: Environment = Environment.development
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.production

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.development


# Singleton — imported by all modules that need configuration.
settings = Settings()
