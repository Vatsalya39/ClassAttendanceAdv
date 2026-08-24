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
    face_detection_max_faces: int = Field(default=100, ge=1)
    insightface_model_pack: str = "buffalo_l"

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
