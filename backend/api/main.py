"""
backend/api/main.py
───────────────────
FastAPI application entry point.

Route modules (Phase 3):
    auth       — student/admin login + refresh
    register   — student enrollment (runs the face pipeline inline)
    attendance — student attendance history
    admin      — async capture, review queue, reports
    org        — institutions/departments/sections CRUD, bulk-enroll, admins

Run locally:
    uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── Startup ───────────────────────────────────────────────────────────────
    import redis.asyncio as aioredis

    from backend.core.queue import make_queue_backend
    from backend.core.storage import make_storage_backend
    from backend.db.session import dispose_engine

    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    app.state.queue = make_queue_backend(settings)
    app.state.storage = make_storage_backend(settings)

    if settings.storage_backend.value == "local":
        Path(settings.storage_local_root).mkdir(parents=True, exist_ok=True)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await app.state.redis.aclose()
    await app.state.queue.close()
    await app.state.storage.close()
    await dispose_engine()


app = FastAPI(
    title="Attendance System API",
    description=(
        "Multi-campus face-recognition attendance system. "
        "Face matching is ALWAYS scoped to a section roster (~60-70 students), "
        "never a global search."
    ),
    version="0.3.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security headers ──────────────────────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # The frontend needs camera access; nothing else.
    response.headers["Permissions-Policy"] = "camera=(self), geolocation=(), microphone=()"
    if settings.is_production:
        # API serves JSON + static images only — lock navigation down hard.
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; img-src 'self'; frame-ancestors 'none'"
        )
    return response


# ── Static files (local-storage enrollment/session images) ────────────────────
if settings.storage_backend.value == "local":
    # Directory must exist before mounting (also created in the lifespan for
    # runtime-created paths; tests import this module outside a lifespan).
    Path(settings.storage_local_root).mkdir(parents=True, exist_ok=True)
    app.mount(
        "/static",
        StaticFiles(directory=settings.storage_local_root),
        name="static",
    )

# ── Route registration ────────────────────────────────────────────────────────
from backend.api.routes import (  # noqa: E402
    admin,
    attendance,
    auth,
    org,
    public,
    register,
)

app.include_router(auth.router)
app.include_router(register.router)
app.include_router(attendance.router)
app.include_router(admin.router)
app.include_router(org.router)
app.include_router(public.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    """
    Liveness probe used by Container Apps and docker-compose healthchecks.
    Returns 200 as long as the process is running.
    """
    return {"status": "ok", "environment": settings.environment.value}
