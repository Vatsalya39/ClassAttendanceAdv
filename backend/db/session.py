"""
backend/db/session.py
─────────────────────
Async engine + session factory shared by the API and the worker.

Usage (FastAPI dependency):
    async def handler(session: AsyncSession = Depends(get_db)) -> ...: ...

Usage (worker):
    async with SessionFactory() as session:
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC001 — must be runtime-resolvable for Depends()
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Lazily create (and cache) the process-wide async engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency yielding a session per request.

    Takes ``request`` only to participate in the app lifespan — the engine
    itself is process-global.
    """
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def dispose_engine() -> None:
    """Called on app/worker shutdown to release the connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
