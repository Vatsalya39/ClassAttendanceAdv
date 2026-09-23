"""
tests/conftest.py
─────────────────
Shared pytest configuration for all phases.

IMPORTANT: Settings() is a module-level singleton instantiated at import
time. We must set required env vars BEFORE any backend module is imported.
We do this at module level here (not inside a fixture) so it runs first.

Test databases:
  - DB-backed tests use a dedicated ``attendance_test`` database (bootstrapped
    automatically from the docker-compose Postgres) so tests NEVER touch the
    app database. Override with TEST_DATABASE_URL for CI.
  - Fixtures skip (not fail) when Postgres / Redis are unreachable so the
    unit suite runs anywhere; start ``docker compose up postgres redis`` for
    full coverage.
"""

from __future__ import annotations

import asyncio
import os

# ── Set required env vars before any backend import ──────────────────────────
_TEST_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/test_db",
    "DATABASE_URL_SYNC": "postgresql+psycopg2://test:test@localhost:5432/test_db",
    "JWT_SECRET_KEY": "test-secret-key-that-is-at-least-32-characters-long",
    "REDIS_URL": "redis://localhost:6379/0",
    "QUEUE_BACKEND": "redis",
    "STORAGE_BACKEND": "local",
    "STORAGE_LOCAL_ROOT": "/tmp/attendance_test_uploads",
    "ENVIRONMENT": "development",
    "LOG_LEVEL": "WARNING",
}
for _k, _v in _TEST_ENV.items():
    os.environ.setdefault(_k, _v)

from typing import TYPE_CHECKING, Any  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_COMPOSE_DB_URL = "postgresql+asyncpg://attendance:attendance@localhost:5432/attendance_db"
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/0")

# Tables in dependency order for TRUNCATE ... CASCADE
_ALL_TABLES = (
    "registration_tokens, audit_log, review_queue, attendance, class_sessions, enrollments, "
    "face_embeddings, students, sections, departments, institutions, admins"
)


async def _bootstrap_test_database() -> str | None:
    """
    Return the test-database URL, creating the dedicated attendance_test DB
    if we're using the local docker-compose Postgres. None ⇒ unreachable.
    """
    if TEST_DATABASE_URL:
        url = TEST_DATABASE_URL
    else:
        admin = create_async_engine(_COMPOSE_DB_URL, isolation_level="AUTOCOMMIT")
        try:
            async with admin.connect() as conn:
                await conn.execute(text("SELECT 1"))
                exists = await conn.scalar(
                    text("SELECT 1 FROM pg_database WHERE datname = 'attendance_test'")
                )
                if not exists:
                    await conn.execute(text("CREATE DATABASE attendance_test"))
        except Exception:
            await admin.dispose()
            return None
        finally:
            await admin.dispose()
        url = "postgresql+asyncpg://attendance:attendance@localhost:5432/attendance_test"

    probe = create_async_engine(url)
    try:
        async with probe.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return None
    finally:
        await probe.dispose()
    return url


@pytest.fixture(scope="session")
def db_engine() -> Any:
    """
    Session-scoped async engine with the full schema freshly created.

    Deliberately a *sync* generator fixture: pytest-asyncio gives each test
    its own event loop, and pooled asyncpg connections are loop-bound. The
    engine itself is loop-agnostic; schema setup runs via asyncio.run() and
    NullPool guarantees no stale connection ever crosses a test's loop.
    """
    url = asyncio.run(_bootstrap_test_database())
    if url is None:
        pytest.skip("Postgres for tests not reachable — run: docker compose up -d postgres")

    from backend.db import Base
    from backend.db.partitions import ensure_partitions

    engine = create_async_engine(url, poolclass=NullPool)

    # Reset schema deterministically (partitions & enum types included).
    async def _create_schema() -> None:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text(f"DROP TABLE IF EXISTS {_ALL_TABLES} CASCADE"))
            await conn.execute(text("DROP TYPE IF EXISTS admin_role CASCADE"))
            await conn.execute(text("DROP TYPE IF EXISTS review_status CASCADE"))
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(lambda sync_conn: ensure_partitions(sync_conn))

    asyncio.run(_create_schema())

    yield engine

    asyncio.run(engine.dispose())


@pytest.fixture
async def db_session(db_engine: Any) -> AsyncIterator[AsyncSession]:
    """Function-scoped session; truncates every table before the test."""
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        await session.execute(text(f"TRUNCATE TABLE {_ALL_TABLES} CASCADE"))
        await session.commit()
        yield session


@pytest.fixture
async def redis_client() -> AsyncIterator[Any]:
    """Function-scoped Redis client on a flushed db; skips if unreachable."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except Exception:
        await client.close()
        pytest.skip("Redis not reachable — run: docker compose up -d redis")
        yield  # pragma: no cover (unreachable after skip)
        return
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()  # type: ignore[attr-defined]  (types-redis lags redis-py 5)


# ── Shared API test fixtures (Phase 3/4) ─────────────────────────────────────


class FakeQueue:
    """In-memory queue double capturing enqueued messages."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def enqueue(self, message: Any) -> None:
        self.messages.append(message)

    def consume(self) -> AsyncIterator[Any]:  # pragma: no cover — unused
        raise NotImplementedError

    async def close(self) -> None:
        pass


@pytest.fixture
def fake_queue() -> FakeQueue:
    return FakeQueue()


@pytest.fixture
def tmp_storage(tmp_path: Any) -> Any:
    from backend.core.storage import LocalStorageBackend

    return LocalStorageBackend(root_path=str(tmp_path / "uploads"))


@pytest.fixture
async def client(
    db_session: Any,
    redis_client: Any,
    fake_queue: FakeQueue,
    tmp_storage: Any,
) -> AsyncIterator[Any]:
    """
    httpx AsyncClient over ASGITransport so request handlers execute on the
    SAME event loop as the fixtures (loop-bound DB/Redis sessions stay valid).
    """
    from httpx import ASGITransport, AsyncClient

    from backend.api.deps import get_queue, get_redis, get_storage
    from backend.api.main import app
    from backend.db.session import get_db

    async def _override_db() -> AsyncIterator[Any]:
        yield db_session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[get_queue] = lambda: fake_queue
    app.dependency_overrides[get_storage] = lambda: tmp_storage
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
