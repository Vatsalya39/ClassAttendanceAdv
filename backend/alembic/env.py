"""
Alembic environment configuration.

Reads DATABASE_URL_SYNC from the environment (never from alembic.ini) so
real credentials are never committed to source control.

In production, DATABASE_URL_SYNC is injected by Azure Container Apps from
a Key Vault reference — no change to this file needed between environments.

Async support: migrations always run synchronously (Alembic's standard mode)
using the psycopg2 driver, even though the application uses asyncpg at
runtime. Both DSNs point at the same database.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# ── Logging ───────────────────────────────────────────────────────────────────
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from backend.core.config import settings  # noqa: E402
from backend.db import Base, models  # noqa: E402, F401 (models register tables)

target_metadata = Base.metadata


# ── DSN override from environment ─────────────────────────────────────────────
def get_url() -> str:
    url = os.environ.get("DATABASE_URL_SYNC") or settings.database_url_sync
    if not url:
        raise RuntimeError(
            "DATABASE_URL_SYNC environment variable is not set. "
            "Copy .env.example to .env and fill in the database connection string."
        )
    return url


# ── Offline migrations (generate SQL script, no live DB connection) ───────────
def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online migrations (run against live DB) ───────────────────────────────────
def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # single-use connection for migrations
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
