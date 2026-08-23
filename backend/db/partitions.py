"""
backend/db/partitions.py
────────────────────────
Monthly-partition maintenance for the ``attendance`` table.

The table is PARTITION BY RANGE (session_date). Inserts fail if no partition
matches the date, so partitions must exist BEFORE the first insert of a
month. ``ensure_partitions`` pre-creates a rolling window of upcoming
partitions (default: current month + 3 ahead) plus a DEFAULT catch-all so a
late/stray date can never fail an insert.

Called from:
  - the initial Alembic migration (sync connection)
  - a scheduled maintenance job (Phase 6 — Container Apps job / pg_partman)

Idempotent: CREATE TABLE IF NOT EXISTS ... PARTITION OF is safe to re-run.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection

DEFAULT_PARTITION_NAME = "attendance_default"


def month_start(d: date) -> date:
    """First day of the month containing ``d``."""
    return d.replace(day=1)


def add_months(d: date, n: int) -> date:
    """Same day-of-month shifted by ``n`` months, clamped to the target month end."""
    month_index = d.year * 12 + (d.month - 1) + n
    year, month0 = divmod(month_index, 12)
    month = month0 + 1
    next_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last_day = (next_first - timedelta(days=1)).day
    return date(year, month, min(d.day, last_day))


def partition_name(d: date) -> str:
    return f"attendance_{d:%Y_%m}"


def _create_partition_sql(name: str, start: date, end: date) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF attendance "
        f"FOR VALUES FROM ('{start:%Y-%m-%d}') TO ('{end:%Y-%m-%d}')"
    )


def ensure_partitions(
    conn: Connection, start: date | None = None, months_ahead: int = 3
) -> list[str]:
    """
    Create the DEFAULT partition and monthly partitions for ``months_ahead``
    months starting at ``start`` (default: current month). Sync variant for
    Alembic migrations. Returns the partition names ensured.
    """
    start = month_start(start or date.today())
    names: list[str] = []

    conn.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {DEFAULT_PARTITION_NAME} "
            "PARTITION OF attendance DEFAULT"
        )
    )
    names.append(DEFAULT_PARTITION_NAME)

    for offset in range(months_ahead + 1):
        month = add_months(start, offset)
        nxt = add_months(month, 1)
        name = partition_name(month)
        conn.execute(text(_create_partition_sql(name, month, nxt)))
        names.append(name)

    return names


async def ensure_partitions_async(
    conn: AsyncConnection, start: date | None = None, months_ahead: int = 3
) -> list[str]:
    """Async variant for tests / the worker's startup check."""
    start = month_start(start or date.today())
    names: list[str] = []

    await conn.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {DEFAULT_PARTITION_NAME} "
            "PARTITION OF attendance DEFAULT"
        )
    )
    names.append(DEFAULT_PARTITION_NAME)

    for offset in range(months_ahead + 1):
        month = add_months(start, offset)
        nxt = add_months(month, 1)
        name = partition_name(month)
        await conn.execute(text(_create_partition_sql(name, month, nxt)))
        names.append(name)

    return names
