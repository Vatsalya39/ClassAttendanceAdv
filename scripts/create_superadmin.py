"""
scripts/create_superadmin.py
────────────────────────────
Bootstrap the FIRST super_admin account (org CRUD endpoints require an
authenticated admin, so the initial account must be created out-of-band).

Usage:
    poetry run python scripts/create_superadmin.py --username admin --password <min-8-chars>

Idempotent: exits cleanly if the username already exists (role untouched).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, ".")

from backend.core.security import hash_password  # noqa: E402
from backend.db import dao  # noqa: E402
from backend.db.models import AdminRole  # noqa: E402
from backend.db.session import get_session_factory  # noqa: E402


async def create(username: str, password: str) -> None:
    factory = get_session_factory()
    async with factory() as session:
        existing = await dao.get_admin_by_username(session, username)
        if existing is not None:
            print(f"Admin '{username}' already exists (role={existing.role.value}) — nothing to do.")
            return
        await dao.create_admin(
            session,
            username=username,
            password_hash=hash_password(password),
            role=AdminRole.super_admin,
            institution_id=None,
            department_id=None,
        )
        await session.commit()
        print(f"Super admin '{username}' created.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the first super admin")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True, help="minimum 8 characters")
    args = parser.parse_args()

    if len(args.password) < 8:
        parser.error("password must be at least 8 characters")

    asyncio.run(create(args.username, args.password))


if __name__ == "__main__":
    main()
