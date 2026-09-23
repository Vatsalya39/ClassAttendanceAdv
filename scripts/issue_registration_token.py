#!/usr/bin/env python3
"""
scripts/issue_registration_token.py
───────────────────────────────────
CLI utility for instructors and administrators to issue single-use student
registration tokens for onboarding.

Usage:
    poetry run python scripts/issue_registration_token.py --roll 25BLC1105
    poetry run python scripts/issue_registration_token.py --roll CSE3B042 --days 60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from backend.db import dao  # noqa: E402
from backend.db.models import Institution  # noqa: E402
from backend.db.session import get_session_factory  # noqa: E402


async def issue(roll_number: str, expires_in_days: int = 30) -> None:
    factory = get_session_factory()
    async with factory() as session:
        # Fetch the default/first institution
        inst = (await session.execute(select(Institution))).scalars().first()
        if inst is None:
            print(
                "Error: No institution found in database. Create one first via Admin Console.",
                file=sys.stderr,
            )
            sys.exit(1)

        token_row = await dao.create_registration_token(
            session,
            roll_number=roll_number.strip(),
            institution_id=inst.institution_id,
            expires_in_days=expires_in_days,
        )
        await session.commit()

        print("\n" + "═" * 60)
        print("  STUDENT REGISTRATION TOKEN ISSUED")
        print("═" * 60)
        print(f"  Roll Number : {token_row.roll_number}")
        print(f"  Institution : {inst.name} ({inst.institution_id})")
        print(f"  Expires At  : {token_row.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("  Single-Use  : Yes")
        print("─" * 60)
        print(f"  TOKEN       : {token_row.token}")
        print("═" * 60)
        print("\nShare this token with the student. They will enter it at:")
        print("http://localhost:3000/register\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue a student registration token")
    parser.add_argument("--roll", required=True, help="Student roll number (e.g. 25BLC1105)")
    parser.add_argument("--days", type=int, default=30, help="Token validity in days (default: 30)")
    args = parser.parse_args()

    asyncio.run(issue(args.roll, args.days))


if __name__ == "__main__":
    main()
