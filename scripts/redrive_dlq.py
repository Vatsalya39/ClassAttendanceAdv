#!/usr/bin/env python3
"""
scripts/redrive_dlq.py
─────────────────────
One-off operational script: receive messages from the Service Bus
capture-jobs DLQ (dead-letter sub-queue) and resubmit them to the active
queue for reprocessing.

Safe because processing is idempotent end-to-end:
  - worker: Redis SET NX claim per job_id
  - attendance: UNIQUE(student_id, session_id, session_date) ON CONFLICT
  - review-queue rows may duplicate if a job half-completed before, but the
    admin resolves/discards them from the review UI.

Usage
─────
    export SERVICE_BUS_CONNECTION_STRING=...   # RootManageSharedAccessKey
    python scripts/redrive_dlq.py --queue capture-jobs [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def redrive(queue_name: str, dry_run: bool) -> int:
    from azure.servicebus import ServiceBusMessage, ServiceBusSubQueue
    from azure.servicebus.aio import ServiceBusClient

    cs = os.environ.get("SERVICE_BUS_CONNECTION_STRING")
    if not cs:
        print("ERROR: SERVICE_BUS_CONNECTION_STRING not set", file=sys.stderr)
        return 2

    moved = 0
    async with (
        ServiceBusClient.from_connection_string(cs) as client,
        client.get_queue_receiver(
            queue_name=queue_name,
            sub_queue=ServiceBusSubQueue.DEAD_LETTER,
        ) as dlq,
    ):
        # Drain what's currently sitting in the DLQ.
        msgs = await dlq.receive_messages(max_message_count=50, max_wait_time=5)
        for msg in msgs:
            body = b"".join(msg.body)
            print(
                f"DLQ msg: id={msg.message_id} delivery={msg.delivery_count} "
                f"len={len(body)} reason={msg.dead_letter_reason}"
            )
            if not dry_run:
                async with client.get_queue_sender(queue_name) as sender:
                    # New message id so SB duplicate-window doesn't block it.
                    await sender.send_messages(
                        ServiceBusMessage(body=body, message_id=f"redrive-{msg.message_id}")
                    )
                await dlq.complete_message(msg)
                moved += 1
    print(f"{'Would move' if dry_run else 'Moved'} {moved} message(s) back to {queue_name}.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Redrive capture-jobs DLQ messages")
    p.add_argument("--queue", default="capture-jobs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    sys.exit(asyncio.run(redrive(args.queue, args.dry_run)))


if __name__ == "__main__":
    main()
