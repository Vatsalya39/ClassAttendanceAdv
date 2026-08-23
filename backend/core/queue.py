"""
backend/core/queue.py
─────────────────────
Pluggable queue abstraction for capture jobs.

Implementations
───────────────
- RedisQueueBackend   — local dev. Uses Redis LIST (LPUSH/BRPOP).
                        Requires only the Redis container from docker-compose.
- ServiceBusBackend   — production. Uses Azure Service Bus.
                        Swap by setting QUEUE_BACKEND=servicebus in env.

Both implement the QueueBackend Protocol so callers never import
transport-specific code. Wire the concrete instance in the FastAPI
lifespan or worker startup.

Usage (API side — enqueue)
--------------------------
    from backend.core.queue import make_queue_backend, CaptureJobMessage
    from backend.core.config import settings

    queue = make_queue_backend(settings)
    await queue.enqueue(CaptureJobMessage(job_id=..., session_id=..., ...))

Usage (worker side — consume)
-----------------------------
    async for message in queue.consume():
        await process(message)
        await message.ack()
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ─── Message schema ───────────────────────────────────────────────────────────


class CaptureJobMessage(BaseModel):
    """Payload enqueued when an admin triggers attendance capture."""

    job_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    section_id: uuid.UUID
    image_url: str  # Blob URL or local path, resolved by the worker
    admin_id: uuid.UUID
    enqueued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str | bytes) -> CaptureJobMessage:
        return cls.model_validate_json(data)


# Callable types for ack/nack functions
AckFn = Callable[[], Coroutine[Any, Any, None]]
NackFn = Callable[[str], Coroutine[Any, Any, None]]


class ReceivedMessage:
    """Wrapper around a received message that supports ack/nack."""

    def __init__(
        self,
        payload: CaptureJobMessage,
        _ack_fn: AckFn,
        _nack_fn: NackFn | None = None,
    ) -> None:
        self.payload = payload
        self._ack_fn = _ack_fn
        self._nack_fn = _nack_fn

    async def ack(self) -> None:
        await self._ack_fn()

    async def nack(self, reason: str = "") -> None:
        if self._nack_fn:
            await self._nack_fn(reason)


# ─── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class QueueBackend(Protocol):
    """Minimal interface for the capture job queue."""

    async def enqueue(self, message: CaptureJobMessage) -> None:
        """Push a job onto the queue."""
        ...

    def consume(self) -> AsyncIterator[ReceivedMessage]:
        """Async generator that yields messages one at a time.

        The caller MUST call ``await message.ack()`` after successful
        processing, or ``await message.nack()`` on unrecoverable failure so
        the message lands in the dead-letter queue (or is requeued by Redis).
        """
        ...

    async def close(self) -> None:
        """Release any held resources (connections, sessions)."""
        ...


# ─── Redis implementation ─────────────────────────────────────────────────────


class RedisQueueBackend:
    """
    Local-dev queue backed by a Redis LIST.

    Enqueue: LPUSH {queue_name} <json>
    Consume: BRPOP {queue_name} with a timeout, yields each message.

    There is no built-in dead-letter in Redis LIST mode. Failed messages
    (nacked) are LPUSH-ed back to a ``{queue_name}:dlq`` list so they can be
    inspected and replayed manually during development.
    """

    def __init__(self, redis_url: str, queue_name: str) -> None:
        self._redis_url = redis_url
        self._queue_name = queue_name
        self._dlq_name = f"{queue_name}:dlq"
        # Typed as Any because the redis.asyncio.Redis type is not available
        # at import time (lazy import). The actual object is redis.asyncio.Redis.
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis  # lazy import — not needed by API tier

            self._client = aioredis.from_url(self._redis_url, decode_responses=False)
        return self._client

    async def enqueue(self, message: CaptureJobMessage) -> None:
        client = await self._get_client()
        await client.lpush(self._queue_name, message.to_json())

    async def consume(self) -> AsyncIterator[ReceivedMessage]:
        client = await self._get_client()
        while True:
            # BRPOP blocks until a message is available (5-second timeout to
            # allow graceful shutdown checks in the worker loop).
            result = await client.brpop(self._queue_name, timeout=5)
            if result is None:
                continue  # timeout — loop again (worker can check stop flag)

            _, raw = result
            payload = CaptureJobMessage.from_json(raw)

            # Capture loop variables explicitly in default args to avoid B023.
            async def ack() -> None:
                pass  # Redis LIST — message is already removed by BRPOP

            async def nack(reason: str = "", _raw: bytes = raw) -> None:
                await client.lpush(self._dlq_name, _raw)

            yield ReceivedMessage(payload=payload, _ack_fn=ack, _nack_fn=nack)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ─── Azure Service Bus implementation ─────────────────────────────────────────


class ServiceBusQueueBackend:
    """
    Production queue backed by Azure Service Bus.

    The connection string is read from settings; in production this value
    is a Key Vault reference injected by Container Apps, not a plaintext env.

    Dead-lettered messages (nacked) go to the Service Bus DLQ automatically.
    KEDA reads the active message count on this queue to scale the worker tier.
    """

    def __init__(self, connection_string: str, queue_name: str) -> None:
        self._connection_string = connection_string
        self._queue_name = queue_name

    async def enqueue(self, message: CaptureJobMessage) -> None:
        from azure.servicebus import ServiceBusMessage
        from azure.servicebus.aio import ServiceBusClient

        async with (
            ServiceBusClient.from_connection_string(self._connection_string) as client,
            client.get_queue_sender(self._queue_name) as sender,
        ):
            sb_msg = ServiceBusMessage(
                body=message.to_json(),
                message_id=str(message.job_id),
            )
            await sender.send_messages(sb_msg)

    async def consume(self) -> AsyncIterator[ReceivedMessage]:
        from azure.servicebus.aio import ServiceBusClient

        async with (
            ServiceBusClient.from_connection_string(self._connection_string) as client,
            client.get_queue_receiver(
                queue_name=self._queue_name,
                max_wait_time=5,
            ) as receiver,
        ):
            async for sb_msg in receiver:
                raw = b"".join(sb_msg.body)
                payload = CaptureJobMessage.from_json(raw)

                # Capture loop variables in default args to avoid B023.
                async def ack(_msg: Any = sb_msg) -> None:
                    await receiver.complete_message(_msg)

                async def nack(reason: str = "", _msg: Any = sb_msg) -> None:
                    await receiver.dead_letter_message(
                        _msg,
                        reason="ProcessingFailed",
                        error_description=reason[:4096],
                    )

                yield ReceivedMessage(payload=payload, _ack_fn=ack, _nack_fn=nack)

    async def close(self) -> None:
        pass  # ServiceBusClient is used as a context manager per-call above


# ─── Factory ──────────────────────────────────────────────────────────────────


def make_queue_backend(settings: Any) -> QueueBackend:
    """
    Instantiate the correct QueueBackend based on QUEUE_BACKEND env var.

    Import this in the FastAPI lifespan and in the worker startup — do not
    construct backends directly in route handlers.

    ``settings`` is typed as Any to avoid a circular import; the actual type
    is ``backend.core.config.Settings``.
    """
    from backend.core.config import QueueBackendType

    if settings.queue_backend == QueueBackendType.redis:
        return RedisQueueBackend(
            redis_url=settings.redis_url,
            queue_name=settings.queue_name,
        )
    elif settings.queue_backend == QueueBackendType.servicebus:
        if not settings.service_bus_connection_string:
            raise RuntimeError(
                "QUEUE_BACKEND=servicebus but SERVICE_BUS_CONNECTION_STRING is not set."
            )
        return ServiceBusQueueBackend(
            connection_string=settings.service_bus_connection_string,
            queue_name=settings.queue_name,
        )
    else:
        raise ValueError(f"Unknown queue backend: {settings.queue_backend}")
