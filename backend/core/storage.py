"""
backend/core/storage.py
───────────────────────
Pluggable storage abstraction for images (enrollment photos + session captures).

Implementations
───────────────
- LocalStorageBackend   — local dev. Saves files to STORAGE_LOCAL_ROOT on disk.
- AzureBlobStorageBackend — production. Uploads to Azure Blob Storage.

Phase 3 wires the concrete instance via make_storage_backend(); Phase 6
switches to AzureBlobStorageBackend by setting STORAGE_BACKEND=azure_blob.

Usage
-----
    from backend.core.storage import make_storage_backend
    from backend.core.config import settings

    storage = make_storage_backend(settings)
    url = await storage.upload(
        data=file_bytes,
        filename="session_abc.jpg",
        container="session-images",
    )
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ─── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal interface for blob / file storage."""

    async def upload(
        self,
        data: bytes,
        filename: str,
        container: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """
        Upload ``data`` and return its public (or SAS-signed) URL.

        Parameters
        ----------
        data        : Raw file bytes.
        filename    : Desired filename/blob name. Implementations may prefix
                      with a UUID to ensure uniqueness.
        container   : Logical container name (maps to Blob container in Azure,
                      or a subdirectory under STORAGE_LOCAL_ROOT locally).
        content_type: MIME type of the content.

        Returns
        -------
        str : URL usable to retrieve the file later (may be a local path for
              dev or an Azure Blob HTTPS URL for production).
        """
        ...

    async def delete(self, url: str) -> None:
        """Delete the file at the given URL/path."""
        ...

    async def download(self, url: str) -> bytes:
        """Fetch the file contents (used by the capture worker)."""
        ...

    async def close(self) -> None:
        """Release held resources."""
        ...


# ─── Local filesystem implementation ─────────────────────────────────────────


class LocalStorageBackend:
    """
    Stores files under ``root_path/{container}/{uuid}_{filename}``.

    In docker-compose the root is mounted as a volume so files survive
    container restarts. The returned URL is a relative path that the API
    serves via a static-files mount (configured in Phase 3).

    NOTE: Not suitable for multi-replica deployments (files are node-local).
    Switch to AzureBlobStorageBackend before scaling beyond one API replica.
    """

    def __init__(self, root_path: str) -> None:
        self._root = Path(root_path)

    async def upload(
        self,
        data: bytes,
        filename: str,
        container: str,
        content_type: str = "image/jpeg",
    ) -> str:
        dest_dir = self._root / container
        dest_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        dest = dest_dir / unique_name
        dest.write_bytes(data)
        # Return a URL-like path that the FastAPI static-files route can serve.
        return f"/static/{container}/{unique_name}"

    async def delete(self, url: str) -> None:
        # url looks like /static/{container}/{filename} — strip the leading prefix.
        rel_path = url.removeprefix("/static/")
        target = self._root / rel_path
        if target.exists():
            target.unlink()

    async def download(self, url: str) -> bytes:
        rel_path = url.removeprefix("/static/")
        target = self._root / rel_path
        return target.read_bytes()

    async def close(self) -> None:
        pass  # Nothing to release for local FS


# ─── Azure Blob Storage implementation ───────────────────────────────────────


class AzureBlobStorageBackend:
    """
    Uploads to Azure Blob Storage, returning the public HTTPS blob URL.

    In production the connection string is a Key Vault reference injected
    by Container Apps — not a plaintext environment variable.

    Biometric data retention: face images must be deletable on request.
    The ``delete`` method permanently removes both the blob and the embedding
    record (embedding deletion is handled by the caller / DAO layer).
    """

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string

    async def upload(
        self,
        data: bytes,
        filename: str,
        container: str,
        content_type: str = "image/jpeg",
    ) -> str:
        from azure.storage.blob import ContentSettings
        from azure.storage.blob.aio import BlobServiceClient

        unique_name = f"{uuid.uuid4().hex}_{filename}"
        async with BlobServiceClient.from_connection_string(self._connection_string) as client:
            blob_client = client.get_blob_client(container=container, blob=unique_name)
            await blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
            return str(blob_client.url)

    async def delete(self, url: str) -> None:
        from azure.storage.blob.aio import BlobServiceClient

        # Parse container and blob name from the URL.
        # URL format: https://{account}.blob.core.windows.net/{container}/{blob}
        parts = url.split("/")
        container = parts[-2]
        blob_name = parts[-1]
        async with BlobServiceClient.from_connection_string(self._connection_string) as client:
            blob_client = client.get_blob_client(container=container, blob=blob_name)
            await blob_client.delete_blob(delete_snapshots="include")

    async def download(self, url: str) -> bytes:
        from azure.storage.blob.aio import BlobServiceClient

        parts = url.split("/")
        container = parts[-2]
        blob_name = parts[-1]
        async with BlobServiceClient.from_connection_string(self._connection_string) as client:
            blob_client = client.get_blob_client(container=container, blob=blob_name)
            downloader = await blob_client.download_blob()
            return await downloader.readall()

    async def close(self) -> None:
        pass  # Context manager used per-call


# ─── Factory ──────────────────────────────────────────────────────────────────


def make_storage_backend(settings: Any) -> StorageBackend:
    """
    Instantiate the correct StorageBackend based on STORAGE_BACKEND env var.

    Phase 3: import and call this in the FastAPI lifespan.
    Phase 6: switch STORAGE_BACKEND=azure_blob in the Container Apps env.
    """
    from backend.core.config import StorageBackendType

    if settings.storage_backend == StorageBackendType.local:
        return LocalStorageBackend(root_path=settings.storage_local_root)
    elif settings.storage_backend == StorageBackendType.azure_blob:
        if not settings.azure_storage_connection_string:
            raise RuntimeError(
                "STORAGE_BACKEND=azure_blob but AZURE_STORAGE_CONNECTION_STRING is not set."
            )
        return AzureBlobStorageBackend(connection_string=settings.azure_storage_connection_string)
    else:
        raise ValueError(f"Unknown storage backend: {settings.storage_backend}")
