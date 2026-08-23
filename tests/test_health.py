"""
tests/test_health.py
────────────────────
Phase 0 smoke test — confirms the FastAPI app starts and /health returns 200.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_status_ok(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_has_environment_field(client: TestClient) -> None:
    data = client.get("/health").json()
    assert "environment" in data
