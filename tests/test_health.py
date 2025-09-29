"""Health endpoint validation for the Agent service."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_agent_status() -> None:
    """/health should surface the service identifier and MCP metadata."""

    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "plasma-engine-agent"
    assert payload["status"] == "ok"

