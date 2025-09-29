import pytest
from app.main import create_app

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    app = create_app()
    return TestClient(app)

def test_health_endpoint(client):
    """Test health check endpoint returns OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "mcp" in data

def test_readiness_endpoint(client):
    """Test readiness check endpoint with orchestration status."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["orchestration_ready"] is True
    assert "mcp_configured" in data

def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "capabilities" in data
    assert "orchestration" in data["capabilities"]
    assert "browser_automation" in data["capabilities"]
    assert "workflow_engine" in data["capabilities"]
    assert "langchain_agents" in data["capabilities"]
