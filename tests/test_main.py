"""Tests for the Agent service main application."""

import pytest
from fastapi import FastAPI
from unittest.mock import patch

from app.main import create_app
from app.config import AgentSettings


class TestAgentApp:
    """Test the Agent service FastAPI application."""

    def test_creates_fastapi_instance(self):
        """Test that create_app returns a FastAPI instance."""
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_health_endpoint_returns_correct_service_name(self, client, test_settings):
        """Test health endpoint returns agent service name."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == test_settings.app_name

    def test_app_with_ai_service_configuration(self, test_settings):
        """Test that app can be configured with AI service settings."""
        assert test_settings.openai_api_key == "test-openai-key"
        assert test_settings.anthropic_api_key == "test-anthropic-key"

        app = create_app(test_settings)
        assert app.title == test_settings.app_name

    @pytest.mark.ai
    def test_ai_integration_configuration(self, test_settings):
        """Test that AI service integration is properly configured."""
        # This would test AI client initialization in a real implementation
        app = create_app(test_settings)
        assert isinstance(app, FastAPI)

    def test_cors_configuration(self, client):
        """Test CORS configuration allows requests."""
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })

        # Should handle preflight request
        assert response.status_code in [200, 204]

    def test_error_handling(self, client):
        """Test that the app handles errors gracefully."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_health_endpoint_concurrent_access(self, client):
        """Test health endpoint under concurrent access."""
        import threading
        import time

        results = []
        errors = []

        def make_request():
            try:
                response = client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_request) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert all(status == 200 for status in results)

    @pytest.mark.performance
    def test_health_endpoint_performance(self, client):
        """Test health endpoint response time."""
        import time

        times = []
        for _ in range(10):
            start = time.perf_counter()
            response = client.get("/health")
            end = time.perf_counter()

            assert response.status_code == 200
            times.append(end - start)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.05  # Less than 50ms average