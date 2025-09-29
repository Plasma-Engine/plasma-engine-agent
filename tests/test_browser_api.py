"""Integration tests for browser automation API endpoints."""

import base64
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestNavigateEndpoint:
    """Test /api/v1/agent/browser/navigate endpoint."""

    def test_navigate_success(self, client):
        """Test successful navigation."""
        with patch("app.routers.browser.BrowserManager") as MockManager:
            mock_manager = AsyncMock()
            mock_page = AsyncMock()
            mock_page.url = "https://example.com"
            mock_page.goto = AsyncMock()
            mock_page.title = AsyncMock(return_value="Example Domain")
            mock_page.content = AsyncMock(return_value="<html>Test</html>")

            mock_manager.start = AsyncMock()
            mock_manager.stop = AsyncMock()
            mock_manager.get_page = AsyncMock()
            mock_manager.get_page.return_value.__aenter__ = AsyncMock(
                return_value=mock_page
            )
            mock_manager.get_page.return_value.__aexit__ = AsyncMock()

            MockManager.return_value = mock_manager

            response = client.post(
                "/api/v1/agent/browser/navigate",
                json={
                    "url": "https://example.com",
                    "browser_type": "chromium",
                    "headless": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["url"] == "https://example.com"
            assert data["title"] == "Example Domain"

    def test_navigate_invalid_url(self, client):
        """Test navigation with invalid URL."""
        response = client.post(
            "/api/v1/agent/browser/navigate",
            json={
                "url": "not-a-valid-url",
                "browser_type": "chromium",
            },
        )
        # Should still accept the request (browser will handle the error)
        assert response.status_code in [200, 500]


class TestScreenshotEndpoint:
    """Test /api/v1/agent/browser/screenshot endpoint."""

    
    def test_screenshot_success(self, client):
        """Test successful screenshot capture."""
        with patch("app.routers.browser.BrowserManager") as MockManager:
            mock_manager = AsyncMock()
            mock_page = AsyncMock()
            mock_page.url = "https://example.com"
            mock_page.goto = AsyncMock()

            screenshot_data = b"fake_screenshot_data"
            mock_manager.start = AsyncMock()
            mock_manager.stop = AsyncMock()
            mock_manager.get_page = AsyncMock()
            mock_manager.get_page.return_value.__aenter__ = AsyncMock(
                return_value=mock_page
            )
            mock_manager.get_page.return_value.__aexit__ = AsyncMock()
            mock_manager.take_screenshot = AsyncMock(return_value=screenshot_data)

            MockManager.return_value = mock_manager

            response = client.post(
                "/api/v1/agent/browser/screenshot",
                json={
                    "url": "https://example.com",
                    "full_page": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "screenshot_base64" in data
            # Verify base64 encoding
            decoded = base64.b64decode(data["screenshot_base64"])
            assert decoded == screenshot_data

    
    def test_screenshot_element(self, client):
        """Test element screenshot capture."""
        with patch("app.routers.browser.BrowserManager") as MockManager:
            mock_manager = AsyncMock()
            mock_page = AsyncMock()
            mock_page.url = "https://example.com"
            mock_page.goto = AsyncMock()

            mock_manager.start = AsyncMock()
            mock_manager.stop = AsyncMock()
            mock_manager.get_page = AsyncMock()
            mock_manager.get_page.return_value.__aenter__ = AsyncMock(
                return_value=mock_page
            )
            mock_manager.get_page.return_value.__aexit__ = AsyncMock()
            mock_manager.take_screenshot = AsyncMock(return_value=b"element_shot")

            MockManager.return_value = mock_manager

            response = client.post(
                "/api/v1/agent/browser/screenshot",
                json={
                    "url": "https://example.com",
                    "full_page": False,
                    "element_selector": ".header",
                },
            )

            assert response.status_code == 200


class TestScrapeEndpoint:
    """Test /api/v1/agent/browser/scrape endpoint."""

    
    def test_scrape_success(self, client):
        """Test successful content scraping."""
        with patch("app.routers.browser.BrowserManager") as MockManager, patch(
            "app.routers.browser.WebScraper"
        ) as MockScraper:
            mock_manager = AsyncMock()
            mock_page = AsyncMock()
            mock_page.url = "https://example.com"
            mock_page.goto = AsyncMock()
            mock_page.content = AsyncMock(return_value="<html>Test</html>")

            mock_manager.start = AsyncMock()
            mock_manager.stop = AsyncMock()
            mock_manager.get_page = AsyncMock()
            mock_manager.get_page.return_value.__aenter__ = AsyncMock(
                return_value=mock_page
            )
            mock_manager.get_page.return_value.__aexit__ = AsyncMock()

            MockManager.return_value = mock_manager

            # Mock scraper
            mock_scraper = MockScraper.return_value
            mock_content = AsyncMock()
            mock_content.url = "https://example.com"
            mock_content.title = "Example"
            mock_content.text = "Sample text"
            mock_content.html = "<html>Test</html>"
            mock_content.metadata = {"description": "Test"}
            mock_content.links = []
            mock_content.images = []
            mock_content.forms = []
            mock_content.tables = []

            mock_scraper.scrape_page = AsyncMock(return_value=mock_content)

            response = client.post(
                "/api/v1/agent/browser/scrape",
                json={
                    "url": "https://example.com",
                    "include_metadata": True,
                    "include_links": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["url"] == "https://example.com"
            assert data["title"] == "Example"
            assert "text" in data

    
    def test_scrape_with_selectors(self, client):
        """Test scraping with CSS selectors."""
        with patch("app.routers.browser.BrowserManager") as MockManager, patch(
            "app.routers.browser.WebScraper"
        ) as MockScraper:
            mock_manager = AsyncMock()
            mock_page = AsyncMock()
            mock_page.url = "https://example.com"
            mock_page.goto = AsyncMock()
            mock_page.content = AsyncMock(return_value="<html>Test</html>")

            mock_manager.start = AsyncMock()
            mock_manager.stop = AsyncMock()
            mock_manager.get_page = AsyncMock()
            mock_manager.get_page.return_value.__aenter__ = AsyncMock(
                return_value=mock_page
            )
            mock_manager.get_page.return_value.__aexit__ = AsyncMock()

            MockManager.return_value = mock_manager

            # Mock scraper
            mock_scraper = MockScraper.return_value
            mock_content = AsyncMock()
            mock_content.url = "https://example.com"
            mock_content.title = "Example"
            mock_content.text = "Sample text"
            mock_content.html = ""
            mock_content.metadata = {}
            mock_content.links = []
            mock_content.images = []
            mock_content.forms = []
            mock_content.tables = []

            mock_scraper.scrape_page = AsyncMock(return_value=mock_content)
            mock_scraper.extract_by_selector = lambda html, sel: [f"Result for {sel}"]

            response = client.post(
                "/api/v1/agent/browser/scrape",
                json={
                    "url": "https://example.com",
                    "css_selectors": ["h1", ".content"],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "custom_extractions" in data


class TestPDFEndpoint:
    """Test /api/v1/agent/browser/pdf endpoint."""

    
    def test_pdf_generation(self, client):
        """Test PDF generation."""
        with patch("app.routers.browser.BrowserManager") as MockManager:
            mock_manager = AsyncMock()
            mock_page = AsyncMock()
            mock_page.url = "https://example.com"
            mock_page.goto = AsyncMock()

            pdf_data = b"fake_pdf_data"
            mock_manager.start = AsyncMock()
            mock_manager.stop = AsyncMock()
            mock_manager.get_page = AsyncMock()
            mock_manager.get_page.return_value.__aenter__ = AsyncMock(
                return_value=mock_page
            )
            mock_manager.get_page.return_value.__aexit__ = AsyncMock()
            mock_manager.generate_pdf = AsyncMock(return_value=pdf_data)

            MockManager.return_value = mock_manager

            response = client.post(
                "/api/v1/agent/browser/pdf",
                json={
                    "url": "https://example.com",
                    "format": "A4",
                    "print_background": True,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "pdf_base64" in data

    
    def test_pdf_chromium_only(self, client):
        """Test PDF generation only works with Chromium."""
        response = client.post(
            "/api/v1/agent/browser/pdf",
            json={
                "url": "https://example.com",
                "browser_type": "firefox",
            },
        )

        assert response.status_code == 400
        assert "only supported in Chromium" in response.json()["detail"]


class TestExecuteScriptEndpoint:
    """Test /api/v1/agent/browser/execute endpoint."""

    
    def test_execute_script(self, client):
        """Test JavaScript execution."""
        with patch("app.routers.browser.BrowserManager") as MockManager:
            mock_manager = AsyncMock()
            mock_page = AsyncMock()
            mock_page.url = "https://example.com"
            mock_page.goto = AsyncMock()

            mock_manager.start = AsyncMock()
            mock_manager.stop = AsyncMock()
            mock_manager.get_page = AsyncMock()
            mock_manager.get_page.return_value.__aenter__ = AsyncMock(
                return_value=mock_page
            )
            mock_manager.get_page.return_value.__aexit__ = AsyncMock()
            mock_manager.execute_script = AsyncMock(
                return_value={"title": "Example Domain"}
            )

            MockManager.return_value = mock_manager

            response = client.post(
                "/api/v1/agent/browser/execute",
                json={
                    "url": "https://example.com",
                    "script": "return {title: document.title}",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "result" in data


class TestBrowserHealthEndpoint:
    """Test /api/v1/agent/browser/health endpoint."""

    
    def test_health_check(self, client):
        """Test browser health endpoint."""
        with patch("app.routers.browser.BrowserManager") as MockManager:
            mock_manager = AsyncMock()
            mock_manager.get_health = AsyncMock(
                return_value={
                    "status": "healthy",
                    "browser_type": "chromium",
                    "contexts": 0,
                    "pool_size": 3,
                }
            )

            MockManager.return_value = mock_manager

            response = client.get("/api/v1/agent/browser/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "browser_type" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.routers", "--cov-report=term-missing"])