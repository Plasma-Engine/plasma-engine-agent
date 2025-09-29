"""Comprehensive tests for browser automation functionality."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from playwright.async_api import Browser, BrowserContext, Page, Playwright

from app.automation import BrowserConfig, BrowserManager, BrowserType, WebScraper


@pytest.fixture
def browser_config():
    """Create test browser configuration."""
    return BrowserConfig(
        browser_type=BrowserType.CHROMIUM,
        headless=True,
        viewport={"width": 1280, "height": 720},
        timeout=15000,
    )


@pytest.fixture
def browser_manager(browser_config):
    """Create browser manager instance."""
    return BrowserManager(config=browser_config, pool_size=2)


@pytest.fixture
def web_scraper():
    """Create web scraper instance."""
    return WebScraper(parser="lxml")


class TestBrowserConfig:
    """Test BrowserConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BrowserConfig()
        assert config.browser_type == BrowserType.CHROMIUM
        assert config.headless is True
        assert config.viewport == {"width": 1920, "height": 1080}
        assert config.timeout == 30000
        assert config.javascript_enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BrowserConfig(
            browser_type=BrowserType.FIREFOX,
            headless=False,
            viewport={"width": 1024, "height": 768},
            timeout=60000,
            javascript_enabled=False,
        )
        assert config.browser_type == BrowserType.FIREFOX
        assert config.headless is False
        assert config.viewport == {"width": 1024, "height": 768}
        assert config.timeout == 60000
        assert config.javascript_enabled is False

    def test_proxy_config(self):
        """Test proxy configuration."""
        config = BrowserConfig(
            proxy={"server": "http://proxy.example.com:8080", "username": "user"}
        )
        assert config.proxy is not None
        assert config.proxy["server"] == "http://proxy.example.com:8080"


class TestBrowserManager:
    """Test BrowserManager functionality."""

    @pytest.mark.asyncio
    async def test_start_stop(self, browser_manager):
        """Test browser start and stop lifecycle."""
        # Mock playwright
        with patch("app.automation.browser_manager.async_playwright") as mock_pw:
            mock_playwright = AsyncMock()
            mock_browser = AsyncMock()
            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_pw.return_value.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value.stop = AsyncMock()

            await browser_manager.start()
            assert browser_manager._browser is not None

            await browser_manager.stop()
            assert browser_manager._browser is None

    @pytest.mark.asyncio
    async def test_get_context(self, browser_manager):
        """Test browser context creation."""
        with patch("app.automation.browser_manager.async_playwright") as mock_pw:
            mock_playwright = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.add_init_script = AsyncMock()
            mock_context.close = AsyncMock()

            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_pw.return_value.start = AsyncMock(return_value=mock_playwright)

            async with browser_manager.get_context() as context:
                assert context is not None
                # Verify anti-detection script was added
                mock_context.add_init_script.assert_called_once()

            # Verify context was closed
            mock_context.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_page(self, browser_manager):
        """Test page creation."""
        with patch("app.automation.browser_manager.async_playwright") as mock_pw:
            mock_playwright = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_context.add_init_script = AsyncMock()
            mock_context.close = AsyncMock()
            mock_page.close = AsyncMock()
            mock_page.set_default_timeout = Mock()

            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_pw.return_value.start = AsyncMock(return_value=mock_playwright)

            async with browser_manager.get_page() as page:
                assert page is not None
                mock_page.set_default_timeout.assert_called_once()

            # Verify page was closed
            mock_page.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_take_screenshot_full_page(self, browser_manager):
        """Test full page screenshot capture."""
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"screenshot_data")

        screenshot = await browser_manager.take_screenshot(mock_page, full_page=True)

        assert screenshot == b"screenshot_data"
        mock_page.screenshot.assert_called_once_with(full_page=True, path=None)

    @pytest.mark.asyncio
    async def test_take_screenshot_element(self, browser_manager):
        """Test element screenshot capture."""
        mock_page = AsyncMock()
        mock_element = AsyncMock()
        mock_element.screenshot = AsyncMock(return_value=b"element_screenshot")
        mock_page.query_selector = AsyncMock(return_value=mock_element)

        screenshot = await browser_manager.take_screenshot(
            mock_page, element_selector=".test-element"
        )

        assert screenshot == b"element_screenshot"
        mock_page.query_selector.assert_called_once_with(".test-element")
        mock_element.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_pdf(self, browser_manager):
        """Test PDF generation."""
        mock_page = AsyncMock()
        mock_page.pdf = AsyncMock(return_value=b"pdf_data")

        pdf = await browser_manager.generate_pdf(mock_page, format="A4")

        assert pdf == b"pdf_data"
        mock_page.pdf.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_script(self, browser_manager):
        """Test JavaScript execution."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value={"result": "success"})

        result = await browser_manager.execute_script(
            mock_page, "return document.title"
        )

        assert result == {"result": "success"}
        mock_page.evaluate.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_cookies(self, browser_manager):
        """Test cookie setting."""
        mock_context = AsyncMock()
        mock_context.add_cookies = AsyncMock()

        cookies = [{"name": "session", "value": "abc123", "domain": "example.com"}]
        await browser_manager.set_cookies(mock_context, cookies)

        mock_context.add_cookies.assert_called_once_with(cookies)

    @pytest.mark.asyncio
    async def test_get_cookies(self, browser_manager):
        """Test cookie retrieval."""
        mock_context = AsyncMock()
        cookies = [{"name": "session", "value": "abc123"}]
        mock_context.cookies = AsyncMock(return_value=cookies)

        result = await browser_manager.get_cookies(mock_context)

        assert result == cookies
        mock_context.cookies.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_selector(self, browser_manager):
        """Test waiting for element."""
        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()

        await browser_manager.wait_for_selector(mock_page, ".test-selector")

        mock_page.wait_for_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_click(self, browser_manager):
        """Test element clicking."""
        mock_page = AsyncMock()
        mock_page.click = AsyncMock()

        await browser_manager.click(mock_page, ".test-button")

        mock_page.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_fill_form(self, browser_manager):
        """Test form filling."""
        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()

        await browser_manager.fill_form(mock_page, "#email", "test@example.com")

        mock_page.fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_option(self, browser_manager):
        """Test option selection."""
        mock_page = AsyncMock()
        mock_page.select_option = AsyncMock()

        await browser_manager.select_option(mock_page, "#country", "USA")

        mock_page.select_option.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_status(self, browser_manager):
        """Test health status check."""
        health = await browser_manager.get_health()

        assert "status" in health
        assert "browser_type" in health
        assert health["browser_type"] == BrowserType.CHROMIUM


class TestWebScraper:
    """Test WebScraper functionality."""

    @pytest.mark.asyncio
    async def test_scrape_page_basic(self, web_scraper):
        """Test basic page scraping."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example Domain")
        mock_page.content = AsyncMock(
            return_value="""
            <html>
                <head><title>Example Domain</title></head>
                <body>
                    <h1>Example Domain</h1>
                    <p>This is an example.</p>
                </body>
            </html>
        """
        )

        content = await web_scraper.scrape_page(
            mock_page,
            include_html=True,
            include_metadata=True,
            include_links=True,
            include_images=True,
        )

        assert content.url == "https://example.com"
        assert content.title == "Example Domain"
        assert "Example Domain" in content.text
        assert "This is an example" in content.text
        assert len(content.html) > 0

    @pytest.mark.asyncio
    async def test_extract_metadata(self, web_scraper):
        """Test metadata extraction."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.content = AsyncMock(
            return_value="""
            <html>
                <head>
                    <meta name="description" content="Test description">
                    <meta property="og:title" content="OG Title">
                    <meta name="twitter:card" content="summary">
                </head>
                <body></body>
            </html>
        """
        )

        content = await web_scraper.scrape_page(mock_page, include_metadata=True)

        assert "description" in content.metadata
        assert content.metadata["description"] == "Test description"
        assert "og" in content.metadata
        assert "twitter" in content.metadata

    @pytest.mark.asyncio
    async def test_extract_links(self, web_scraper):
        """Test link extraction."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.content = AsyncMock(
            return_value="""
            <html>
                <body>
                    <a href="https://example.com/page1">Page 1</a>
                    <a href="/page2" title="Page 2">Page 2</a>
                </body>
            </html>
        """
        )

        content = await web_scraper.scrape_page(mock_page, include_links=True)

        assert len(content.links) == 2
        assert content.links[0]["href"] == "https://example.com/page1"
        assert content.links[0]["text"] == "Page 1"

    @pytest.mark.asyncio
    async def test_extract_images(self, web_scraper):
        """Test image extraction."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.content = AsyncMock(
            return_value="""
            <html>
                <body>
                    <img src="https://example.com/img1.jpg" alt="Image 1">
                    <img src="/img2.png" alt="Image 2" width="100" height="100">
                </body>
            </html>
        """
        )

        content = await web_scraper.scrape_page(mock_page, include_images=True)

        assert len(content.images) == 2
        assert content.images[0]["src"] == "https://example.com/img1.jpg"
        assert content.images[0]["alt"] == "Image 1"
        assert content.images[1]["width"] == "100"

    @pytest.mark.asyncio
    async def test_extract_forms(self, web_scraper):
        """Test form extraction."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.content = AsyncMock(
            return_value="""
            <html>
                <body>
                    <form action="/submit" method="POST" name="contact">
                        <input type="text" name="name" placeholder="Name" required>
                        <input type="email" name="email" placeholder="Email">
                        <select name="country">
                            <option value="us">USA</option>
                            <option value="uk">UK</option>
                        </select>
                        <textarea name="message"></textarea>
                    </form>
                </body>
            </html>
        """
        )

        content = await web_scraper.scrape_page(mock_page, include_forms=True)

        assert len(content.forms) == 1
        form = content.forms[0]
        assert form["action"] == "/submit"
        assert form["method"] == "POST"
        assert len(form["fields"]) == 4
        assert form["fields"][0]["name"] == "name"
        assert form["fields"][0]["required"] is True

    @pytest.mark.asyncio
    async def test_extract_tables(self, web_scraper):
        """Test table extraction."""
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Test")
        mock_page.content = AsyncMock(
            return_value="""
            <html>
                <body>
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Age</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>John</td>
                                <td>30</td>
                            </tr>
                            <tr>
                                <td>Jane</td>
                                <td>25</td>
                            </tr>
                        </tbody>
                    </table>
                </body>
            </html>
        """
        )

        content = await web_scraper.scrape_page(mock_page, include_tables=True)

        assert len(content.tables) == 1
        table = content.tables[0]
        assert len(table) == 2
        assert table[0]["Name"] == "John"
        assert table[0]["Age"] == "30"
        assert table[1]["Name"] == "Jane"

    def test_extract_by_selector(self, web_scraper):
        """Test CSS selector extraction."""
        html = """
            <html>
                <body>
                    <div class="content">
                        <h2>Title 1</h2>
                        <h2>Title 2</h2>
                    </div>
                </body>
            </html>
        """

        results = web_scraper.extract_by_selector(html, "h2")
        assert len(results) == 2
        assert results[0] == "Title 1"
        assert results[1] == "Title 2"

    def test_extract_by_selector_attribute(self, web_scraper):
        """Test CSS selector with attribute extraction."""
        html = """
            <html>
                <body>
                    <a href="/page1">Link 1</a>
                    <a href="/page2">Link 2</a>
                </body>
            </html>
        """

        results = web_scraper.extract_by_selector(html, "a", attribute="href")
        assert len(results) == 2
        assert results[0] == "/page1"
        assert results[1] == "/page2"

    def test_extract_by_xpath(self, web_scraper):
        """Test XPath extraction."""
        html = """
            <html>
                <body>
                    <div class="content">
                        <p>Paragraph 1</p>
                        <p>Paragraph 2</p>
                    </div>
                </body>
            </html>
        """

        results = web_scraper.extract_by_xpath(html, "//p/text()")
        assert len(results) == 2
        assert "Paragraph 1" in results
        assert "Paragraph 2" in results

    def test_clean_text(self, web_scraper):
        """Test text cleaning."""
        dirty_text = "  This   is   \n\n\n  messy    text  \n\n  "
        clean_text = web_scraper.clean_text(dirty_text)

        # The clean_text method removes extra whitespace and normalizes
        assert "This is" in clean_text
        assert "messy text" in clean_text
        assert not clean_text.startswith(" ")
        assert not clean_text.endswith(" ")
        # Check that multiple spaces are reduced to single space
        assert "  " not in clean_text


@pytest.mark.asyncio
class TestBrowserIntegration:
    """Integration tests for browser automation."""

    async def test_full_navigation_flow(self, browser_manager):
        """Test complete navigation workflow."""
        with patch("app.automation.browser_manager.async_playwright") as mock_pw:
            # Setup mocks
            mock_playwright = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()

            mock_page.url = "https://example.com"
            mock_page.goto = AsyncMock()
            mock_page.close = AsyncMock()
            mock_page.set_default_timeout = Mock()

            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_context.add_init_script = AsyncMock()
            mock_context.close = AsyncMock()

            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_pw.return_value.start = AsyncMock(return_value=mock_playwright)
            mock_pw.return_value.stop = AsyncMock()

            # Execute navigation
            await browser_manager.start()

            async with browser_manager.get_page() as page:
                await page.goto("https://example.com")
                assert page.url == "https://example.com"

            await browser_manager.stop()

            # Verify cleanup
            mock_page.close.assert_called()
            mock_context.close.assert_called()

    async def test_concurrent_contexts(self, browser_manager):
        """Test concurrent browser contexts."""
        with patch("app.automation.browser_manager.async_playwright") as mock_pw:
            mock_playwright = AsyncMock()
            mock_browser = AsyncMock()

            def create_mock_context():
                context = AsyncMock()
                context.add_init_script = AsyncMock()
                context.close = AsyncMock()
                return context

            contexts = [create_mock_context() for _ in range(3)]
            mock_browser.new_context = AsyncMock(side_effect=contexts)

            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_pw.return_value.start = AsyncMock(return_value=mock_playwright)

            await browser_manager.start()

            # Create multiple contexts concurrently
            tasks = []
            for _ in range(3):

                async def use_context():
                    async with browser_manager.get_context() as ctx:
                        await asyncio.sleep(0.01)

                tasks.append(use_context())

            await asyncio.gather(*tasks)

            # Verify all contexts were created and closed
            assert mock_browser.new_context.call_count == 3
            for context in contexts:
                context.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.automation", "--cov-report=term-missing"])