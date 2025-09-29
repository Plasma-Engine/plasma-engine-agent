"""Browser automation manager using Playwright with support for multiple browsers."""

import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)


class BrowserType(str, Enum):
    """Supported browser types."""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


@dataclass
class BrowserConfig:
    """Configuration for browser automation."""

    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    viewport: Dict[str, int] = field(
        default_factory=lambda: {"width": 1920, "height": 1080}
    )
    user_agent: Optional[str] = None
    proxy: Optional[Dict[str, str]] = None
    locale: str = "en-US"
    timezone: str = "America/New_York"
    timeout: int = 30000  # 30 seconds
    ignore_https_errors: bool = False
    javascript_enabled: bool = True
    bypass_csp: bool = False
    extra_http_headers: Optional[Dict[str, str]] = None
    geolocation: Optional[Dict[str, float]] = None
    permissions: Optional[List[str]] = None
    color_scheme: Optional[str] = None  # "light", "dark", "no-preference"


class BrowserManager:
    """Manages browser instances with connection pooling and resource cleanup."""

    def __init__(self, config: Optional[BrowserConfig] = None, pool_size: int = 3):
        """Initialize browser manager with configuration.

        Args:
            config: Browser configuration settings
            pool_size: Maximum number of concurrent browser contexts
        """
        self.config = config or BrowserConfig()
        self.pool_size = pool_size
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._contexts: List[BrowserContext] = []
        self._lock: Optional[asyncio.Lock] = None

    async def start(self) -> None:
        """Start Playwright and launch browser."""
        # Initialize lock if not already done
        if self._lock is None:
            self._lock = asyncio.Lock()

        if self._playwright is None:
            logger.info(f"Starting Playwright with {self.config.browser_type} browser")
            self._playwright = await async_playwright().start()

            # Select browser based on config
            if self.config.browser_type == BrowserType.CHROMIUM:
                browser_launcher = self._playwright.chromium
            elif self.config.browser_type == BrowserType.FIREFOX:
                browser_launcher = self._playwright.firefox
            elif self.config.browser_type == BrowserType.WEBKIT:
                browser_launcher = self._playwright.webkit
            else:
                raise ValueError(f"Unsupported browser type: {self.config.browser_type}")

            # Launch browser with configuration
            launch_options = {
                "headless": self.config.headless,
                "proxy": self.config.proxy,
            }

            # Add anti-bot detection bypasses
            if self.config.browser_type == BrowserType.CHROMIUM:
                launch_options["args"] = [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                ]

            self._browser = await browser_launcher.launch(**launch_options)
            logger.info("Browser launched successfully")

    async def stop(self) -> None:
        """Stop browser and cleanup resources."""
        logger.info("Stopping browser manager")

        # Close all contexts
        for context in self._contexts:
            try:
                await context.close()
            except Exception as e:
                logger.warning(f"Error closing context: {e}")

        self._contexts.clear()

        # Close browser
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self._browser = None

        # Stop playwright
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.warning(f"Error stopping playwright: {e}")
            self._playwright = None

        logger.info("Browser manager stopped")

    @asynccontextmanager
    async def get_context(self) -> AsyncGenerator[BrowserContext, None]:
        """Get a browser context from the pool (context manager).

        Yields:
            Browser context ready for use
        """
        # Initialize lock if not already done
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if not self._browser:
                await self.start()

            # Create context options
            context_options = {
                "viewport": self.config.viewport,
                "user_agent": self.config.user_agent,
                "locale": self.config.locale,
                "timezone_id": self.config.timezone,
                "ignore_https_errors": self.config.ignore_https_errors,
                "java_script_enabled": self.config.javascript_enabled,
                "bypass_csp": self.config.bypass_csp,
            }

            # Add optional settings
            if self.config.extra_http_headers:
                context_options["extra_http_headers"] = self.config.extra_http_headers

            if self.config.geolocation:
                context_options["geolocation"] = self.config.geolocation

            if self.config.permissions:
                context_options["permissions"] = self.config.permissions

            if self.config.color_scheme:
                context_options["color_scheme"] = self.config.color_scheme

            # Create new context
            context = await self._browser.new_context(**context_options)

            # Add anti-detection scripts
            await context.add_init_script(
                """
                // Override navigator.webdriver
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });

                // Override plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });

                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });

                // Override chrome runtime
                window.chrome = {
                    runtime: {}
                };

                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """
            )

            self._contexts.append(context)

        try:
            yield context
        finally:
            # Initialize lock if not already done
            if self._lock is None:
                self._lock = asyncio.Lock()

            async with self._lock:
                try:
                    await context.close()
                    self._contexts.remove(context)
                except Exception as e:
                    logger.warning(f"Error cleaning up context: {e}")

    @asynccontextmanager
    async def get_page(self) -> AsyncGenerator[Page, None]:
        """Get a new page from a browser context.

        Yields:
            Page ready for navigation and interaction
        """
        async with self.get_context() as context:
            page = await context.new_page()
            page.set_default_timeout(self.config.timeout)

            try:
                yield page
            finally:
                try:
                    await page.close()
                except Exception as e:
                    logger.warning(f"Error closing page: {e}")

    async def navigate(
        self,
        url: str,
        wait_until: str = "networkidle",
        timeout: Optional[int] = None,
    ) -> Page:
        """Navigate to URL and return the page.

        Args:
            url: URL to navigate to
            wait_until: Wait condition ("load", "domcontentloaded", "networkidle", "commit")
            timeout: Optional timeout override

        Returns:
            Page object after navigation
        """
        async with self.get_page() as page:
            logger.info(f"Navigating to {url}")
            await page.goto(
                url,
                wait_until=wait_until,
                timeout=timeout or self.config.timeout,
            )
            return page

    async def execute_script(
        self, page: Page, script: str, *args: any
    ) -> Any:
        """Execute JavaScript on a page.

        Args:
            page: Page to execute script on
            script: JavaScript code to execute
            *args: Arguments to pass to script

        Returns:
            Result of script execution

        Raises:
            Exception: If script execution fails
        """
        try:
            logger.debug(f"Executing script: {script[:100]}...")
            return await page.evaluate(script, *args)
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            raise

    async def take_screenshot(
        self,
        page: Page,
        full_page: bool = True,
        element_selector: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """Take screenshot of page or element.

        Args:
            page: Page to screenshot
            full_page: Whether to capture full scrollable page
            element_selector: Optional CSS selector to screenshot specific element
            path: Optional file path to save screenshot

        Returns:
            Screenshot as bytes
        """
        logger.info(
            f"Taking screenshot (full_page={full_page}, selector={element_selector})"
        )

        if element_selector:
            element = await page.query_selector(element_selector)
            if not element:
                raise ValueError(f"Element not found: {element_selector}")
            screenshot = await element.screenshot(path=path)
        else:
            screenshot = await page.screenshot(full_page=full_page, path=path)

        return screenshot

    async def generate_pdf(
        self,
        page: Page,
        path: Optional[Union[str, Path]] = None,
        format: str = "A4",
        print_background: bool = True,
        margin: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Generate PDF from page.

        Args:
            page: Page to convert to PDF
            path: Optional file path to save PDF
            format: Paper format (A4, Letter, etc.)
            print_background: Whether to print background graphics
            margin: Optional margins (top, right, bottom, left)

        Returns:
            PDF as bytes
        """
        logger.info(f"Generating PDF (format={format})")

        pdf_options = {
            "format": format,
            "print_background": print_background,
        }

        if path:
            pdf_options["path"] = path

        if margin:
            pdf_options["margin"] = margin

        pdf = await page.pdf(**pdf_options)
        return pdf

    async def set_cookies(
        self, context: BrowserContext, cookies: List[Dict[str, Any]]
    ) -> None:
        """Set cookies in browser context.

        Args:
            context: Browser context
            cookies: List of cookie dictionaries

        Raises:
            Exception: If cookie setting fails
        """
        try:
            logger.debug(f"Setting {len(cookies)} cookies")
            await context.add_cookies(cookies)
        except Exception as e:
            logger.error(f"Failed to set cookies: {e}")
            raise

    async def get_cookies(self, context: BrowserContext) -> List[Dict[str, Any]]:
        """Get all cookies from browser context.

        Args:
            context: Browser context

        Returns:
            List of cookie dictionaries

        Raises:
            Exception: If cookie retrieval fails
        """
        try:
            cookies = await context.cookies()
            logger.debug(f"Retrieved {len(cookies)} cookies")
            return cookies
        except Exception as e:
            logger.error(f"Failed to get cookies: {e}")
            raise

    async def clear_cookies(self, context: BrowserContext) -> None:
        """Clear all cookies from browser context.

        Args:
            context: Browser context
        """
        logger.debug("Clearing cookies")
        await context.clear_cookies()

    async def set_storage(
        self, context: BrowserContext, origin: str, storage: Dict[str, Any]
    ) -> None:
        """Set local storage for an origin.

        Args:
            context: Browser context
            origin: Origin URL
            storage: Storage dictionary

        Raises:
            Exception: If storage setting fails
        """
        try:
            logger.debug(f"Setting storage for {origin}")
            await context.add_init_script(
                f"""
                localStorage.clear();
                for (const [key, value] of Object.entries({storage})) {{
                    localStorage.setItem(key, value);
                }}
            """
            )
        except Exception as e:
            logger.error(f"Failed to set storage for {origin}: {e}")
            raise

    async def wait_for_selector(
        self,
        page: Page,
        selector: str,
        timeout: Optional[int] = None,
        state: str = "visible",
    ) -> None:
        """Wait for element to appear.

        Args:
            page: Page to wait on
            selector: CSS selector
            timeout: Optional timeout override
            state: Element state to wait for ("attached", "detached", "visible", "hidden")
        """
        logger.debug(f"Waiting for selector: {selector}")
        await page.wait_for_selector(
            selector,
            timeout=timeout or self.config.timeout,
            state=state,
        )

    async def click(
        self,
        page: Page,
        selector: str,
        timeout: Optional[int] = None,
    ) -> None:
        """Click on element.

        Args:
            page: Page to click on
            selector: CSS selector
            timeout: Optional timeout override
        """
        logger.debug(f"Clicking selector: {selector}")
        await page.click(selector, timeout=timeout or self.config.timeout)

    async def fill_form(
        self,
        page: Page,
        selector: str,
        text: str,
        timeout: Optional[int] = None,
    ) -> None:
        """Fill form input.

        Args:
            page: Page with form
            selector: CSS selector for input
            text: Text to fill
            timeout: Optional timeout override
        """
        logger.debug(f"Filling form: {selector}")
        await page.fill(selector, text, timeout=timeout or self.config.timeout)

    async def select_option(
        self,
        page: Page,
        selector: str,
        value: Union[str, List[str]],
        timeout: Optional[int] = None,
    ) -> None:
        """Select option from dropdown.

        Args:
            page: Page with dropdown
            selector: CSS selector for select element
            value: Option value(s) to select
            timeout: Optional timeout override
        """
        logger.debug(f"Selecting option: {selector} = {value}")
        await page.select_option(
            selector, value, timeout=timeout or self.config.timeout
        )

    async def get_health(self) -> Dict[str, Any]:
        """Get browser manager health status.

        Returns:
            Health status dictionary with browser status, type, contexts, and pool size
        """
        return {
            "status": "healthy" if self._browser else "not_started",
            "browser_type": self.config.browser_type,
            "contexts": len(self._contexts),
            "pool_size": self.pool_size,
        }