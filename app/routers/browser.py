"""Browser automation API endpoints."""

import base64
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from ..automation import BrowserConfig, BrowserManager, BrowserType, WebScraper

router = APIRouter(prefix="/api/v1/agent/browser", tags=["browser"])

# Global browser manager instance (singleton pattern)
_browser_manager: Optional[BrowserManager] = None


def get_browser_manager() -> BrowserManager:
    """Get or create browser manager singleton."""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager()
    return _browser_manager


class NavigateRequest(BaseModel):
    """Request model for page navigation."""

    url: str = Field(..., description="URL to navigate to")
    browser_type: BrowserType = Field(
        default=BrowserType.CHROMIUM, description="Browser type to use"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    wait_until: str = Field(
        default="networkidle",
        description="Wait condition (load, domcontentloaded, networkidle, commit)",
    )
    timeout: Optional[int] = Field(
        default=None, description="Navigation timeout in milliseconds"
    )
    user_agent: Optional[str] = Field(default=None, description="Custom user agent")
    proxy: Optional[Dict[str, str]] = Field(default=None, description="Proxy configuration")
    viewport: Optional[Dict[str, int]] = Field(
        default=None, description="Viewport dimensions"
    )
    javascript_enabled: bool = Field(default=True, description="Enable JavaScript")


class NavigateResponse(BaseModel):
    """Response model for page navigation."""

    success: bool
    url: str
    title: str
    content: str
    status_code: Optional[int] = None


class ScreenshotRequest(BaseModel):
    """Request model for screenshot capture."""

    url: str = Field(..., description="URL to capture")
    browser_type: BrowserType = Field(
        default=BrowserType.CHROMIUM, description="Browser type to use"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    full_page: bool = Field(default=True, description="Capture full scrollable page")
    element_selector: Optional[str] = Field(
        default=None, description="CSS selector to capture specific element"
    )
    timeout: Optional[int] = Field(
        default=None, description="Navigation timeout in milliseconds"
    )
    viewport: Optional[Dict[str, int]] = Field(
        default=None, description="Viewport dimensions"
    )


class ScreenshotResponse(BaseModel):
    """Response model for screenshot capture."""

    success: bool
    url: str
    screenshot_base64: str
    width: Optional[int] = None
    height: Optional[int] = None


class ScrapeRequest(BaseModel):
    """Request model for web scraping."""

    url: str = Field(..., description="URL to scrape")
    browser_type: BrowserType = Field(
        default=BrowserType.CHROMIUM, description="Browser type to use"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    include_html: bool = Field(default=False, description="Include raw HTML")
    include_metadata: bool = Field(default=True, description="Include meta tags")
    include_links: bool = Field(default=True, description="Include links")
    include_images: bool = Field(default=True, description="Include images")
    include_forms: bool = Field(default=False, description="Include form data")
    include_tables: bool = Field(default=False, description="Include table data")
    include_scripts: bool = Field(default=False, description="Include script tags")
    include_styles: bool = Field(default=False, description="Include style tags")
    css_selectors: Optional[List[str]] = Field(
        default=None, description="CSS selectors to extract specific content"
    )
    xpath_expressions: Optional[List[str]] = Field(
        default=None, description="XPath expressions to extract specific content"
    )
    timeout: Optional[int] = Field(
        default=None, description="Navigation timeout in milliseconds"
    )
    user_agent: Optional[str] = Field(default=None, description="Custom user agent")


class ScrapeResponse(BaseModel):
    """Response model for web scraping."""

    success: bool
    url: str
    title: Optional[str] = None
    text: str
    html: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    links: List[Dict[str, str]] = Field(default_factory=list)
    images: List[Dict[str, str]] = Field(default_factory=list)
    forms: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[List[Dict[str, str]]] = Field(default_factory=list)
    custom_extractions: Dict[str, List[Any]] = Field(default_factory=dict)


class PDFRequest(BaseModel):
    """Request model for PDF generation."""

    url: str = Field(..., description="URL to convert to PDF")
    browser_type: BrowserType = Field(
        default=BrowserType.CHROMIUM, description="Browser type to use"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    format: str = Field(default="A4", description="Paper format (A4, Letter, etc.)")
    print_background: bool = Field(
        default=True, description="Print background graphics"
    )
    margin: Optional[Dict[str, str]] = Field(default=None, description="Page margins")
    timeout: Optional[int] = Field(
        default=None, description="Navigation timeout in milliseconds"
    )


class PDFResponse(BaseModel):
    """Response model for PDF generation."""

    success: bool
    url: str
    pdf_base64: str


class ExecuteScriptRequest(BaseModel):
    """Request model for JavaScript execution."""

    url: str = Field(..., description="URL to execute script on")
    script: str = Field(..., description="JavaScript code to execute")
    browser_type: BrowserType = Field(
        default=BrowserType.CHROMIUM, description="Browser type to use"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    timeout: Optional[int] = Field(
        default=None, description="Navigation timeout in milliseconds"
    )


class ExecuteScriptResponse(BaseModel):
    """Response model for JavaScript execution."""

    success: bool
    url: str
    result: Any


@router.post(
    "/navigate",
    response_model=NavigateResponse,
    status_code=status.HTTP_200_OK,
    summary="Navigate to URL",
    description="Navigate to a URL and return page content",
)
async def navigate_to_url(
    request: NavigateRequest,
    browser_manager: BrowserManager = Depends(get_browser_manager),
) -> NavigateResponse:
    """Navigate to URL and return page content."""
    try:
        # Configure browser
        config = BrowserConfig(
            browser_type=request.browser_type,
            headless=request.headless,
            viewport=request.viewport or {"width": 1920, "height": 1080},
            user_agent=request.user_agent,
            proxy=request.proxy,
            timeout=request.timeout or 30000,
            javascript_enabled=request.javascript_enabled,
        )

        # Create temporary browser manager with config
        temp_manager = BrowserManager(config=config)

        try:
            await temp_manager.start()

            async with temp_manager.get_page() as page:
                # Navigate to URL
                await page.goto(
                    request.url,
                    wait_until=request.wait_until,
                    timeout=config.timeout,
                )

                # Get page details
                title = await page.title()
                content = await page.content()

                logger.info(f"Successfully navigated to {request.url}")

                return NavigateResponse(
                    success=True,
                    url=page.url,
                    title=title,
                    content=content,
                )

        finally:
            await temp_manager.stop()

    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Navigation failed: {str(e)}",
        )


@router.post(
    "/screenshot",
    response_model=ScreenshotResponse,
    status_code=status.HTTP_200_OK,
    summary="Capture screenshot",
    description="Capture screenshot of a web page or specific element",
)
async def capture_screenshot(
    request: ScreenshotRequest,
    browser_manager: BrowserManager = Depends(get_browser_manager),
) -> ScreenshotResponse:
    """Capture screenshot of page or element."""
    try:
        # Configure browser
        config = BrowserConfig(
            browser_type=request.browser_type,
            headless=request.headless,
            viewport=request.viewport or {"width": 1920, "height": 1080},
            timeout=request.timeout or 30000,
        )

        # Create temporary browser manager with config
        temp_manager = BrowserManager(config=config)

        try:
            await temp_manager.start()

            async with temp_manager.get_page() as page:
                # Navigate to URL
                await page.goto(request.url, wait_until="networkidle")

                # Capture screenshot
                screenshot = await temp_manager.take_screenshot(
                    page,
                    full_page=request.full_page,
                    element_selector=request.element_selector,
                )

                # Encode to base64
                screenshot_base64 = base64.b64encode(screenshot).decode("utf-8")

                logger.info(f"Screenshot captured for {request.url}")

                return ScreenshotResponse(
                    success=True,
                    url=page.url,
                    screenshot_base64=screenshot_base64,
                    width=config.viewport["width"],
                    height=config.viewport["height"],
                )

        finally:
            await temp_manager.stop()

    except Exception as e:
        logger.error(f"Screenshot capture failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screenshot capture failed: {str(e)}",
        )


@router.post(
    "/scrape",
    response_model=ScrapeResponse,
    status_code=status.HTTP_200_OK,
    summary="Scrape web content",
    description="Extract structured content from a web page",
)
async def scrape_content(
    request: ScrapeRequest,
    browser_manager: BrowserManager = Depends(get_browser_manager),
) -> ScrapeResponse:
    """Scrape and extract content from web page."""
    try:
        # Configure browser
        config = BrowserConfig(
            browser_type=request.browser_type,
            headless=request.headless,
            viewport={"width": 1920, "height": 1080},
            user_agent=request.user_agent,
            timeout=request.timeout or 30000,
        )

        # Create temporary browser manager and scraper
        temp_manager = BrowserManager(config=config)
        scraper = WebScraper()

        try:
            await temp_manager.start()

            async with temp_manager.get_page() as page:
                # Navigate to URL
                await page.goto(request.url, wait_until="networkidle")

                # Scrape page
                content = await scraper.scrape_page(
                    page,
                    include_html=request.include_html,
                    include_metadata=request.include_metadata,
                    include_links=request.include_links,
                    include_images=request.include_images,
                    include_forms=request.include_forms,
                    include_tables=request.include_tables,
                    include_scripts=request.include_scripts,
                    include_styles=request.include_styles,
                )

                # Custom extractions
                custom_extractions = {}

                # Extract by CSS selectors
                if request.css_selectors:
                    html = await page.content()
                    for selector in request.css_selectors:
                        extracted = scraper.extract_by_selector(html, selector)
                        custom_extractions[f"css_{selector}"] = extracted

                # Extract by XPath
                if request.xpath_expressions:
                    html = await page.content()
                    for xpath in request.xpath_expressions:
                        extracted = scraper.extract_by_xpath(html, xpath)
                        custom_extractions[f"xpath_{xpath}"] = extracted

                logger.info(f"Content scraped from {request.url}")

                return ScrapeResponse(
                    success=True,
                    url=content.url,
                    title=content.title,
                    text=content.text,
                    html=content.html,
                    metadata=content.metadata,
                    links=content.links,
                    images=content.images,
                    forms=content.forms,
                    tables=content.tables,
                    custom_extractions=custom_extractions,
                )

        finally:
            await temp_manager.stop()

    except Exception as e:
        logger.error(f"Content scraping failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content scraping failed: {str(e)}",
        )


@router.post(
    "/pdf",
    response_model=PDFResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate PDF",
    description="Generate PDF from a web page",
)
async def generate_pdf(
    request: PDFRequest,
    browser_manager: BrowserManager = Depends(get_browser_manager),
) -> PDFResponse:
    """Generate PDF from web page."""
    try:
        # Configure browser (only Chromium supports PDF)
        if request.browser_type != BrowserType.CHROMIUM:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF generation only supported in Chromium",
            )

        config = BrowserConfig(
            browser_type=BrowserType.CHROMIUM,
            headless=request.headless,
            timeout=request.timeout or 30000,
        )

        # Create temporary browser manager
        temp_manager = BrowserManager(config=config)

        try:
            await temp_manager.start()

            async with temp_manager.get_page() as page:
                # Navigate to URL
                await page.goto(request.url, wait_until="networkidle")

                # Generate PDF
                pdf = await temp_manager.generate_pdf(
                    page,
                    format=request.format,
                    print_background=request.print_background,
                    margin=request.margin,
                )

                # Encode to base64
                pdf_base64 = base64.b64encode(pdf).decode("utf-8")

                logger.info(f"PDF generated for {request.url}")

                return PDFResponse(
                    success=True,
                    url=page.url,
                    pdf_base64=pdf_base64,
                )

        finally:
            await temp_manager.stop()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {str(e)}",
        )


@router.post(
    "/execute",
    response_model=ExecuteScriptResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute JavaScript",
    description="Execute JavaScript code on a web page",
)
async def execute_javascript(
    request: ExecuteScriptRequest,
    browser_manager: BrowserManager = Depends(get_browser_manager),
) -> ExecuteScriptResponse:
    """Execute JavaScript on page."""
    try:
        # Configure browser
        config = BrowserConfig(
            browser_type=request.browser_type,
            headless=request.headless,
            timeout=request.timeout or 30000,
            javascript_enabled=True,
        )

        # Create temporary browser manager
        temp_manager = BrowserManager(config=config)

        try:
            await temp_manager.start()

            async with temp_manager.get_page() as page:
                # Navigate to URL
                await page.goto(request.url, wait_until="networkidle")

                # Execute script
                result = await temp_manager.execute_script(page, request.script)

                logger.info(f"Script executed on {request.url}")

                return ExecuteScriptResponse(
                    success=True,
                    url=page.url,
                    result=result,
                )

        finally:
            await temp_manager.stop()

    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Script execution failed: {str(e)}",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Browser health check",
    description="Check browser automation service health",
)
async def browser_health(
    browser_manager: BrowserManager = Depends(get_browser_manager),
) -> Dict[str, Any]:
    """Check browser automation service health."""
    try:
        health = await browser_manager.get_health()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )