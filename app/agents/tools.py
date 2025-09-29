"""Custom tool definitions for LangChain agents with MCP integration."""

import asyncio
import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

from ..automation.browser_manager import BrowserManager


class ToolCategory(str, Enum):
    """Categories for organizing tools."""

    SEARCH = "search"
    BROWSER = "browser"
    CALCULATOR = "calculator"
    CODE = "code"
    MCP = "mcp"
    CUSTOM = "custom"


class ToolMetadata(BaseModel):
    """Metadata for tool registration and discovery."""

    name: str
    description: str
    category: ToolCategory
    requires_api_key: bool = False
    rate_limit_per_minute: int = 60
    enabled: bool = True


class ToolRegistry:
    """Registry for managing and discovering LangChain tools."""

    def __init__(self) -> None:
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._usage_stats: Dict[str, int] = {}
        self._browser_manager: Optional[BrowserManager] = None

    async def initialize_browser(self) -> None:
        """Initialize browser manager for browser-based tools."""
        if not self._browser_manager:
            self._browser_manager = BrowserManager()

    def register_tool(self, tool: BaseTool, metadata: ToolMetadata) -> None:
        """Register a tool with metadata."""
        self._tools[metadata.name] = tool
        self._metadata[metadata.name] = metadata
        self._usage_stats[metadata.name] = 0

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a specific category."""
        return [
            self._tools[name]
            for name, meta in self._metadata.items()
            if meta.category == category and meta.enabled
        ]

    def get_all_tools(self) -> List[BaseTool]:
        """Get all enabled tools."""
        return [
            self._tools[name] for name, meta in self._metadata.items() if meta.enabled
        ]

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get metadata for a tool."""
        return self._metadata.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools with their metadata."""
        return [
            {
                "name": name,
                "description": meta.description,
                "category": meta.category,
                "requires_api_key": meta.requires_api_key,
                "rate_limit": meta.rate_limit_per_minute,
                "enabled": meta.enabled,
                "usage_count": self._usage_stats.get(name, 0),
            }
            for name, meta in self._metadata.items()
        ]

    def increment_usage(self, name: str) -> None:
        """Track tool usage for analytics."""
        if name in self._usage_stats:
            self._usage_stats[name] += 1

    async def create_browser_tool(self) -> BaseTool:
        """Create browser navigation tool."""
        await self.initialize_browser()

        async def navigate_browser(url: str) -> str:
            """Navigate to a URL and return page content."""
            try:
                if not self._browser_manager:
                    return "Error: Browser not initialized"

                page = await self._browser_manager.get_page()
                await page.goto(url, wait_until="networkidle")
                content = await page.content()
                return f"Successfully navigated to {url}. Page length: {len(content)} chars"
            except Exception as e:
                return f"Error navigating to {url}: {str(e)}"

        return StructuredTool.from_function(
            func=navigate_browser,
            name="browser_navigate",
            description="Navigate to a URL using a browser and return page information. Input should be a valid URL.",
            coroutine=navigate_browser,
        )

    async def create_screenshot_tool(self) -> BaseTool:
        """Create screenshot capture tool."""
        await self.initialize_browser()

        async def capture_screenshot(url: str) -> str:
            """Capture a screenshot of a webpage."""
            try:
                if not self._browser_manager:
                    return "Error: Browser not initialized"

                page = await self._browser_manager.get_page()
                await page.goto(url, wait_until="networkidle")
                screenshot_path = f"/tmp/screenshot_{url.replace('://', '_').replace('/', '_')}.png"
                await page.screenshot(path=screenshot_path)
                return f"Screenshot saved to {screenshot_path}"
            except Exception as e:
                return f"Error capturing screenshot: {str(e)}"

        return StructuredTool.from_function(
            func=capture_screenshot,
            name="browser_screenshot",
            description="Capture a screenshot of a webpage. Input should be a valid URL.",
            coroutine=capture_screenshot,
        )

    def create_calculator_tool(self) -> BaseTool:
        """Create calculator tool for mathematical expressions."""

        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression safely."""
            try:
                import numexpr

                # Sanitize input - only allow numbers, operators, and parentheses
                allowed_chars = set("0123456789+-*/().% ")
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters in expression"

                result = numexpr.evaluate(expression).item()
                return f"Result: {result}"
            except Exception as e:
                return f"Error calculating expression: {str(e)}"

        return Tool(
            name="calculator",
            func=calculate,
            description="Evaluate mathematical expressions. Input should be a valid mathematical expression like '2 + 2' or '10 * 5 / 2'.",
        )

    def create_search_tool(self) -> BaseTool:
        """Create DuckDuckGo search tool."""
        search = DuckDuckGoSearchResults(num_results=5)
        return Tool(
            name="web_search",
            func=search.run,
            description="Search the web for current information. Input should be a search query string.",
        )

    def create_wikipedia_tool(self) -> BaseTool:
        """Create Wikipedia search tool."""
        api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
        return WikipediaQueryRun(api_wrapper=api_wrapper)

    async def initialize_default_tools(self) -> None:
        """Initialize all default tools."""
        # Search tools
        search_tool = self.create_search_tool()
        self.register_tool(
            search_tool,
            ToolMetadata(
                name="web_search",
                description="Search the web using DuckDuckGo",
                category=ToolCategory.SEARCH,
                rate_limit_per_minute=30,
            ),
        )

        wikipedia_tool = self.create_wikipedia_tool()
        self.register_tool(
            wikipedia_tool,
            ToolMetadata(
                name="wikipedia",
                description="Search Wikipedia for information",
                category=ToolCategory.SEARCH,
                rate_limit_per_minute=30,
            ),
        )

        # Calculator tool
        calc_tool = self.create_calculator_tool()
        self.register_tool(
            calc_tool,
            ToolMetadata(
                name="calculator",
                description="Evaluate mathematical expressions",
                category=ToolCategory.CALCULATOR,
                rate_limit_per_minute=100,
            ),
        )

        # Browser tools
        browser_tool = await self.create_browser_tool()
        self.register_tool(
            browser_tool,
            ToolMetadata(
                name="browser_navigate",
                description="Navigate to URLs with browser",
                category=ToolCategory.BROWSER,
                rate_limit_per_minute=20,
            ),
        )

        screenshot_tool = await self.create_screenshot_tool()
        self.register_tool(
            screenshot_tool,
            ToolMetadata(
                name="browser_screenshot",
                description="Capture webpage screenshots",
                category=ToolCategory.BROWSER,
                rate_limit_per_minute=10,
            ),
        )


def create_custom_tool(
    name: str,
    description: str,
    func: Callable[[str], str],
    category: ToolCategory = ToolCategory.CUSTOM,
) -> tuple[BaseTool, ToolMetadata]:
    """Create a custom tool with metadata."""
    tool = Tool(name=name, func=func, description=description)

    metadata = ToolMetadata(
        name=name, description=description, category=category, enabled=True
    )

    return tool, metadata


def validate_tool_result(result: str, max_length: int = 10000) -> str:
    """Validate and sanitize tool results."""
    if not result:
        return "Tool returned empty result"

    # Truncate if too long
    if len(result) > max_length:
        return result[:max_length] + f"... (truncated from {len(result)} chars)"

    return result