"""Browser automation module for web scraping and interaction."""

from .browser_manager import BrowserManager, BrowserConfig, BrowserType
from .web_scraper import WebScraper, ScrapedContent

__all__ = ["BrowserManager", "BrowserConfig", "BrowserType", "WebScraper", "ScrapedContent"]