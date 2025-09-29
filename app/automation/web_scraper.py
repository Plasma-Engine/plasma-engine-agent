"""Web scraping utilities with BeautifulSoup and content extraction."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup, Tag
from loguru import logger
from playwright.async_api import Page


@dataclass
class ScrapedContent:
    """Container for scraped web content."""

    url: str
    title: Optional[str] = None
    text: str = ""
    html: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[List[Dict[str, str]]] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)
    styles: List[str] = field(default_factory=list)


class WebScraper:
    """Web scraping and content extraction utilities."""

    def __init__(self, parser: str = "lxml"):
        """Initialize web scraper.

        Args:
            parser: BeautifulSoup parser to use ("lxml", "html.parser", "html5lib")
        """
        self.parser = parser

    async def scrape_page(
        self,
        page: Page,
        include_html: bool = True,
        include_metadata: bool = True,
        include_links: bool = True,
        include_images: bool = True,
        include_forms: bool = False,
        include_tables: bool = False,
        include_scripts: bool = False,
        include_styles: bool = False,
    ) -> ScrapedContent:
        """Scrape content from a Playwright page.

        Args:
            page: Playwright page to scrape
            include_html: Include raw HTML
            include_metadata: Include meta tags
            include_links: Include links
            include_images: Include images
            include_forms: Include form data
            include_tables: Include table data
            include_scripts: Include script tags
            include_styles: Include style tags

        Returns:
            Scraped content object
        """
        logger.info(f"Scraping page: {page.url}")

        # Get HTML content
        html = await page.content()
        soup = BeautifulSoup(html, self.parser)

        # Extract title
        title = await page.title()

        # Extract text content
        text = self._extract_text(soup)

        # Create content object
        content = ScrapedContent(
            url=page.url,
            title=title,
            text=text,
            html=html if include_html else "",
        )

        # Extract metadata
        if include_metadata:
            content.metadata = self._extract_metadata(soup)

        # Extract links
        if include_links:
            content.links = self._extract_links(soup, base_url=page.url)

        # Extract images
        if include_images:
            content.images = self._extract_images(soup, base_url=page.url)

        # Extract headers
        content.headers = await self._extract_headers(page)

        # Extract forms
        if include_forms:
            content.forms = self._extract_forms(soup)

        # Extract tables
        if include_tables:
            content.tables = self._extract_tables(soup)

        # Extract scripts
        if include_scripts:
            content.scripts = self._extract_scripts(soup)

        # Extract styles
        if include_styles:
            content.styles = self._extract_styles(soup)

        logger.info(f"Scraped {len(content.text)} characters from {page.url}")
        return content

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Cleaned text content
        """
        # Remove script and style elements
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML head.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Standard meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                metadata[name] = content

        # Open Graph tags
        og_tags = {}
        for meta in soup.find_all("meta", property=re.compile("^og:")):
            og_tags[meta.get("property")] = meta.get("content")
        if og_tags:
            metadata["og"] = og_tags

        # Twitter Card tags
        twitter_tags = {}
        for meta in soup.find_all("meta", attrs={"name": re.compile("^twitter:")}):
            twitter_tags[meta.get("name")] = meta.get("content")
        if twitter_tags:
            metadata["twitter"] = twitter_tags

        # JSON-LD structured data
        json_ld = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json

                json_ld.append(json.loads(script.string))
            except Exception as e:
                logger.warning(f"Failed to parse JSON-LD: {e}")
        if json_ld:
            metadata["json_ld"] = json_ld

        return metadata

    def _extract_links(
        self, soup: BeautifulSoup, base_url: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Extract all links from HTML.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links

        Returns:
            List of link dictionaries
        """
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href")
            text = a_tag.get_text(strip=True)
            title = a_tag.get("title", "")

            # Resolve relative URLs if base_url provided
            if base_url and href:
                from urllib.parse import urljoin

                href = urljoin(base_url, href)

            links.append({"href": href, "text": text, "title": title})

        return links

    def _extract_images(
        self, soup: BeautifulSoup, base_url: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Extract all images from HTML.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs

        Returns:
            List of image dictionaries
        """
        images = []
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src")
            alt = img_tag.get("alt", "")
            title = img_tag.get("title", "")
            width = img_tag.get("width", "")
            height = img_tag.get("height", "")

            # Resolve relative URLs if base_url provided
            if base_url and src:
                from urllib.parse import urljoin

                src = urljoin(base_url, src)

            images.append(
                {
                    "src": src,
                    "alt": alt,
                    "title": title,
                    "width": width,
                    "height": height,
                }
            )

        return images

    async def _extract_headers(self, page: Page) -> Dict[str, str]:
        """Extract HTTP response headers.

        Args:
            page: Playwright page

        Returns:
            Dictionary of headers
        """
        # Get the response from the page
        response = await page.goto(page.url, wait_until="commit")
        if response:
            return response.headers
        return {}

    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract form data from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of form dictionaries
        """
        forms = []
        for form_tag in soup.find_all("form"):
            form_data = {
                "action": form_tag.get("action", ""),
                "method": form_tag.get("method", "GET").upper(),
                "name": form_tag.get("name", ""),
                "id": form_tag.get("id", ""),
                "fields": [],
            }

            # Extract input fields
            for input_tag in form_tag.find_all(["input", "textarea", "select"]):
                field = {
                    "type": input_tag.get("type", "text"),
                    "name": input_tag.get("name", ""),
                    "id": input_tag.get("id", ""),
                    "value": input_tag.get("value", ""),
                    "placeholder": input_tag.get("placeholder", ""),
                    "required": input_tag.has_attr("required"),
                }

                # Extract options for select elements
                if input_tag.name == "select":
                    options = []
                    for option in input_tag.find_all("option"):
                        options.append(
                            {
                                "value": option.get("value", ""),
                                "text": option.get_text(strip=True),
                            }
                        )
                    field["options"] = options

                form_data["fields"].append(field)

            forms.append(form_data)

        return forms

    def _extract_tables(self, soup: BeautifulSoup) -> List[List[Dict[str, str]]]:
        """Extract table data from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of tables (each table is a list of row dictionaries)
        """
        tables = []
        for table_tag in soup.find_all("table"):
            rows = []

            # Extract headers
            headers = []
            header_row = table_tag.find("thead")
            if header_row:
                for th in header_row.find_all("th"):
                    headers.append(th.get_text(strip=True))

            # Extract data rows
            tbody = table_tag.find("tbody") or table_tag
            for tr in tbody.find_all("tr"):
                row = {}
                cells = tr.find_all(["td", "th"])

                if not headers and cells:
                    # Use cell index as key if no headers
                    for i, cell in enumerate(cells):
                        row[f"col_{i}"] = cell.get_text(strip=True)
                else:
                    # Use headers as keys
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            row[headers[i]] = cell.get_text(strip=True)

                if row:
                    rows.append(row)

            if rows:
                tables.append(rows)

        return tables

    def _extract_scripts(self, soup: BeautifulSoup) -> List[str]:
        """Extract JavaScript from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of script contents
        """
        scripts = []
        for script_tag in soup.find_all("script"):
            if script_tag.string:
                scripts.append(script_tag.string.strip())
            src = script_tag.get("src")
            if src:
                scripts.append(f"// External script: {src}")

        return scripts

    def _extract_styles(self, soup: BeautifulSoup) -> List[str]:
        """Extract CSS from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of style contents
        """
        styles = []
        for style_tag in soup.find_all("style"):
            if style_tag.string:
                styles.append(style_tag.string.strip())

        for link_tag in soup.find_all("link", rel="stylesheet"):
            href = link_tag.get("href")
            if href:
                styles.append(f"/* External stylesheet: {href} */")

        return styles

    def extract_by_selector(
        self, html: str, selector: str, attribute: Optional[str] = None
    ) -> List[Union[str, Dict[str, Any]]]:
        """Extract elements by CSS selector.

        Args:
            html: HTML content
            selector: CSS selector
            attribute: Optional attribute to extract (if None, returns text)

        Returns:
            List of extracted values
        """
        soup = BeautifulSoup(html, self.parser)
        elements = soup.select(selector)

        results = []
        for element in elements:
            if attribute:
                value = element.get(attribute)
                if value:
                    results.append(value)
            else:
                results.append(element.get_text(strip=True))

        return results

    def extract_by_xpath(self, html: str, xpath: str) -> List[str]:
        """Extract elements by XPath expression.

        Args:
            html: HTML content
            xpath: XPath expression

        Returns:
            List of extracted text values
        """
        from lxml import etree

        tree = etree.HTML(html)
        elements = tree.xpath(xpath)

        results = []
        for element in elements:
            if isinstance(element, str):
                results.append(element)
            elif hasattr(element, "text"):
                results.append(element.text or "")

        return results

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Normalize line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text