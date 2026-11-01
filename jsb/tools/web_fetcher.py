"""
JSBrother — Web Fetcher Tool
Best-effort HTML extraction for online job and project postings using requests + BeautifulSoup,
with primary fallback to raw pasted text per §4 and §11 (M3).
"""

import re
import logging
from typing import Union
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

logger = logging.getLogger("jsbrother.web_fetcher")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def is_valid_url(text: str) -> bool:
    """Checks if a string is a standard HTTP/HTTPS URL."""
    clean = (text or "").strip()
    return clean.startswith("http://") or clean.startswith("https://")


def fetch_url_text(url: str, timeout: int = 10) -> str:
    """
    Fetches web page content, strips HTML tags, scripts, navigation, and footers,
    and returns clean readable text.
    Raises RuntimeError on connection errors or blocking.
    """
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch posting from URL '{url}': {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg"]):
        tag.decompose()

    # Get body or main content
    main_content = soup.find("main") or soup.find("article") or soup.find("body") or soup
    text = main_content.get_text(separator="\n")

    # Collapse blank lines and whitespace
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)

    if not clean_text:
        raise RuntimeError(f"URL '{url}' returned empty page text.")

    return clean_text


def fetch_posting_or_text(source: Union[str, Path]) -> str:
    """
    Primary entry point:
    1. If source is a file path to an existing .txt file, reads and returns it.
    2. If source is an HTTP/HTTPS URL, attempts best-effort web fetching.
    3. Otherwise, returns source directly as raw pasted text.
    """
    if isinstance(source, Path):
        if source.is_file():
            return source.read_text(encoding="utf-8")
        source = str(source)

    text_input = (source or "").strip()
    if not text_input:
        return ""

    # Check if existing local file
    try:
        potential_path = Path(text_input)
        if potential_path.is_file():
            return potential_path.read_text(encoding="utf-8")
    except Exception:
        pass

    # Check if URL
    if is_valid_url(text_input):
        logger.info(f"Fetching job posting from URL: {text_input}")
        return fetch_url_text(text_input)

    # Pasted text
    return text_input


class WebFetcherInput(BaseModel):
    source: str = Field(
        description="URL of the job/project posting, local text file path, or raw pasted posting text."
    )


web_fetcher_tool = StructuredTool.from_function(
    func=fetch_posting_or_text,
    name="web_fetcher",
    description="Fetches text from a job posting URL or returns raw pasted text.",
    args_schema=WebFetcherInput,
)
