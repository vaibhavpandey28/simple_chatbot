import os
import re
from urllib.parse import quote_plus, urlparse

import requests
from dotenv import load_dotenv

from core.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

MARKET_SEARCH_ENABLED = os.getenv("MARKET_SEARCH_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MARKET_SEARCH_TIMEOUT = float(os.getenv("MARKET_SEARCH_TIMEOUT", "8"))
MARKET_SEARCH_MAX_RESULTS = int(os.getenv("MARKET_SEARCH_MAX_RESULTS", "5"))
MARKET_ALLOWED_DOMAINS = [
    d.strip().lower()
    for d in os.getenv(
        "MARKET_ALLOWED_DOMAINS",
        "amazon.in,flipkart.com,myntra.com,ajio.com,nykaa.com,tatacliq.com",
    ).split(",")
    if d.strip()
]

_DDG_LITE = "https://lite.duckduckgo.com/lite/"


def _is_domain_allowed(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return any(host == d or host.endswith(f".{d}") for d in MARKET_ALLOWED_DOMAINS)


def _extract_price(text: str) -> str:
    # Cheap best-effort price matcher (INR/USD variants)
    m = re.search(r"((?:₹|rs\.?|inr|\$)\s?\d[\d,]*(?:\.\d{1,2})?)", text, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def search_market_web(query: str, limit: int = 5) -> list[dict]:
    """Best-effort market lookup using DuckDuckGo Lite results filtered to ecommerce domains."""
    if not MARKET_SEARCH_ENABLED:
        return []

    q = (query or "").strip()
    if not q:
        return []

    max_results = max(1, min(int(limit or 5), MARKET_SEARCH_MAX_RESULTS, 10))
    domain_part = " OR ".join([f"site:{d}" for d in MARKET_ALLOWED_DOMAINS])
    composed_query = f"{q} ({domain_part})"

    try:
        resp = requests.get(
            f"{_DDG_LITE}?q={quote_plus(composed_query)}",
            timeout=MARKET_SEARCH_TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
    except Exception:
        logger.exception("Market search request failed")
        return []

    html = resp.text or ""
    # DuckDuckGo lite uses anchors for results; keep parser simple and resilient.
    anchors = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE | re.DOTALL)

    results: list[dict] = []
    for href, raw_title in anchors:
        title = re.sub(r"<.*?>", "", raw_title)
        title = re.sub(r"\s+", " ", title).strip()
        if not href.startswith("http"):
            continue
        if not _is_domain_allowed(href):
            continue

        price = _extract_price(title)
        source = (urlparse(href).hostname or "").lower()
        results.append(
            {
                "title": title,
                "price": price,
                "url": href,
                "source": source,
                "availability": "unknown",
            }
        )
        if len(results) >= max_results:
            break

    return results
