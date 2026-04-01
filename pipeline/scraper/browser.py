"""
Shared Playwright browser session manager.

A single BrowserContext is reused for the entire crawl run.
Call `init_browser()` once at startup and `close_browser()` at shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from config.settings import settings

logger = logging.getLogger(__name__)

_playwright: Playwright | None = None
_browser: Browser | None = None
_context: BrowserContext | None = None

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_BLOCKED_RESOURCE_TYPES = {"image", "font", "media"}


async def init_browser(block_resources: bool = True) -> None:
    """Launch Playwright Chromium and create a shared BrowserContext."""
    global _playwright, _browser, _context
    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(
        headless=settings.HEADLESS,
        slow_mo=settings.PLAYWRIGHT_SLOW_MO,
    )
    _context = await _browser.new_context(
        user_agent=_USER_AGENT,
        locale="ru-RU",
        timezone_id="Asia/Almaty",
    )
    _context.set_default_timeout(settings.BROWSER_TIMEOUT_MS)

    if block_resources:
        await _context.route(
            "**/*",
            lambda route, req: (
                route.abort()
                if req.resource_type in _BLOCKED_RESOURCE_TYPES
                else route.continue_()
            ),
        )
    logger.info("Browser initialised (headless=%s)", settings.HEADLESS)


async def close_browser() -> None:
    """Close the shared browser and stop the Playwright subprocess."""
    global _playwright, _browser, _context
    if _context:
        await _context.close()
        _context = None
    if _browser:
        await _browser.close()
        _browser = None
    if _playwright:
        await _playwright.stop()
        _playwright = None
    logger.info("Browser closed")


@asynccontextmanager
async def get_page() -> AsyncGenerator[Page, None]:
    """Yield a new Page from the shared context. Closes the page on exit."""
    if _context is None:
        raise RuntimeError("Browser not initialised. Call init_browser() first.")
    page = await _context.new_page()
    try:
        yield page
    finally:
        await page.close()


async def fetch_html(url: str, wait_until: str = "domcontentloaded") -> str:
    """Navigate to `url` and return the full rendered HTML."""
    async with get_page() as page:
        response = await page.goto(url, wait_until=wait_until)
        if response and response.status >= 400:
            raise ValueError(f"HTTP {response.status} fetching {url}")
        content = await page.content()
        if len(content) < 500:
            logger.warning("Suspiciously short response (%d chars) for %s", len(content), url)
        return content
